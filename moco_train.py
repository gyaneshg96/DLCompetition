import os
import glob
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import lightly
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np

from pytorch_lightning.metrics import Metric



num_workers = 8
batch_size = 256
memory_bank_size = 4096
seed = 1
max_epochs = 5

pl.seed_everything(seed)

# MoCo v2 uses SimCLR augmentations, additionally, disable blur
collate_fn = lightly.data.SimCLRCollateFunction(
    input_size=96,
    gaussian_blur=0.,
)

# Augmentations typically used to train on cifar-10
train_classifier_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.RandomCrop(96, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])

# No additional augmentations for the test set
test_transforms = torchvision.transforms.Compose([
    # torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=lightly.data.collate.imagenet_normalize['mean'],
        std=lightly.data.collate.imagenet_normalize['std'],
    )
])


class MocoModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # create a ResNet backbone and remove the classification head
        # resnet = lightly.models.ResNetGenerator('resnet-18', 1, num_classes=800, num_splits=8)
        resnet = torchvision.models.resnet34(pretrained=False)
        backbone = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.AdaptiveAvgPool2d(1),
        )

        # create a moco based on ResNet
        self.resnet_moco = \
            lightly.models.MoCo(backbone, num_ftrs=512, out_dim=1024, m=0.99, batch_shuffle=True)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1,
            memory_bank_size=memory_bank_size)

    def forward(self, x):
        self.resnet_moco(x)

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        (x0, x1), _, _ = batch
        y0, y1 = self.resnet_moco(x0, x1)
        loss = self.criterion(y0, y1)
        self.log('train_loss_ssl', loss, sync_dist=True)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()


    def configure_optimizers(self):
        optim = torch.optim.SGD(self.resnet_moco.parameters(), lr=6e-2,
                                momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

class MyAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, outputs: torch.Tensor, target: torch.Tensor):
        # outputs, target = self._input_format(outputs, target)
        preds = torch.argmax(nn.Softmax(dim=1)(outputs), dim=1)
        # print(preds)
        # print(target)
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        return self.correct.float() / self.total


class Classifier(pl.LightningModule):
    def __init__(self, model, lr=0.1):
        super().__init__()
        # create a moco based on ResNet
        self.resnet_moco = model
        self.lr = lr

        # freeze the layers of moco
        for p in self.resnet_moco.parameters():  # reset requires_grad
            p.requires_grad = False

        # we create a linear layer for our downstream classification
        # model
        self.fc = nn.Sequential(
            nn.Linear(512,2048),
            nn.ReLU(),
            nn.Linear(2048,800)
        )

        self.accuracy = MyAccuracy()

    def forward(self, x):
        with torch.no_grad():
            y_hat = self.resnet_moco.backbone(x).squeeze()
            y_hat = nn.functional.normalize(y_hat, dim=1)
        y_hat = self.fc(y_hat)
        return y_hat

    # We provide a helper method to log weights in tensorboard
    # which is useful for debugging.
    def custom_histogram_weights(self):
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(
                name, params, self.current_epoch)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss_fc', loss)
        self.accuracy(y_hat, y)
        self.log('train_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)
        return loss

    def training_epoch_end(self, outputs):
        self.custom_histogram_weights()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        self.accuracy(y_hat, y)
        self.log('val_acc', self.accuracy.compute(),
                 on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.fc.parameters(), lr=self.lr, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 400)
        return [optim], [scheduler]

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transform):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /dataset
            split: The split you want to used, it should be one of train, val or unlabeled.
            transform: the transform you want to applied to the images.
        """

        self.split = split
        self.transform = transform

        self.image_dir = os.path.join(root, split)
        label_path = os.path.join(root, f"{split}_label_tensor.pt")

        self.num_images = len(os.listdir(self.image_dir))

        if os.path.exists(label_path):
            self.labels = torch.load(label_path)
        else:
            self.labels = -1 * torch.ones(self.num_images, dtype=torch.long)

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        with open(os.path.join(self.image_dir, f"{idx}.png"), 'rb') as f:
            img = Image.open(f).convert('RGB')

        return self.transform(img), self.labels[idx]

# train_dataset = CustomDataset(root="/dataset/", split="train", transform=train_classifier_transforms)
# test_dataset = CustomDataset(root="/dataset", split="val", transform=test_transforms)

dataset_train_moco = lightly.data.LightlyDataset(
    input_dir="/dataset/unlabeled"
)

print("Dataset Loaded")

dataloader_train_moco = torch.utils.data.DataLoader(
    dataset_train_moco,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    drop_last=True,
    num_workers=num_workers
)

# dataloader_train_classifier = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=batch_size,
#     shuffle=True,
#     drop_last=True,
#     num_workers=num_workers
# )

# dataloader_test = torch.utils.data.DataLoader(
#     test_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     drop_last=False,
#     num_workers=num_workers
# )


model = MocoModel.load_from_checkpoint(checkpoint_path="/scratch/gg2501/moco/example.ckpt")
# model = MocoModel()

max_epochs = 80

#gpus = 1 if torch.cuda.is_available() else 0

trainer = pl.Trainer(precision=16, accelerator='dp', max_epochs=max_epochs, gpus=-1,
                     progress_bar_refresh_rate=100,
                     default_root_dir='/scratch/gg2501/moco/')

print("Begin training")

trainer.fit(model, dataloader_train_moco)

trainer.save_checkpoint("/scratch/gg2501/moco/final_moco.ckpt")




