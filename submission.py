# Feel free to modifiy this file.

from torchvision import models, transforms

team_id = 11
team_name = "Neural Crusaders"
email_address = "gg2501@nyu.edu"

def get_model():
    return models.resnet34(num_classes=800)

eval_transform = transforms.Compose([
    transforms.ToTensor(),
])