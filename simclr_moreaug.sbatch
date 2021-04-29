#!/bin/bash

<<<<<<< HEAD
#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
=======
#SBATCH --gres=gpu:2
#SBATCH --partition=n1s16-t4-2
>>>>>>> a6c1a9a6d581741d1d7445b48e6c38e67862d2a8
#SBATCH --account=dl11
#SBATCH --time=24:00:00
#SBATCH --output=demo_%j.out
#SBATCH --error=demo_%j.err
#SBATCH --exclusive
#SBATCH --requeue

/share/apps/local/bin/p2pBandwidthLatencyTest > /dev/null 2>&1

set -x

mkdir /tmp/$USER
export SINGULARITY_CACHEDIR=/tmp/$USER

cp -rp /scratch/DL21SP/student_dataset.sqsh /tmp
echo "Dataset is copied to /tmp"

cd $HOME/DLCompetition

singularity exec --nv \
--bind /scratch \
--overlay $SCRATCH/conda.ext3:rw \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
<<<<<<< HEAD
python SimCLR/run.py --fp16-precision -a resnet18 \
        -b 256 -data /dataset/ -dataset-name dldataset_aug7 \
        -checkpoint-dir $SCRATCH/simclr/ --checkpoint-step 5 \
        --log-every-n-steps 200 --epochs 50
=======
python SimCLR/run.py --multi-gpu --fp16-precision -a resnet18 \
        -b 2048 -data /dataset/ -dataset-name dldataset_aug7 \
        -checkpoint-dir $SCRATCH/simclr/ --checkpoint-step 5 \
        --log-every-n-steps 200 --epochs 100
>>>>>>> a6c1a9a6d581741d1d7445b48e6c38e67862d2a8
"