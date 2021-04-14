#!/bin/bash

#SBATCH --gres=gpu:2
#SBATCH --partition=n1s16-t4-2
#SBATCH --account=dl11
#SBATCH --time=1:00:00
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
python SimCLR/run.py --multi-gpu --fp16-precision -a resnet50 \
        -b 256 -data /dataset/ -dataset-name dldataset \
        -checkpoint-dir $SCRATCH/simclr/ --checkpoint-step 5 \
        --log-every-n-steps 200 --epochs 1
"
