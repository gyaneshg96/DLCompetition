#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition=n1s8-t4-1
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
--overlay /scratch/gg2501/conda.ext3:rw \
--overlay /tmp/student_dataset.sqsh:ro \
/share/apps/images/cuda11.1-cudnn8-devel-ubuntu18.04.sif \
/bin/bash -c "
source /ext3/env.sh
conda activate dev
python FixMatch-pytorch/train.py --dataset dldataset --arch wideresnet --total-steps 32768 --wdecay 0.001 --expand-labels --batch-size 16 --out $SCRATCH/checkpoints/fixmatch/
python eval.py --checkpoint-path $SCRATCH/checkpoints/fixmatch/fixmatch.pth
"
