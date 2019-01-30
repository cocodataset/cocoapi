root_dir=$1
singularity_image=$2
conffile=$3

if test $(which singularity); then 
    singularity exec -B /mnt:/mnt -s /usr/bin/zsh --no-home --nv ${root_dir}/../../${singularity_image} zsh -ic "python -u detection_train.py --config config/${conffile}"
else 
    singularity exec -B /mnt:/mnt -s /usr/bin/zsh --no-home --nv ${root_dir}/../../${singularity_image} zsh -ic "python -u detection_train.py"
fi
