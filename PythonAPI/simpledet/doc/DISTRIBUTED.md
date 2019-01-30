#### Requirement
Here we only provide a guide to launch distributed training with singularity, please make sure your singularity works by checking [INSTALL.md](./doc/INSTALL.md)

#### Setup
1. obtain the mxnet launcher and place it in the parent directory of the simpledet working directory
```bash
git clone https://github.com/RogerChern/mxnet-dist-lancher.git lancher
```

2. mv `data`, `pretrain_model`, `experiments` outside of simpledet and symink them back.
This step is to avoid unnecessary `rsync` of large binary files in the working directory during launching.

3. after step 1 and 2, your directory should be as following
```
lancher/
simpledet/
  data -> /path/to/data
  pretrain_model -> /path/to/pretain_model
  experiments -> /path/to/experiments
  ...
```

4. make a hostfile containing hostnames of all nodes, these nodes would be accessed from our launch node by ssh without password
simpledet/hostfile.txt
```
node1
node2
```

5. change the singulariy mounting point in `scripts/dist_worker.sh`

6. change working directories in `scritps/train_hpc.sh`

7. launch distributed training with scripts
```bash
bash scritps/train_hpc.sh
```
