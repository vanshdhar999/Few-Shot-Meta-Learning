# Few Shot Meta Learning

<img src="https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?&style=for-the-badge&logo=PyTorch&logoColor=white" />  <a href="https://wandb.ai/meta-learners/projects" alt="W&B Dashboard">  <img src="https://img.shields.io/badge/WandB-Dashboard-gold.svg" /></a> <a href="https://dagshub.com/arjun2000ashok/FSL-SSL" alt="DAGsHub Dashboard"><img src="https://img.shields.io/badge/DAGsHub-Project-blue.svg" /></a>

# Table of contents

**Please click each of the sections below to jump to its contents.**

1. [Installation Instructions](#installation)
2. [Datasets](#datasets)
3. [Training & Inference](#training)
     
     3.1. [General Instructions](#general)

     3.2. [Examples](#examples)

<div id='installation' />

# 1. Installation Instructions

Execute this to install all the dependencies:

```pip install -r requirements.txt```

<div id='datasets' />

# 2. Datasets

In total, we support a total of 10 datasets for training and inference. Apart from this, we support one more dataset for the domain selection experiments.

The datasets, details, download links, location are below:

| Dataset          | Download Link                                              | Extraction Location               |
| ---------------- | ---------------------------------------------------------- | --------------------------------- |
| VGG flowers      | https://www.kaggle.com/arjun2000ashok/vggflowers/          | `filelists/flowers/images`      |

<br>

<div id='training' />

# 3. Training & Inference

<div id='general' />

## 3.1 General Instructions

**All scripts must be executed from the `root` folder**

The `train.py` file is used for training, validation and testing.

It trains the few-shot model for a fixed number of episodes, with periodic evalution on the validation set, followed by testing on the test set.

Note that all the results reported are based on training for a fixed number of epochs, and then evaluating using the best model found using the validation set.

Please see `utils/io_utils.py` for all the arguments and their default values. Here are some sufficient examples:

`python train.py --help` will print the help for all the necessary arguments:

```
usage: train.py [-h] [--dataset DATASET] [--model MODEL] [--method METHOD]
                [--train_n_way TRAIN_N_WAY] [--test_n_way TEST_N_WAY]
                [--n_shot N_SHOT] [--train_aug [TRAIN_AUG]]
                [--jigsaw [JIGSAW]] [--lbda LBDA] [--lr LR]
                [--optimization OPTIMIZATION] [--loadfile LOADFILE]
                [--finetune] [--random] [--n_query N_QUERY]
                [--image_size IMAGE_SIZE] [--debug] [--json_seed JSON_SEED]
                [--date DATE] [--rotation [ROTATION]] [--grey]
                [--low_res [LOW_RES]] [--firstk FIRSTK] [--testiter TESTITER]
                [--wd WD] [--bs BS] [--iterations ITERATIONS] [--useVal]
                [--scheduler [SCHEDULER]] [--lbda_jigsaw LBDA_JIGSAW]
                [--lbda_rotation LBDA_ROTATION] [--pretrain [PRETRAIN]]
                [--dataset_unlabel DATASET_UNLABEL]
                [--dataset_unlabel_percentage DATASET_UNLABEL_PERCENTAGE]
                [--dataset_percentage DATASET_PERCENTAGE] [--bn_type BN_TYPE]
                [--test_bs TEST_BS] [--split SPLIT] [--save_iter SAVE_ITER]
                [--adaptation] [--device DEVICE] [--seed SEED] [--amp [AMP]]
                [--num_classes NUM_CLASSES] [--save_freq SAVE_FREQ]
                [--start_epoch START_EPOCH] [--stop_epoch STOP_EPOCH]
                [--resume] [--warmup] [--eval_interval EVAL_INTERVAL]
                [--run_name RUN_NAME] [--run_id RUN_ID]
                [--semi_sup [SEMI_SUP]] [--sup_ratio SUP_RATIO]
                [--only_test [ONLY_TEST]] [--project PROJECT]
                [--save_model [SAVE_MODEL]] [--demo [DEMO]]
                [--only_train [ONLY_TRAIN]] [--sweep [SWEEP]]
```

`--device` is used to specify the GPU device.
`--seed` is used to specify the seed. The default 
`--train_aug` is used to specify if there should be data augmentation. All results in the paper and the report are done with data augmentation. It is by default `True`.
`--stop_epoch` is used the number of epochs. It is recommended to run the small dataset models for 500 epochs, and the miniImageNet models for 700 epochs. The best model picked by validation will be evaluated at the end.
`--lr` is the learning rate. 

`--loadfile` can be used to provide a path for loading a pretrained model. Note that the model should be of the same architecture as given in the `--model` argument. 


`--only_train` can be used to only train the models, and stop before testing them.
`--only_test` can be used to only test the models. Note that a model path needs to be provided if you are testing a pretrained model.

`--resume` can be used to resume a run. If `--resume` is provided, the `--run_id` must also be provided to resume the corresponding W&B run. By default, as `line 362` of `train.py` indicates, the last model will be retrieved from W&B automatically, and loaded. Note that I have saved the epoch also, and hence the `--start_epoch` will be automatically set. The `--stop_epoch` must be provided in all cases.

`--bn_type` can be set to `1`, `2` or `3` to set the respective type of batch norm used.
  
`NUM_WORKERS` for all the dataloaders can be set at `config/configs.py`.


<div id='examples' />

## 3.2 Examples

**NOTE**: For the exact commands used, you can refer to each run in the respective project in the W&B dashboard. We provide a detailed list of all the possible configurations below. 
  
### Supervised training

For training a ProtoNet 3-way 5-shot model on Flowers dataset with resnet18 and image size 224, for 600 epochs (LR=0.01):

`python train.py --dataset=flowers --model=resnet18 --method=protonet --train_n_way=5 --test_n_way=5 --n_shot=5 --train_aug --stop_epoch=600 --lr=0.01`.
 

<div id='hyperparams' />
								

`Res` refers to resnet-18.																			

This README file was edited according to our purpose and the original repository deals with more than what was required.
