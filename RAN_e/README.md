# RAN-e
This repository provides scripts and recipe to train hardware-efficient RAN-e-GT and RAN-e-C models as described in the paper [Restructurable Activation Networks](https://arxiv.org/abs/2208.08562).
> **NOTE:**
This codebase has been built by leveraging and modifying code from the project [NVIDIA/DeepLearningExamples](https://github.com/NVIDIA/DeepLearningExamples).
The original unmodified code can be found at this [commit](https://github.com/NVIDIA/DeepLearningExamples/tree/ec2bef8444c4c0dad9269bdb4164610fd8a91570/TensorFlow2/Classification/ConvNets/efficientnet).


## Overview
Contained are 2 types of models to train:

- `RAN-e-GT`
- `RAN-e-C`

Relevant shell scripts for training the networks as well as environment set-up can be found in the ```./scripts``` directory. These convenience-scripts offered to train and evaluate are tailored to 8-GPU machines specifically.

## Quick Start
- First set up you environment using the helper script:
```bash ./scripts/env_setup.sh```

- To train one of the two models, simply execute:
```bash ./scripts/train_RAN_e_{C, GT}_350ep.sh```

- To evaluate a pretrained model, similarly execute:
```bash ./scripts/eval_RAN_e_{C, GT}.sh```

Taking care to ensure the `MODEL_DIR`, `DATA_DIR` and `INDX` variables in the scripts are set correctly for your particular environment.

## Finer Details
The following section offers more context to the model structure and parameter-values used.

### Scripts and sample code

The following lists the content for each folder:
- `scripts/` - shell scripts to train and evaluate RAN-e models
- `model/` - building blocks and RAN-e model definitions
- `runtime/` - detailed procedure for each running mode
- `utils/` - support util functions for `runner.py`

### Parameters

Important parameters for training are listed below

- `mode` {`train_and_eval`,`train`,`eval`}
- `arch` - {`ran_e_GT-b0`, `ran_e_C-b0`}
- `model_dir` - The folder where model checkpoints are saved
- `data_dir` - The folder where data resides
- `augmenter_name` - Type of Augmentation
- `max_epochs` - The number of training epochs
- `warmup_epochs` - The number of epochs of warmup
- `train_batch_size` - The training batch size per GPU
- `eval_batch_size` - The evaluation batch size per GPU
- `lr_init` - The learning rate for a batch size of 128, effective learning rate will be automatically scaled according to the global training batch size

The main script `main.py` specific parameters are:
```
 --model_dir MODEL_DIR
                        The directory where the model and training/evaluation
                        summariesare stored.
  --save_checkpoint_freq SAVE_CHECKPOINT_FREQ
                        Number of epochs to save checkpoint.
  --data_dir DATA_DIR   The location of the input data. Files should be named
                        `train-*` and `validation-*`.
  --mode MODE           Mode to run: `train`, `eval`, `train_and_eval`, `predict` or
                        `export`.
  --arch ARCH           The type of the model, e.g. RAN-e-GT, etc.
  --dataset DATASET     The name of the dataset, e.g. ImageNet, etc.
  --log_steps LOG_STEPS
                        The interval of steps between logging of batch level
                        stats.
  --use_xla             Set to True to enable XLA
  --use_amp             Set to True to enable AMP
  --num_classes NUM_CLASSES
                        Number of classes to train on.
  --batch_norm BATCH_NORM
                        Type of Batch norm used.
  --activation ACTIVATION
                        Type of activation to be used.
  --optimizer OPTIMIZER
                        Optimizer to be used.
  --moving_average_decay MOVING_AVERAGE_DECAY
                        The value of moving average.
  --label_smoothing LABEL_SMOOTHING
                        The value of label smoothing.
  --max_epochs MAX_EPOCHS
                        Number of epochs to train.
  --num_epochs_between_eval NUM_EPOCHS_BETWEEN_EVAL
                        Eval after how many steps of training.
  --steps_per_epoch STEPS_PER_EPOCH
                        Number of steps of training.
  --warmup_epochs WARMUP_EPOCHS
                        Number of steps considered as warmup and not taken
                        into account for performance measurements.
  --lr_init LR_INIT     Initial value for the learning rate.
  --lr_decay LR_DECAY   Type of LR Decay.
  --lr_decay_rate LR_DECAY_RATE
                        LR Decay rate.
  --lr_decay_epochs LR_DECAY_EPOCHS
                        LR Decay epoch.
  --weight_decay WEIGHT_DECAY
                        Weight Decay scale factor.
  --weight_init {fan_in,fan_out}
                        Model weight initialization method.
  --train_batch_size TRAIN_BATCH_SIZE
                        Training batch size per GPU.
  --augmenter_name AUGMENTER_NAME
                        Type of Augmentation during preprocessing only during
                        training.
  --eval_batch_size EVAL_BATCH_SIZE
                        Evaluation batch size per GPU.
  --resume_checkpoint   Resume from a checkpoint in the model_dir.
  --use_dali            Use dali for data loading and preprocessing of train
                        dataset.
  --use_dali_eval       Use dali for data loading and preprocessing of eval
                        dataset.
  --dtype DTYPE         Only permitted
                        `float32`,`bfloat16`,`float16`,`fp32`,`bf16`
```
