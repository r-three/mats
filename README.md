# MaTS
This repository contains the official code for the paper: "Merging by Matching Models in Task Subspaces".

## Setup

1. Create a virtual environment and activate it.
```
python3.8 -m venv env
source env/bin/activate
```
2. Install dependencies 
```
python -m pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
3. Set environment variables (This step has to be done every session)

`HUGGINGFACE_HUB_CACHE` path can change depending on your setup. 
```
source bin/setup.sh {DEVICE_ID}
export HUGGINGFACE_HUB_CACHE=~/.cache 
```

## Training 

Run the `training` script with 

`-c` the list of configs for the model

`-td {key}={value}` any training dataset config parameters to update. 

`-ed {key}={value}` any evaluation dataset config parameters to update. 

`-tr {key}={value}` any training run config parameters to update. 

`-er {key}={value}` any evaluation run config parameters to update. 

`-m {key}={value}` any model config parameters to update.

`-w {number of GPUs}` for distributed training 

### Full-Model Fine-Tuning 

T5-Large for Single Task (`Paws`)
```
python src/training.py -c configs/models/t5_large.json configs/training_run/individual_task_T0_run.json  configs/training_dataset/p3_individual_task.json configs/evaluation_dataset/p3_validation.json configs/evaluation_run/individual_task.json -td train_dataset=paws -ed evaluation_dataset=paws  -tr micro_train_batch_size=16  -er eval_batch_size=32
```
T5-Large for Multiple Tasks (`p3_eight_qa` dataset mixture)
```
python src/training.py -c configs/models/t5_large.json configs/training_run/p3_eight_qa_T0_run.json  configs/training_dataset/p3_eight_qa.json configs/evaluation_dataset/p3_validation.json configs/evaluation_run/p3_eight_qa.json -tr micro_train_batch_size=16   -er eval_batch_size=32 
```

### Parameter-Efficient Fine-tuning

$(IA)^3$  on a Single Task (`Paws`)  
```
python src/training.py -c configs/models/t5_large.json configs/models/ia3.json configs/training_run/individual_task_T0_run_ia3.json configs/training_dataset/p3_individual_task.json configs/evaluation_dataset/p3_validation.json configs/evaluation_run/individual_task.json -td train_dataset=paws -ed evaluation_dataset=paws  -tr micro_train_batch_size=16  -er eval_batch_size=32 
```

$(IA)^3$   on Multiple Tasks (`p3_eight_qa` dataset mixture)
```
python src/training.py -c configs/models/t5_large.json configs/models/ia3.json configs/training_run/p3_eight_qa_T0_run_ia3.json  configs/training_dataset/p3_eight_qa.json  configs/evaluation_dataset/p3_validation.json configs/evaluation_run/p3_eight_qa.json   -tr micro_train_batch_size=16   -er eval_batch_size=32 
```

## Merging

All the examples below are for merging models that were trained on datasets from the `p3_eight_qa` dataset mixture. 

The results will be saved in 
```
exp_out/merging/{instruction_format}/{dataset_mixture}/{pretrained_model}/{checkpoint_descriptor}/{merging_method}/
``` 
For example, conjugate gradients for $(IA)^3$ with the blockwise Fisher merging objective is stored at 
```
exp_out/merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/conjugate_gradients/fisher_merging_blockwise_empirical_fisher_validation_initialize_task_vectors_model_lambda_1.0_iterations_100/
```

The arguments are the same as the arguments for training, along with the following generic arguments. 

`--model_lambda` lambda for merging. The exact purpose of lambda depends on the merging method. Default to iterating over values from 0 to 1 with a step size of 0.1 

`-d {dataset_mixture}` dataset mixture of datasets to evaluate.

`--checkpoint_descriptor {checkpoint_descriptor}` key which stores the filepaths of which checkpoint to use for each model. The filepaths are in `src/merging/utils.checkpoint_filepaths.py` in a dictionary of the format: 
```
{pretrained_model_name: {
    instruction_format: {
        checkpoint_decriptor: {
            dataset: filepath,
            dataset: filepath, 
            .
            .
            .
        }
    }
}}
```

Methods involving computing some metadata (i.e. diagonal Fisher merging, RegMean, and the conjugate gradient method with the RegMean objective or blockwise Fisher merging objective) also have the following arguments. 

`--split {split}` which split to compute the Fisher on 

`--use_true_fisher` whether to use the true or empirical Fisher. Defaults to using the empirical Fisher 

`-f {fisher_approximation}` which Fisher approximation to use - either `diagonal` or `blockwise`. Note this argument is only for diagonal Fisher merging or the conjugate gradient method with the blockwise Fisher merging objective. 

### Evaluating each individual datasets 
- Full-Model Fine-tuning 
```
python src/merging/individual_models.py   -c configs/models/t5_large.json  configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32
```
- $(IA)^3$ 
```
python src/merging/individual_models.py   -c configs/models/t5_large.json configs/models/ia3.json  configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3  -er eval_batch_size=32
```

### Average
- Full-Model Fine-tuning 
```
python src/merging/average.py  -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32
```
- $(IA)^3$ 
```
python src/merging/average.py  -c configs/models/t5_large.json configs/models/ia3.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3  -er eval_batch_size=32
```

### Diagonal Fisher Merging
First, we save the diagonal Fisher for each model. Then, we compute the merged model using diagonal Fisher merging. 

#### Save the Diagonal Fisher
- Full-Model Fine-tuning 
```
python src/merging/save_metadata/save_fisher.py  -c configs/models/t5_large.json configs/evaluation_run/fisher.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  --split validation -f diagonal 
```
- $(IA)^3$ 
```
python src/merging/save_metadata/save_fisher.py  -c configs/models/t5_large.json configs/models/ia3.json configs/evaluation_run/fisher.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3  --split validation -f diagonal 
```

#### Merge the models using the Diagonal Fisher 
- Full-Model Fine-tuning 
```
python src/merging/diagonal_fisherMerging.py -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32 --split validation 
```
- $(IA)^3$ 
```
python src/merging/diagonal_fisherMerging.py -c configs/models/t5_large.json configs/models/ia3.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/flan_validation.json -d flan --checkpoint_descriptor ia3  -er eval_batch_size=32 --split validation 
```


### Task Vectors 

- Full-Model Fine-tuning 
```
python src/merging/task_vectors.py  -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32
```

- $(IA)^3$
```
python src/merging/task_vectors.py  -c configs/models/t5_large.json configs/models/ia3.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3  -er eval_batch_size=32
```

### TIES 
- Full-Model Fine-tuning 
```
python src/merging/ties.py  -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32
```
-  $(IA)^3$ 
```
python src/merging/ties.py  -c configs/models/t5_large.json configs/models/ia3.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3  -er eval_batch_size=32
```

### RegMean
First, we save the Gram matrix for each model. Then, we compute the merged model using RegMean. Note that RegMean cannot be applied to $(IA)^3$ which do not have linear layers. 

#### Saving the Gram Matrix 
```
python src/merging/save_metadata/save_gram_matrix.py  -c configs/models/t5_large.json configs/evaluation_run/fisher.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  --split validation  
```
Note that this save the Gram matrix of the input activations and the Gram matrix of the output activation gradients. 
RegMean only uses the Gram matrix of the input activations. The conjugate gradient method with the blockwise Fisher merging objective uses the Gram matrix of the input activations and the Gram matrix of the output activation gradients. 

#### Merge the models using RegMean  
```
python src/merging/regmean.py -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32  --split validation  
```

### Save Blockwise Fisher  

For $(IA)^3$, Blockwise Fisher merging saves the blockwise Fisher. For full-model fine-tuning, Blockwise Fisher saves the Gram matrix of the input activations and the Gram matrix of the output activation gradients. 

-  (IA)^3
```
python src/merging/save_metadata/save_fisher.py  -c configs/models/t5_large.json configs/models/ia3.json configs/evaluation_run/fisher.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3   --split validation   -f blockwise 
```
- Full-Model FIne-tuning
```
python src/merging/save_metadata/save_gram_matrix.py  -c configs/models/t5_large.json configs/evaluation_run/fisher.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  --split validation  
```
This is the same command for storing the Gram matrices for RegMean. 

### Conjugate Gradient Method

#### Average Objective
-  $(IA)^3$
```
python src/merging/conjugateGradient_average.py     -c configs/models/t5_large.json configs/models/ia3.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3   -er eval_batch_size=32  --split validation --num_iterations 100 --model_lambda 1.0 --initialization average  
```
- Full-Model Fine-tuning 
```
python src/merging/conjugateGradient_average.py     -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model   -er eval_batch_size=32  --split validation --num_iterations 100 --model_lambda 1.0 --initialization average  
```

#### Diagonal Fisher merging Objective 
- $(IA)^3$
```
python src/merging/conjugateGradient_diagonalFisher.py    -c configs/models/t5_large.json configs/models/ia3.json   configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3   -er eval_batch_size=32  --split validation --num_iterations 100 --model_lambda 1.0 
```
- Full-Model Fine-tuning  
```
python src/merging/conjugateGradient_diagonalFisher.py    -c configs/models/t5_large.json   configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model   -er eval_batch_size=32  --split validation --num_iterations 100 --model_lambda 1.0 
```

#### RegMean Objective 
This only holds for full-model fine-tuning. 
```
python src/merging/conjugateGradient_fisherMerging.py  -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32  --split validation  --model_lambda 1.0 --num_iterations 100   --initialization exp_out/merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/task_vectors/model_lambda_0.3/merged_model.pt 
```

#### Fisher Merging Objective 
- $(IA)^3$
```
python src/merging/conjugateGradient_fisherMerging.py  -c  configs/models/t5_large.json configs/models/ia3.json  configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor ia3  -er eval_batch_size=32  --split validation  --model_lambda 1.0 --num_iterations 10   --initialization  exp_out/merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/task_vectors/model_lambda_0.2/merged_model.pt -f blockwise   
```
- Full Model 
P3
```
python src/merging/conjugateGradient_fisherMerging.py  -c configs/models/t5_large.json configs/evaluation_run/individual_task.json configs/evaluation_dataset/p3_validation.json -d p3_eight_qa --checkpoint_descriptor full_model  -er eval_batch_size=32  --split validation  --model_lambda 1.0 --num_iterations 100  --use_backward --initialization  exp_out/merging/p3/p3_eight_qa/google-t5-large-lm-adapt/full_model/task_vectors/model_lambda_0.3/merged_model.pt
```

## Evaluation 

Run the `inference` script with

`-c` the config of the experiment with the model 

`-k` any evaluation config parameters to update.

`-e` experiment directory with model to evaluate 

`--merged_model` merged model to evaluate 

```
python src/inference.py -e  exp_out/p3/quartz/facebook-opt-1.3b/full_model/2023-10-21-11-34-24 --checkpoint_idx 299  -ed split=train   -er eval_batch_size=32
```

```
python src/inference.py --merged_model exp_out/merging/p3/p3_eight_qa/google-t5-large-lm-adapt/ia3/average/merged_model.pt -ed evaluation_split=test -i p3_eight_qa  -er eval_batch_size=32 
```

## Checkpoints

The models for `p3_eight_qa` can be found at this [google cloud storage bucket](https://console.cloud.google.com/storage/browser/merging_by_matching_models_in_task_subspaces;tab=objects?forceOnBucketsSortingFiltering=true&authuser=1&hl=en&project=craffel&prefix=&forceOnObjectsSortingFiltering=false). 
This includes 
- checkpoints for merging under `exp_out/p3`
- multitask trained checkpoint under `exp_out/p3_eight_qa`
- merged models using various methods under `exp_out/merging`

When downloading models, the directory structure should match the structure in the bucket, with `exp_out` under `mms`. 
