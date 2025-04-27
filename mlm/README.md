---
library_name: transformers
license: mit
base_model: law-ai/InLegalBERT
tags:
- generated_from_trainer
model-index:
- name: BhLegalBert
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# BhLegalBert

This model is a fine-tuned version of [law-ai/InLegalBERT](https://huggingface.co/law-ai/InLegalBERT) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1901

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 10
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 2.6855        | 1.0    | 204  | 2.3569          |
| 1.9921        | 2.0    | 408  | 1.7673          |
| 1.7707        | 3.0    | 612  | 1.5404          |
| 1.3795        | 4.0    | 816  | 1.4402          |
| 1.3544        | 5.0    | 1020 | 1.2991          |
| 1.1963        | 6.0    | 1224 | 1.2556          |
| 1.1734        | 7.0    | 1428 | 1.2880          |
| 1.114         | 8.0    | 1632 | 1.2167          |
| 1.0242        | 9.0    | 1836 | 1.1970          |
| 1.0534        | 9.9545 | 2030 | 1.1901          |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.6.0+cu118
- Datasets 3.5.0
- Tokenizers 0.21.1
