---
license: apache-2.0
base_model: facebook/wav2vec2-base
tags:
- generated_from_trainer
metrics:
- wer
model-index:
- name: parp-wave2vec
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# parp-wave2vec

This model is a fine-tuned version of [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 3.0206
- Wer: 1.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 128
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 10

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer |
|:-------------:|:-----:|:----:|:---------------:|:---:|
| 5.1061        | 3.12  | 100  | 3.8624          | 1.0 |
| 3.1403        | 6.25  | 200  | 3.3085          | 1.0 |
| 2.9793        | 9.38  | 300  | 3.0206          | 1.0 |


### Framework versions

- Transformers 4.36.2
- Pytorch 2.0.1+cu117
- Datasets 2.16.1
- Tokenizers 0.15.0
