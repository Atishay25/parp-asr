---
license: apache-2.0
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
- Loss: 0.4483
- Wer: 0.3476

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
- train_batch_size: 64
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 40

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 7.2839        | 1.59  | 100  | 5.8388          | 1.0    |
| 3.3061        | 3.17  | 200  | 3.2376          | 1.0    |
| 2.991         | 4.76  | 300  | 3.0763          | 1.0    |
| 2.9309        | 6.35  | 400  | 2.9807          | 1.0    |
| 2.8255        | 7.94  | 500  | 2.7915          | 1.0    |
| 2.4385        | 9.52  | 600  | 2.0330          | 1.0139 |
| 1.6806        | 11.11 | 700  | 1.0553          | 0.8019 |
| 0.7871        | 12.7  | 800  | 0.5798          | 0.5345 |
| 0.423         | 14.29 | 900  | 0.4795          | 0.4583 |
| 0.2885        | 15.87 | 1000 | 0.4599          | 0.4204 |
| 0.2297        | 17.46 | 1100 | 0.4404          | 0.3953 |
| 0.1869        | 19.05 | 1200 | 0.4463          | 0.3857 |
| 0.1478        | 20.63 | 1300 | 0.4319          | 0.3751 |
| 0.1386        | 22.22 | 1400 | 0.4364          | 0.3715 |
| 0.1158        | 23.81 | 1500 | 0.4448          | 0.3652 |
| 0.1076        | 25.4  | 1600 | 0.4324          | 0.3528 |
| 0.098         | 26.98 | 1700 | 0.4406          | 0.3607 |
| 0.0933        | 28.57 | 1800 | 0.4367          | 0.3547 |
| 0.0848        | 30.16 | 1900 | 0.4341          | 0.3526 |
| 0.0773        | 31.75 | 2000 | 0.4330          | 0.3550 |
| 0.0721        | 33.33 | 2100 | 0.4418          | 0.3493 |
| 0.0716        | 34.92 | 2200 | 0.4379          | 0.3494 |
| 0.067         | 36.51 | 2300 | 0.4369          | 0.3497 |
| 0.064         | 38.1  | 2400 | 0.4494          | 0.3488 |
| 0.06          | 39.68 | 2500 | 0.4483          | 0.3476 |


### Framework versions

- Transformers 4.30.0
- Pytorch 2.0.1+cu117
- Datasets 2.16.1
- Tokenizers 0.13.3
