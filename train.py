import torch 
torch.cuda.current_device()

from datasets import load_dataset, load_metric
from datasets import ClassLabel
import random
import pandas as pd

timit = load_dataset("timit_asr", data_dir='./data')
timit = timit.remove_columns(["phonetic_detail", "word_detail", "dialect_region", "id", "sentence_type", "speaker_id"])

def show_random_elements(dataset, num_examples=10):
    assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
    picks = []
    for _ in range(num_examples):
        pick = random.randint(0, len(dataset)-1)
        while pick in picks:
            pick = random.randint(0, len(dataset)-1)
        picks.append(pick)
    
    df = pd.DataFrame(dataset[picks])
    print(df.head())

## text normalization    
import re
chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["text"] = re.sub(chars_to_ignore_regex, '', batch["text"]).lower() + " "
    return batch

timit = timit.map(remove_special_characters)
show_random_elements(timit["train"].remove_columns(["audio", "file"]))

def extract_all_chars(batch):
    all_text = " ".join(batch["text"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = timit.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=timit.column_names["train"])
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_list = list(set(vocabs["train"]["vocab"][0]) | set(vocabs["test"]["vocab"][0]))
vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]
vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)

## instantiate an object of the Wav2Vec2CTCTokenizer class.
import json
from transformers import Wav2Vec2CTCTokenizer

with open('vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")

repo_name = "atishayj25/parp-wave2vec"
#tokenizer.push_to_hub(repo_name)

from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Processor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

import IPython.display as ipd
import numpy as np
import random

def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched" to ensure mapping is correct
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])
    
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch

timit = timit.map(prepare_dataset, remove_columns=timit.column_names["train"], num_proc=4)

# trim out audio sequences that are longer than 4sec. 
max_input_length_in_sec = 4.0
timit["train"] = timit["train"].filter(lambda x: x < max_input_length_in_sec * processor.feature_extractor.sampling_rate, input_columns=["input_length"])

from transformers import Wav2Vec2ForCTC

pred_processor = Wav2Vec2Processor.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo")
finetuned_model = Wav2Vec2ForCTC.from_pretrained("patrickvonplaten/wav2vec2-base-timit-demo").to("cuda")

wer_metric = load_metric("wer")
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

import torch 

def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = finetuned_model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = pred_processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)

    return batch

# decode Timit test set 
#results = timit["test"].map(map_to_result, remove_columns=timit["test"].column_names)
#print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))
#show_random_elements(results)
import torch.nn.utils.prune as prune

def pruning_bert(model, px, model_type='wav2vec_small'):
    """
    prune out wav2vec 2.0 BERT: 12 transformer layers for BASE, and 24 
                                transformer layers for LARGE

    note: position encoding, projection heads, layernorm statistics are not pruned. 
    """
    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))        
    print('num_transformer_blocks is', num_transformer_blocks)

    parameters_to_prune =[]
    for ii in range(num_transformer_blocks):
        parameters_to_prune.append((model.encoder.layers[ii].attention.k_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].attention.k_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].attention.v_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].attention.v_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].attention.q_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].attention.q_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].attention.out_proj, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].attention.out_proj, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].feed_forward.intermediate_dense, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].feed_forward.intermediate_dense, 'bias'))
        parameters_to_prune.append((model.encoder.layers[ii].feed_forward.output_dense, 'weight'))
        parameters_to_prune.append((model.encoder.layers[ii].feed_forward.output_dense, 'bias'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )
        
def unprune_bert(model, model_type='wav2vec_small'):
    """
    remove pruning forward pre-hook. This is useful when we want to tweek the learned pruned mask, which is used in PARP.
    """
    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))
    print('num_transformer_blocks is', num_transformer_blocks)

    parameters_to_prune =[]
    for ii in range(num_transformer_blocks):
        parameters_to_prune.append(model.encoder.layers[ii].attention.k_proj)
        parameters_to_prune.append(model.encoder.layers[ii].attention.v_proj)
        parameters_to_prune.append(model.encoder.layers[ii].attention.q_proj)
        parameters_to_prune.append(model.encoder.layers[ii].attention.out_proj)
        parameters_to_prune.append(model.encoder.layers[ii].feed_forward.intermediate_dense)
        parameters_to_prune.append(model.encoder.layers[ii].feed_forward.output_dense)

    for ii in range(0, len(parameters_to_prune)): # applying both weight+bias masks
        prune.remove(parameters_to_prune[ii], 'weight')
        prune.remove(parameters_to_prune[ii], 'bias')

def see_weight_rate(model, model_type='wav2vec_small'):
    """ check a model's zero rate 
    """
    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))        
    print('num_transformer_blocks is', num_transformer_blocks)

    sum_list_2, zero_sum_2 = 0, 0
    for ii in range(num_transformer_blocks):
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.k_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.k_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.k_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.k_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.v_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.v_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.v_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.v_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.q_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.q_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.q_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.q_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.out_proj.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.out_proj.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].attention.out_proj.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].attention.out_proj.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].feed_forward.intermediate_dense.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].feed_forward.intermediate_dense.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].feed_forward.intermediate_dense.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].feed_forward.intermediate_dense.bias == 0))

        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].feed_forward.output_dense.weight.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].feed_forward.output_dense.weight == 0))
        sum_list_2 = sum_list_2 + float(model.encoder.layers[ii].feed_forward.output_dense.bias.nelement())
        zero_sum_2 = zero_sum_2 + float(torch.sum(model.encoder.layers[ii].feed_forward.output_dense.bias == 0))

    bert_zero_rate = 100 * zero_sum_2 / sum_list_2
    print('BERT zero rate is {0:.2f}'.format(bert_zero_rate))
    return bert_zero_rate

pruning_rate = 0.5
pruning_bert(finetuned_model.wav2vec2, pruning_rate, model_type='wav2vec_small')
see_weight_rate(finetuned_model.wav2vec2)

mask_dict = {}; weight_dict = {}
model_dict = finetuned_model.state_dict()

for key in model_dict.keys():
    if 'mask' in key:
        mask_dict[key] = model_dict[key]
    else:
        weight_dict[key] = model_dict[key]

torch.save(mask_dict, 'pruned-w2v2_' + str(pruning_rate) + '_mask.pt')
torch.save(weight_dict, 'pruned-w2v2_' + str(pruning_rate) + '_weight.pt')

# decode Timit test set again
#results = timit["test"].map(map_to_result, remove_columns=timit["test"].column_names)

#print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

#show_random_elements(results)

def apply_pruning_mask(model, mask_dict, model_type='wav2vec_small'):
    """
    apply existing pruning mask to a pre-trained wav2vec 2.0. 
    """
    if model_type == 'wav2vec_small':
        num_transformer_blocks = 12
    elif model_type == 'libri960_big' or model_type == 'xlsr_53_56k':
        num_transformer_blocks = 24
    else:
        print('model type {} not supported'.format(model_type))        
    print('num_transformer_blocks is', num_transformer_blocks)

    parameters_to_prune =[]
    mask_list_w, mask_list_b = [], [] # maks list for weight and bias
    for ii in range(num_transformer_blocks):
        parameters_to_prune.append(model.encoder.layers[ii].attention.k_proj)
        mask_list_w.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.k_proj.weight_mask'])
        mask_list_b.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.k_proj.bias_mask'])
        parameters_to_prune.append(model.encoder.layers[ii].attention.v_proj)
        mask_list_w.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.v_proj.weight_mask'])
        mask_list_b.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.v_proj.bias_mask'])
        parameters_to_prune.append(model.encoder.layers[ii].attention.q_proj)
        mask_list_w.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.q_proj.weight_mask'])
        mask_list_b.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.q_proj.bias_mask'])
        parameters_to_prune.append(model.encoder.layers[ii].attention.out_proj)
        mask_list_w.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.out_proj.weight_mask'])
        mask_list_b.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.attention.out_proj.bias_mask'])
        parameters_to_prune.append(model.encoder.layers[ii].feed_forward.intermediate_dense)
        mask_list_w.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.feed_forward.intermediate_dense.weight_mask'])
        mask_list_b.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.feed_forward.intermediate_dense.bias_mask'])
        parameters_to_prune.append(model.encoder.layers[ii].feed_forward.output_dense)
        mask_list_w.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.feed_forward.output_dense.weight_mask'])
        mask_list_b.append(mask_dict['wav2vec2.encoder.layers.' + str(ii) + '.feed_forward.output_dense.bias_mask'])

    for ii in range(0, len(parameters_to_prune)): # applying both weight+bias masks
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'weight', mask=mask_list_w[ii])
        prune.CustomFromMask.apply(parameters_to_prune[ii], 'bias', mask=mask_list_b[ii])

from transformers import Wav2Vec2ForCTC

# load pre-trained model (not the finetuned one)
pretrained_model = Wav2Vec2ForCTC.from_pretrained(
    "facebook/wav2vec2-base", 
    ctc_loss_reduction="mean", 
    pad_token_id=processor.tokenizer.pad_token_id,
)

pretrained_model = pretrained_model.to('cuda')

# apply the 50% pruning mask back to pre-traiend initialization 
apply_pruning_mask(pretrained_model.wav2vec2, mask_dict)

# double-check the pre-trained model now has 50% sparsity 
see_weight_rate(pretrained_model.wav2vec2)

import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
    
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

from transformers import TrainingArguments

training_args = TrainingArguments(
  output_dir=repo_name,
  group_by_length=True,
  per_device_train_batch_size=32,
  gradient_accumulation_steps=1, 
  evaluation_strategy="steps",
  num_train_epochs=40,           # orignally 30 epochs used, taking ~3 hours
  fp16=False,
  gradient_checkpointing=True,
  save_steps=200,
  eval_steps=100,
  logging_steps=100,
  learning_rate=1e-4,
  weight_decay=0.005,
  warmup_steps=1000,
  save_total_limit=5,
  push_to_hub=True,
)

from transformers import Trainer

trainer = Trainer(
    model=pretrained_model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=timit["train"],
    eval_dataset=timit["test"],
    tokenizer=processor.feature_extractor,
)

torch.cuda.empty_cache()

trainer.train()
trainer.push_to_hub()

import copy
finetuned_pruned_model = copy.deepcopy(finetuned_model)
finetuned_pruned_model.wav2vec2.encoder.layers[11].attention.k_proj.weight_mask

from safetensors.torch import load_file

#finetuned_pruned_model.load_state_dict(torch.load("atishayj25/parp-wave2vec/checkpoint-300/model.safetensors"))
finetuned_pruned_model.load_state_dict(load_file("atishayj25/parp-wave2vec/checkpoint-1200/model.safetensors"))
finetuned_pruned_model.eval()
see_weight_rate(finetuned_pruned_model.wav2vec2)

# decode Timit test set again
def map_to_result(batch):
    with torch.no_grad():
        input_values = torch.tensor(batch["input_values"], device="cuda").unsqueeze(0)
        logits = finetuned_pruned_model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    batch["pred_str"] = processor.batch_decode(pred_ids)[0]
    batch["text"] = processor.decode(batch["labels"], group_tokens=False)
    
    return batch

results = timit["test"].map(map_to_result, remove_columns=timit["test"].column_names)

print("Test WER: {:.3f}".format(wer_metric.compute(predictions=results["pred_str"], references=results["text"])))

show_random_elements(results)