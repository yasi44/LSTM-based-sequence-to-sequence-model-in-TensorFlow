import numpy as np
import tensorflow as tf
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, BartForConditionalGeneration, BartTokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Attention, Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Fine-tuning T5/BART in PyTorch
model_name = "t5-small"
t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(model_name)

def translate_t5(input_text):
    input_ids = t5_tokenizer("translate English to French: " + input_text, return_tensors="pt").input_ids
    output_ids = t5_model.generate(input_ids)
    return t5_tokenizer.decode(output_ids[0], skip_special_tokens=True)

print(translate_t5("I think we must proceed with that"))

# # Fine-tuning BART in PyTorch
# model_name = "facebook/bart-base"
# bart_tokenizer = BartTokenizer.from_pretrained(model_name)
# bart_model = BartForConditionalGeneration.from_pretrained(model_name)

# def translate_bart(input_text):
#     input_ids = bart_tokenizer("translate English to French: " + input_text, return_tensors="pt").input_ids
#     output_ids = bart_model.generate(input_ids)
#     return bart_tokenizer.decode(output_ids[0], skip_special_tokens=True)

# print(translate_bart("I think we must proceed with that"))
