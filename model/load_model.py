from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from transformers import BertTokenizer, BertModel
from transformers import get_linear_schedule_with_warmup

import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split, RandomSampler, SequentialSampler

import pandas as pd


class Classifier():
    def __init__(self, args):
        # model_name = ['bert-base-cased' , 'gpt2-large']
        if args.model_name == "bert_base-cased":
            self.tokenizer = BertTokenizer.from_pretrained(args.model_name)
            if args.load_model:
                self.model.load_state_dict(torch.load(args.model_path, weights_only=True))
            else:
                self.model = BertModel.from_pretrained(args.model_name)
            #do this
        elif args.model_name == 'gpt2-large':
            configuration = GPT2Config.from_pretrained(args.model_name)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            if args.load_model:
                self.model.load_state_dict(torch.load(args.model_path, weights_only=True))
            else:
                self.model = GPT2LMHeadModel.from_pretrained(args.model_name, config=configuration)
            #do this

        if args.scheduler != None:
            self.scheduler = 
    def finetune(self, args, dataset):
    
    def validate(self, args, dataset):
    
    def save_model(self, save_path):

