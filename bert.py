# -*- coding: utf-8 -*-
"""BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oMeAdmecvCuV9Tup-rR2V_T2bEaaSxbj
"""

import argparse
import logging
import os
import random
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertForMultipleChoice
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def load_model(model_name):
    """
    This function loads the a pretrained model and add a classifier layer on top
    Inputs:
        model_name - name of the pretrained model
        in_features: input dimension of classifier layer
        num_classes: number of classes which is the dimension of output of classifier layer
        freeze_lm: boolean parameter indicating if to freeze weights of pretrained model
    """
    ## Load pretrained model
    if model_name == 'bert_for_multiple_choice':
      model = BertForMultipleChoice.from_pretrained("bert-base-uncased")
    else:
      pass

    ## freeze all weights in LM except last layer
    for name, param in model.named_parameters():
      if name != 'classifier':
        param.requires_grad = False

    return model

def create_dataset(data_dir, max_length):
    """
    This function creates dataset for dataloader
    Inputs:
        data_path: path to dataset
    """
    ## Load pretrained model and tokenizer of chosen model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    data = []
    for filename in os.listdir(data_dir):
        # Check if the item is a file (not a subdirectory)
        file_path = os.path.join(data_dir, filename)
        if os.path.isfile(file_path):
            ## Load data
            with open(file_path, "r") as file:
              f = file.read()
              f = json.loads(f)
              data.append(f)

    ## Get diagolue, choices, and answers
    contexts = [sample['article'] for sample in data]
    choices = [sample['options'] for sample in data]
    map_to_int = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    labels = [map_to_int[sample['answers']] for sample in data]

    ## Create data by concatenating context with each choice
    X = [tokenizer([context]*4,
                   choice,
                   return_tensors="pt",
                   max_length=max_length,
                   truncation=True,
                   padding="max_length") for context, choice in zip(contexts, choices)]
    y = [torch.tensor(label) for label in labels]
    return X,y

class MultuDataset(Dataset):
    """
    Custom dataset class
    """
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Multu_Module(pl.LightningModule):
    """
    Torch lightning training pipeline
    """
    def __init__(self, model_name, optimizer_name):
          """
          Inputs:
              args - user defined arguments
          """
          super().__init__()
          self.model = load_model(model_name)
          self.loss_module = nn.CrossEntropyLoss()
          self.optimizer_name = optimizer_name

    def forward(self, instance):
        return self.model(instance)

    def configure_optimizers(self):
        if self.optimizer_name == "Adam":
            optimizer = optim.AdamW(
                self.parameters())
        elif self.optimizer_name == "SGD":
            optimizer = optim.SGD(self.parameters())
        else:
            assert False, f"Unknown optimizer: \"{self.optimizer_name}\""

    def training_step(self, batch, batch_idx):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        preds = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        loss = preds.loss
        acc = (np.argmax(preds) == labels).float().mean()
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        preds = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        acc = (np.argmax(preds) == labels).float().mean()
        self.log('val_acc', acc)

    def test_step(self, batch, batch_idx):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        preds = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        acc = (np.argmax(preds) == labels).float().mean()
        self.log('test_acc', acc)

    def unpack_batch(self, batch):
        input_ids = batch[0]['input_ids']
        attention_masks = batch[0]['attention_mask']
        token_type_ids = batch[0]['token_type_ids']
        labels = batch[1]
        return input_ids, attention_masks, token_type_ids, labels

def fine_tune(args):
    """
    Function to conduct fine tuning
    Inputs:
        args - user defined arguments
    """
    # Create dataset
    train_X, train_y = create_dataset(args.train_data_path, args.max_length)
    train_dataset = MultuDataset(train_X, train_y)
    val_X, val_y = create_dataset(args.train_data_path, args.max_length)
    val_dataset = MultuDataset(val_X, val_y)
    test_X, test_y = create_dataset(args.train_data_path, args.max_length)
    test_dataset = MultuDataset(test_X, test_y)

    # Create dataloader
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=args.num_workers)
    val_loader = data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(args.checkpoint_path, args.model_name),
                         accelerator="gpu" if str(args.device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=args.max_epochs,
                         enable_progress_bar=True)
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard
    model = Multu_Module(args.model_name, args.optimizer_name)
    # # Check whether pretrained model exists. If yes, load it and skip training
    # pretrained_filename = os.path.join(args.checkpoint_path, args.model_name + ".ckpt")
    # if os.path.isfile(pretrained_filename):
    #     print(f"Found pretrained model at {pretrained_filename}, loading...")
    #     model = Multu_Module.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    # else:
    #     pl.seed_everything(42) # To be reproducable
    #     model = Multu_Module(args)
    #     trainer.fit(model, train_loader, val_loader)
    #     model = Multu_Module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result

def parseArgs():
    """
    Function to parse user input argument
    """
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--model_name", default='bert_for_multiple_choice', type=str, required=True,
                        help="name of pre-trained-language model")
    parser.add_argument("--optimizer_name", default='Adam', type=str, required=True,
                        help="optimizer")
    parser.add_argument("--train_data_path", default=None, type=str, required=True,
                        help="path of training data")
    parser.add_argument("--val_data_path", default=None, type=str, required=True,
                        help="path of validation data")
    parser.add_argument("--test_data_path", default=None, type=str, required=True,
                        help="path of test data")
    parser.add_argument("--max_length", default=512, type=int, required=True,
                        help="Maximum length of input sequence")
    parser.add_argument("--batch_size", default=8, type=int, required=True,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--checkpoint_path", default=None, type=str, required=True,
                        help="path to store checkpoints")
    parser.add_argument("--max_epochs", default=10, type=int, required=False,
                        help="Maximum number of epochs, most likely the number of epochs")
    parser.add_argument("--device", default='cpu', type=str, required=False,
                        help="cpu or cuda for training")
    parser.add_argument("--num_workers", default=4, type=int, required=False,
                        help="number of cpus/gpus to run on parallel")

    args = parser.parse_args()
    return args

def main():
  args = parseArgs()
  print(args)
  model, result = fine_tune(args)

if __name__ == "__main__":
    main()

