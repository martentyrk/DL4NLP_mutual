# -*- coding: utf-8 -*-
"""BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oMeAdmecvCuV9Tup-rR2V_T2bEaaSxbj
"""

import argparse
import logging
from dataset import processors
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import BertConfig, BertForMultipleChoice
from dataset import load_and_cache_examples
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def load_model(model_name, num_classes, freeze_lm=True):
    """
    This function loads the a pretrained model and add a classifier layer on top
    Inputs:
        model_name - name of the pretrained model
        in_features: input dimension of classifier layer
        num_classes: number of classes which is the dimension of output of classifier layer
        freeze_lm: boolean parameter indicating if to freeze weights of pretrained model
    """
    ## Load pretrained model
    model_config = BertConfig.from_pretrained(model_name, num_labels=num_classes)
    model = BertForMultipleChoice.from_pretrained(model_name,config=model_config)

    ## freeze all weights in LM to reduce computational complex
    if freeze_lm:
        for name, param in model.named_parameters():
            if name != 'classifier':
                param.requires_grad = False
            else:
                param.requires_grad = True

    ## Define a classifier layer and add it to the LM

    return model

class Mutual_Module(pl.LightningModule):
    """
    Torch lightning training pipeline
    """
    def __init__(self, args):
        """
          Inputs:
              args - user defined arguments
        """
        super().__init__()
        processor = processors[args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
        self.model = load_model(args.model_name,
                                num_labels,
                                args.freeze_lm)
        self.loss_module = nn.CrossEntropyLoss()


    def forward(self, instance):
        return self.model(torch.Tensor(instance))


    #TODO: Add hparameter for learning rate
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 1e-3)    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10)
        return [optimizer], [scheduler]
        

    def training_step(self, batch):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        
            
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        loss, logits = outputs[:2]
        
        preds = logits.detach().cpu().numpy()
        acc = (np.argmax(preds) == labels).float().mean()
        
        loss = self.loss_module(logits, labels)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        _, logits = outputs[:2]

        
        preds = logits.detach().cpu().numpy()
        acc = (np.argmax(preds) == labels).float().mean()
        self.log('val_acc', acc)

    def test_step(self, batch):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        
        
       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        _, logits = outputs[:2]

        
        preds = logits.detach().cpu().numpy()
        preds_pos_1 = np.argmax(preds, axis=1)
        out_label_ids = labels.detach().cpu().numpy()
        acc = simple_accuracy(preds_pos_1, out_label_ids)
        self.log('test_acc', acc)
        
    def unpack_batch(self, batch):
        input_ids = batch[0]
        attention_masks = batch[1]
        token_type_ids = batch[2]
        labels = batch[3]
        return input_ids, attention_masks, token_type_ids, labels

def fine_tune(args):
    """
    Function to conduct fine tuning
    Inputs:
        args - user defined arguments
    """
    # Create dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    train_dataset, val_dataset = load_and_cache_examples(args, args.task_name, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size)    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    test_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(default_root_dir=os.path.join(args.checkpoint_path, args.model_name + "_model"),
                         accelerator="gpu" if str(args.device).startswith("cuda") else "cpu",
                         devices=1,
                         max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True)
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(args.checkpoint_path, "_" + args.model_name + ".ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = Mutual_Module.load_from_checkpoint(pretrained_filename) # Automatically loads the model with the saved hyperparameters
    else:
        pl.seed_everything(args.seed) # To be reproducable
        model = Mutual_Module(args)
        trainer.fit(model, train_loader, val_loader)
        model = Mutual_Module.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # Load best checkpoint after training

    # Test best model on validation and test set
    val_result = trainer.test(model, val_loader, verbose=False)
    test_result = trainer.test(model, test_loader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}

    return model, result
