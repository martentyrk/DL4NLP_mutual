# -*- coding: utf-8 -*-
"""BERT.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oMeAdmecvCuV9Tup-rR2V_T2bEaaSxbj
"""
from dataset import processors
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# from dotenv import load_dotenv
from pytorch_lightning.loggers import CometLogger
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from transformers import BertConfig, BertForMultipleChoice
from dataset import load_and_cache_examples
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
# load_dotenv()

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
            if name.startswith('classifier'):
                param.requires_grad = False

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
        self.save_hyperparameters()
        processor = processors[args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
        self.model = load_model(args.model_name,
                                num_labels,
                                args.freeze_lm)
        self.loss_module = nn.CrossEntropyLoss()

        self.margin = args.contrastive_margin
        self.use_contrastive = args.use_contrastive
        self.contrastive_weight = args.contrastive_weight
        self.use_correlation = args.use_correlation
        self.correlation_weight = args.correlation_weight


    def forward(self, instance):
        return self.model(torch.Tensor(instance))


    #TODO: Add hparameter for learning rate
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 1e-3)    
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=10)
        return [optimizer], [scheduler]

    # Todo: contrasive loss, still need customization
    def contrastive_loss(self, outputs, labels):
        # Extract the logits for each option
        logits = outputs.logits
        # Compute distances between each option and the context
        distances = torch.norm(logits.unsqueeze(1) - logits.unsqueeze(0), dim=-1)
        # Fetch distances of the correct answers
        positive_distances = distances[torch.arange(distances.size(0)), labels]
        # Fetch the smallest negative distance (i.e., the distance of the closest incorrect option to the context)
        negative_distances, _ = distances.scatter(1, labels.unsqueeze(1), float('inf')).min(dim=-1)
        # Contrastive loss: encourage the distance of the correct answer to be at least a 'margin' smaller than that of the incorrect answer
        loss = torch.relu(positive_distances - negative_distances + self.margin)
        return loss.mean()

    # Todo: correlation loss
    def correlation_loss(self, outputs):
        # Extract the logits for each option
        logits = outputs.logits

        mean_logits = logits.mean(dim=0)
        # Compute the centered logits
        centered_logits = logits - mean_logits
        # Compute the covariance matrix
        covariance = torch.mm(centered_logits, centered_logits.t())
        # Compute the correlation matrix
        correlation = covariance / torch.sqrt(
            torch.mm(covariance.diag().view(-1, 1), covariance.diag().view(1, -1))
        )
        # Exclude the diagonal elements
        num_options = correlation.size(0)
        device = logits.device
        off_diagonal = correlation - torch.eye(num_options, device=device)
        loss = off_diagonal.pow(2).sum()
        return loss

    def correlation_loss_embeddings(self, embeddings):
        # print('EMBEDS: ', embeddings.shape)
        # covariance
        # meanVector = embeddings.mean(dim=0)
        # # print('mean vector: ', meanVector.shape)
        # # centereVectors = embeddings - meanVector
        # centereVectors = embeddings-embeddings.mean(dim=0)
        # # print('centere vectors: ', centereVectors)
        # # estimate covariance matrix
        # featureDim = meanVector.shape[0]
        # dataCount = embeddings.shape[0]
        # covMatrix = ((centereVectors.t()) @ centereVectors) / (dataCount - 1)
        # # print('cov matrix: ', covMatrix)
        #
        # # normalize covariance matrix
        # stdVector = torch.std(embeddings, dim=0)
        # # print('std: ', stdVector)
        # sigmaSigmaMatrix = (stdVector.unsqueeze(1)) @ (stdVector.unsqueeze(0))
        # # print('sigma: ', sigmaSigmaMatrix)
        # normalizedConvMatrix = covMatrix / sigmaSigmaMatrix
        # # print('normal: ', normalizedConvMatrix)
        #
        # deltaMatrix = normalizedConvMatrix - torch.eye(featureDim).to(self.device)
        # # print('delta: ', deltaMatrix)
        #
        # loss = torch.norm(deltaMatrix)  # Frobenius norm
        #
        # print('loss in method: ', loss)

        mean_embeddings = embeddings.mean(dim=0)
        # Compute the centered logits
        centered_embeddings = embeddings - mean_embeddings
        # Compute the covariance matrix
        covariance = torch.mm(centered_embeddings, centered_embeddings.t())
        # Compute the correlation matrix
        correlation = covariance / torch.sqrt(
            torch.mm(covariance.diag().view(-1, 1), covariance.diag().view(1, -1))
        )
        # Exclude the diagonal elements
        num_options = correlation.size(0)
        device = embeddings.device
        off_diagonal = correlation - torch.eye(num_options, device=device)
        loss = off_diagonal.pow(2).sum()
        # print('loss: ', loss)

        return loss
        

    def training_step(self, batch):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)

        pooler_output = None

        def model_hook(module, input_, output):
            nonlocal pooler_output
            pooler_output = output #embeddings

        pooler_hook = self.model.bert.pooler.register_forward_hook(model_hook)
            
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)

        pooler_hook.remove()

        loss, logits = outputs[:2]
        
        preds = logits.detach().cpu().numpy()
        preds_pos_1 = np.argmax(preds, axis=1)
        out_label_ids = labels.detach().cpu().numpy()
        acc = simple_accuracy(preds_pos_1, out_label_ids)
        
        loss = self.loss_module(logits, labels)

        if self.use_contrastive or self.use_correlation:
            if self.use_contrastive and self.use_correlation:
                print('USING BOTH')
                loss += self.contrastive_weight * self.contrastive_loss_embeddings(pooler_output, labels) + self.correlation_weight * self.correlation_loss_embeddings(pooler_output)
            elif self.use_contrastive:
                print('USING CONTRASTIVE')
                loss += self.contrastive_weight * self.contrastive_loss_embeddings(pooler_output, labels)
            else:
                print('USING CORRELATION')
                loss += self.correlation_weight * self.correlation_loss_embeddings(pooler_output)

        # print('pooler output: ', pooler_output)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        print('loss in train step: ', loss)
        return loss

    def validation_step(self, batch, verbose):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        _, logits = outputs[:2]

        
        preds = logits.detach().cpu().numpy()
        preds_pos_1 = np.argmax(preds, axis=1)
        out_label_ids = labels.detach().cpu().numpy()
        acc = simple_accuracy(preds_pos_1, out_label_ids)
        self.log('val_acc', acc)

    def test_step(self, batch, verbose):
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    
    test_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    # Create a PyTorch Lightning trainer with the generation callback
    comet_logger = CometLogger(
        api_key=os.getenv('COMET_API_KEY'), ## change to your api key
        project_name="mutual",
        workspace=os.getenv('WORKSPACE'),
        save_dir="checkpoint/", 
    )
    trainer = pl.Trainer(default_root_dir=os.path.join(args.checkpoint_path, args.model_name + "_model"),
                         accelerator=args.device,
                         devices=1,
                         max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True,
                         logger=comet_logger)
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
