from dataset_modified import processors
import os
import json
import argparse
from pytorch_lightning.loggers import CometLogger
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from transformers import BertConfig, BertForMultipleChoice
from dataset_modified import load_and_cache_examples
from Customized_TOD import TOD
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

os.environ["TOKENIZERS_PARALLELISM"] = "false"
load_dotenv()

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
    if model_name.lower() == 'bert':
        model_config = BertConfig.from_pretrained("bert-base-uncased", num_labels=num_classes)
        model = BertForMultipleChoice.from_pretrained("bert-base-uncased",config=model_config)

        ## freeze all weights in LM to reduce computational complex
        if freeze_lm:
            for name, param in model.named_parameters():
                if name.startswith('classifier'):
                    param.requires_grad = False
    elif model_name.lower() == 'tod_bert':
        root_model =  AutoModel.from_pretrained('TODBERT/TOD-BERT-JNT-V1')
        ## freeze all weights in LM except last layer
        if freeze_lm:
            for name, param in root_model.named_parameters():
                param.requires_grad = False
        ## Get customized tod model
        model = TOD(root_model, num_classes)
    else:
        print('Model name is incorrect!')
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
        self.args = args


    def forward(self, instance):
        return self.model(torch.Tensor(instance))


    #TODO: Add hparameter for learning rate
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), 1e-5) 
        if self.args.lr_schedular:
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 25], gamma=0.1)
            return [optimizer], [scheduler]
        return optimizer
        

    def training_step(self, batch):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)    
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        loss, logits = outputs.loss, outputs.logits
        
        preds = logits.detach().cpu().numpy()
        preds_pos_1 = np.argmax(preds, axis=-1)
        out_label_ids = labels.detach().cpu().numpy()
        acc = simple_accuracy(preds_pos_1, out_label_ids)
        
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, verbose):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        logits = outputs.logits
        
        preds = logits.detach().cpu().numpy()
        preds_pos_1 = np.argmax(preds, axis=-1)
        out_label_ids = labels.detach().cpu().numpy()
        acc = simple_accuracy(preds_pos_1, out_label_ids)
        self.log('val_acc', acc)

    def test_step(self, batch, verbose):
        input_ids, attention_masks, token_type_ids, labels = self.unpack_batch(batch)
        
        
       
        outputs = self.model(input_ids=input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids, labels=labels)
        logits = outputs.logits

        
        preds = logits.detach().cpu().numpy()
        preds_pos_1 = np.argmax(preds, axis=-1)
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
    if args.model_name.lower() == 'bert':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif args.model_name.lower() == 'tod_bert':
        tokenizer = AutoTokenizer.from_pretrained("TODBERT/TOD-BERT-JNT-V1")
    else:
        print("please check model name!")
    train_dataset, val_dataset = load_and_cache_examples(args, args.task_name, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)    
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_dataset, _ = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    experiment_name = str(args.model_name + "_batch_size" + str(args.batch_size) + '_epochs'+str(args.max_epochs))
    # Create a PyTorch Lightning trainer with the generation callback
    comet_logger = CometLogger(
        api_key=os.getenv('COMET_API_KEY'), ## change to your api key
        project_name="mutual",
        workspace=os.getenv('WORKSPACE'),
        save_dir="checkpoint/", 
        experiment_name=experiment_name
    )
    trainer = pl.Trainer(default_root_dir=os.path.join(args.checkpoint_path, args.model_name),
                         accelerator=args.device,
                         devices=1,
                         max_epochs=args.max_epochs,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc", dirpath=experiment_name),
                                    LearningRateMonitor("epoch")],
                         enable_progress_bar=True,
                         logger=comet_logger)
    trainer.logger._log_graph = True         # If True, we plot the computation graph in tensorboard

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(args.checkpoint_path, args.model_name + ".ckpt")
    
    print(pretrained_filename, ' pretrained filename!!!!!!')
    
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
