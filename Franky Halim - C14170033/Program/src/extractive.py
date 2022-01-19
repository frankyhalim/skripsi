import os
import sys
import logging
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser, Namespace
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from pooling import Pooling
from data import pad_batch_collate, DatasetIndoNews, BertData
from classifier import TransformerEncoderClassifier
from transformers import AdamW, get_linear_schedule_with_warmup
from helpers import (
    block_ngrams,
    compute_rouge_score,
    get_average_length
)

logger = logging.getLogger(__name__)


class ExtractiveSummarizer(pl.LightningModule):
    def __init__(self, hparams):
        super(ExtractiveSummarizer, self).__init__()
        
        ref_summary = 1 if hparams.ref_summary else 0
            
        # To reload training hyperparameters
        self.checkpoint = None
        loaded_checkpoint = getattr(hparams, "load_checkpoint", None)
        if loaded_checkpoint:
            self.checkpoint = torch.load(loaded_checkpoint)
            logger.info("Using loaded checkpoint model with this hyperparameter:\n %s", str(self.checkpoint['hyperparameters']))
            hparams = self.checkpoint['hyperparameters']
        else:
            logger.info("Currently not using any checkpoint model")
            
        if type(hparams) is not Namespace:
            hparams = Namespace(**hparams) 
        
        self.save_hyperparameters(hparams)
        self.bert = BertData(hparams.model_name_or_path)
        self.word_embedding_model = self.bert.get_model()
        self.pooling_model = Pooling()
        self.emd_model_frozen = False
        if hparams.num_frozen_steps > 0:
            self.emd_model_frozen = True
            self.freeze_word_embedding_model()

        if hparams.classifier == "transformer_position":
            self.encoder = TransformerEncoderClassifier(
                d_model=self.word_embedding_model.config.hidden_size,
                dropout=hparams.classifier_dropout,
                num_layers=hparams.classifier_transformer_num_layers,
            )
        else:
            logger.error(
                "%s is not a valid value for `--classifier`. Exiting...",
                hparams.classifier,
            )
            sys.exit(1)
   
        self.ref_summary = ref_summary        
        self.dir_path = self.hparams.save_dir
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none") 
        self.tokenizer = self.bert.get_tokenizer()
        self.fold = self.hparams.seed
        self.save_path_train = "temp_train"
        self.save_path_val = "temp_val"
        self.save_path_test = "temp_test"
        self.save_file = None
        self.pad_batch_collate = None
        self.datasets = None
        self.optimizer = None
        self.train_dataloader_object = None  
        self.global_step_tracker = None
        self.min_loss = None
        self.avg_train_loss = None
        self.avg_val_loss = None
        self.raw_rouge = None
        self.rouge_score = None
        
        self.epoch = self.current_epoch + 1
        self.temp_train_dir = "train_evaluation_score"
        self.temp_train_gold = f"{self.save_path_train}/"
        self.temp_train_pred = f"{self.save_path_train}/"
        self.temp_val_dir = "val_evaluation_score"
        self.temp_val_gold = f"{self.save_path_val}/"
        self.temp_val_pred = f"{self.save_path_val}/"
        self.temp_test_dir = "test_evaluation_score"
        self.temp_test_gold = f"{self.save_path_test}/"
        self.temp_test_pred = f"{self.save_path_test}/"
        
        self.list_train_loss_epoch = []
        self.list_train_raw_rouge = []
        self.list_train_rouge_score = []
        
        self.list_val_loss_epoch = []
        self.list_val_raw_rouge = []
        self.list_val_rouge_score = []
        
        self.test_preds = []
        self.test_labels = []
        
        self.all_pred_train = ""
        self.all_pred_val = ""
        self.all_pred_test = ""
        self.all_gold_train = ""
        self.all_gold_val = ""
        self.all_gold_test = ""


    def forward(
        self,
        input_ids,
        attention_mask,
        sent_rep_mask=None,
        token_type_ids=None,
        sent_rep_token_ids=None,
        **kwargs,
    ):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if not self.hparams.no_use_token_type_ids:
            inputs["token_type_ids"] = token_type_ids

        outputs = self.word_embedding_model(**inputs, **kwargs)
        word_vectors = outputs[0]

        sents_vec, mask = self.pooling_model(
            word_vectors=word_vectors,
            sent_rep_token_ids=sent_rep_token_ids,
            sent_rep_mask=sent_rep_mask,
        )

        sent_scores = self.encoder(sents_vec, mask)
        return sent_scores, mask

    def unfreeze_word_embedding_model(self): 
        for param in self.word_embedding_model.parameters():
            param.requires_grad = True

    def freeze_word_embedding_model(self): 
        for param in self.word_embedding_model.parameters():
            param.requires_grad = False

    def compute_loss(self, outputs, labels, mask):
        loss = self.loss_func(outputs, labels.float()) 
        
        # Set all padding values to zero
        loss = loss * mask.float() 
        
        # Add up all the loss values for each sequence (including padding because
        # padding values are zero and thus will have no effect)
        sum_loss_per_sequence = loss.sum(dim=1) 
        
        # Count the number of losses that are not padding per sequence
        num_not_padded_per_sequence = mask.sum(dim=1).float()
        
        # Find the average loss per sequence
        average_per_sequence = sum_loss_per_sequence / num_not_padded_per_sequence
        
        # Get the sum of the average loss per sequence
        sum_avg_seq_loss = average_per_sequence.sum() 
        
        # Get the mean of `average_per_sequence`
        batch_size = average_per_sequence.size(0)
        mean_avg_seq_loss = sum_avg_seq_loss / batch_size  
        
        return mean_avg_seq_loss
    

    def setup(self, stage):
        if stage == "fit": 
            self.word_embedding_model = self.bert.get_model_with_config(
                self.hparams.model_name_or_path, model_config=self.word_embedding_model.config
            )
            if self.checkpoint is not None:
                self.load_state_dict(self.checkpoint['model_state_dict'])
                self.epoch = self.checkpoint['epoch'] + 1
                logger.info("Epoch start from %s", self.epoch)
                
                self.list_train_loss_epoch = self.checkpoint['train_histories']['loss']
                self.list_val_loss_epoch = self.checkpoint['val_histories']['loss']
                self.min_loss = min(self.list_val_loss_epoch)
            else:
                logger.info("Training without checkpoint")
            
        if stage == "test":
            if self.checkpoint is None:
                logger.info("Need to specify path checkpoint model to test model")
                sys.exit(1)
            self.load_state_dict(self.checkpoint['model_state_dict'])
            

    def prepare_data(self):
        datasets = {}
        data_splits = [
            self.hparams.train_name,
            self.hparams.val_name,
            self.hparams.test_name,
        ]
        
        for corpus_type in data_splits:
            full_path = self.hparams.data_path + str(self.hparams.seed) + "/" + corpus_type + ".indonews.bert.pt"
            torch_data = torch.load(full_path)
            data = [x for x in torch_data]
            max_seq_length = min(round(get_average_length(self.bert, data)), self.hparams.max_seq_length)
            datasets[corpus_type] = DatasetIndoNews(data, self.bert, max_seq_length)
        
        self.datasets = datasets
        self.pad_batch_collate = pad_batch_collate
        

    def train_dataloader(self):
        if self.train_dataloader_object:
            return self.train_dataloader_object
        if not hasattr(self,"datasets"):
            self.prepare_data()
        self.global_step_tracker = 0
        
        train_dataset = self.datasets[self.hparams.train_name]
        train_dataloader = DataLoader(
            train_dataset,
            num_workers=self.hparams.dataloader_num_workers, 
            batch_size=self.hparams.batch_size,
            collate_fn=self.pad_batch_collate,
            shuffle=True
        )
        self.train_dataloader_object = train_dataloader
        return train_dataloader

    def val_dataloader(self):
        valid_dataset = self.datasets[self.hparams.val_name]
        valid_dataloader = DataLoader(
            valid_dataset,
            num_workers=self.hparams.dataloader_num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=self.pad_batch_collate,
            shuffle=False
        )
        return valid_dataloader

    def test_dataloader(self):
        test_dataset = self.datasets[self.hparams.test_name]
        test_dataloader = DataLoader(
            test_dataset,
            num_workers=self.hparams.dataloader_num_workers,
            batch_size=self.hparams.batch_size,
            collate_fn=self.pad_batch_collate,
            shuffle=False
        )
        return test_dataloader

    def configure_optimizers(self):
        self.train_dataloader_object = self.train_dataloader()
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, 
                          eps=self.hparams.adam_epsilon, weight_decay=self.hparams.weight_decay)
        
        last_epoch = -1
        
        # Check load checkpoint model
        if self.checkpoint:
            logger.info("Currently using loaded optimizer")
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
            last_epoch = self.checkpoint['epoch'] - 1
            
        total_steps = len(self.train_dataloader_object) * self.hparams.max_epochs
        scheduler = {
            'scheduler': get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=total_steps, last_epoch=last_epoch),
            'interval':'step'
            }

        return [optimizer], [scheduler]
    

    def training_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]
        sources = batch["source"]

        # Delete labels, source, target so now batch contains everything to be inputted into the model
        del batch["labels"]
        del batch['source']
        del batch['target']
        

        # If global_step has increased by 1:
        # Begin training the `word_embedding_model` after `num_frozen_steps` steps
        if (self.global_step_tracker + 1) == self.trainer.global_step:
            self.global_step_tracker = self.trainer.global_step

            if self.emd_model_frozen and (self.trainer.global_step > self.hparams.num_frozen_steps):
                self.emd_model_frozen = False
                self.unfreeze_word_embedding_model()
        
        # Compute model forward (forward pass to compute output with mask by passing batch data to the model)
        outputs, mask = self.forward(**batch)
        
        # Compute loss
        train_loss = self.compute_loss(outputs, labels, mask)
        outputs = torch.sigmoid(outputs)
        
        # For compute F1 ROUGE score
        system_summaries = []
        ref_summaries = []
        for idx, label in enumerate(labels):
            temp = ""
            for idy, l in enumerate(label):
                if l:
                    temp += sources[idx][idy]
                    temp += '<q>'
            temp = temp[:-3]        
            ref_summaries.append(temp)
            
        source_ids = (
            torch.argsort(outputs, dim=1, descending=True)
        )
        
        for idx, (source, source_ids, target) in enumerate(
            zip(sources, source_ids, ref_summaries)
        ):
            pos = []
            for sent_idx, i in enumerate(source_ids):
                if i >= len(source):
                    continue
                
                pos.append(i.item())
                
                if len(pos) == self.hparams.test_k:
                    break
            pos.sort()
            selected_sentences = "<q>".join([source[i] for i in pos])
            system_summaries.append(selected_sentences)
        
        
        if self.hparams.no_use_token_type_ids:
            temp_token_type_ids = "no-token-type-ids"
        else:
            temp_token_type_ids = "use-token-type-ids"
        model_name_or_path = self.hparams.model_name_or_path.replace('/','-')
        self.save_file = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            model_name_or_path, temp_token_type_ids,
            self.hparams.classifier_transformer_num_layers, self.hparams.seed, self.hparams.learning_rate,
            self.hparams.classifier_dropout, self.hparams.batch_size, self.hparams.max_epochs
            )
            
        self.temp_train_gold = self.save_file + "/train_gold_" + str(self.epoch) + ".txt"
        self.temp_train_pred = self.save_file + "/train_pred_" + str(self.epoch) + ".txt"
        
        os.makedirs(self.save_file, exist_ok=True)
        
        for pred in system_summaries:
            self.all_pred_train += str(pred).strip() + "\n"
        for gold in ref_summaries:
            self.all_gold_train += str(gold).strip() + "\n"
        
        train_dict = {
            'epoch': self.epoch,
            'loss': train_loss,
            }
        for name, value in train_dict.items():
            self.log('train/'+name, float(value), prog_bar=True, sync_dist=True)
         
        return {'loss':train_loss}
    
    def training_epoch_end(self, outputs): 
        avg_train_loss = torch.stack(
            [x['loss'] for x in outputs]
            ).mean()
        
        with open(self.temp_train_pred, 'w') as save_pred, open(self.temp_train_gold, 'w') as save_gold:
            save_pred.write(self.all_pred_train)
            save_gold.write(self.all_gold_train)
            
        self.list_train_loss_epoch.append(avg_train_loss)
        
        train_dict = {
            'epoch': self.epoch,
            'loss': avg_train_loss
            }
        for name, value in train_dict.items():
            self.log('train/'+name, float(value), prog_bar=True, sync_dist=True)
        
        self.avg_train_loss = avg_train_loss
          
        if self.hparams.no_use_token_type_ids:
            temp_token_type_ids = "no-token-type-ids"
        else:
            temp_token_type_ids = "use-token-type-ids"
        model_name_or_path = self.hparams.model_name_or_path.replace('/','-')
        self.save_file = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            model_name_or_path, temp_token_type_ids,
            self.hparams.classifier_transformer_num_layers, self.hparams.seed, self.hparams.learning_rate,
            self.hparams.classifier_dropout, self.hparams.batch_size, self.epoch
            )
        
        ckpt_path = f"{self.dir_path}/{self.save_file}.bin"        
        optimizer = self.optimizers()
        optimizer = optimizer.optimizer
        
        train_histories = {
            'loss':self.list_train_loss_epoch,
            }
        val_histories = {
            'loss':self.list_val_loss_epoch,
            }
        
        saved_checkpoint = {
            'epoch': self.epoch,
            'hyperparameters': self.hparams,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_histories': train_histories,
            'val_histories': val_histories,
            }
        os.makedirs(f"{self.dir_path}",exist_ok=True)
        logger.info("Currently saving model in epoch %s", str(self.epoch))
        torch.save(saved_checkpoint, ckpt_path)
        
        best_path = "best_bert_model_checkpoints"
        best_name = ""
        best_checkpoint = {}
        
        # Save best model checkpoint if current validation loss is lower than minimum loss of previous epoch
        if not self.min_loss:
            self.min_loss = self.avg_val_loss
            best_name = self.save_file
            best_checkpoint = saved_checkpoint
        else:
            if self.avg_val_loss < self.min_loss:
                self.min_loss = self.avg_val_loss
                best_name = self.save_file
                best_checkpoint = saved_checkpoint
                os.makedirs(best_path,exist_ok=True)
                logger.info("Currently saving best bert model in epoch %s", str(self.epoch))
                best_ckpt_path = f"{best_path}/{best_name}.bin"
                torch.save(best_checkpoint, best_ckpt_path)
        
        self.all_pred_train = ""
        self.all_gold_train = ""        
        logger.info("Epoch %2d | Min val loss: %f | Current val loss: %f" %(self.epoch, self.min_loss, self.avg_val_loss))
        self.epoch += 1
        

    def validation_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]
        sources = batch["source"]

        # Delete labels, source, target so now batch contains everything to be inputted into the model
        del batch["labels"]
        del batch["source"]
        del batch["target"]

        # Compute model forward
        outputs, mask = self.forward(**batch)

        # Compute loss
        val_loss = self.compute_loss(outputs, labels, mask)
        outputs = torch.sigmoid(outputs)
        
        # For compute F1 ROUGE score
        system_summaries = []
        ref_summaries = []
        for idx, label in enumerate(labels):
            temp = ""
            for idy, l in enumerate(label):
                if l:
                    temp += sources[idx][idy]
                    temp += '<q>'
            temp = temp[:-3]        
            ref_summaries.append(temp)
        source_ids = (
            torch.argsort(outputs, dim=1, descending=True)
        )
        for idx, (source, source_ids, target) in enumerate(
            zip(sources, source_ids, ref_summaries)
        ):
            pos = []
            for sent_idx, i in enumerate(source_ids):
                if i >= len(source):
                    continue
                
                pos.append(i.item())
                
                if len(pos) == self.hparams.test_k:
                    break
            pos.sort()
            selected_sentences = "<q>".join([source[i] for i in pos])
            system_summaries.append(selected_sentences)
        
        
        if self.hparams.no_use_token_type_ids:
            temp_token_type_ids = "no-token-type-ids"
        else:
            temp_token_type_ids = "use-token-type-ids"
        model_name_or_path = self.hparams.model_name_or_path.replace('/','-')
        self.save_file = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            model_name_or_path, temp_token_type_ids,
            self.hparams.classifier_transformer_num_layers, self.hparams.seed, self.hparams.learning_rate,
            self.hparams.classifier_dropout, self.hparams.batch_size, self.hparams.max_epochs
            )
            
        self.temp_val_gold = self.save_file + "/val_gold_"+ str(self.epoch) +".txt"
        self.temp_val_pred = self.save_file + "/val_pred_"+ str(self.epoch) +".txt"
        
        os.makedirs(self.save_file, exist_ok=True) 
 
        for pred in system_summaries:
            self.all_pred_val += str(pred).strip() + "\n"
        for gold in ref_summaries:
            self.all_gold_val += str(gold).strip() + "\n"
            
        val_dict = {
            'epoch': self.epoch,
            'loss': val_loss
            }
        
        for name, value in val_dict.items():
            self.log('val/'+name, float(value), prog_bar=True, sync_dist=True)
            
        return {'loss':val_loss}
    
    def validation_epoch_end(self, outputs):
        # Get the average loss over all evaluation runs
        avg_val_loss = torch.stack(
            [x['loss'] for x in outputs]
        ).mean()
        
        with open(self.temp_val_pred, 'w') as save_pred, open(self.temp_val_gold, 'w') as save_gold:
            save_pred.write(self.all_pred_val)
            save_gold.write(self.all_gold_val)
        
        self.avg_val_loss = avg_val_loss
        self.list_val_loss_epoch.append(avg_val_loss)
        
        val_dict = {
            'epoch': self.epoch,
            'loss': avg_val_loss,
            }
        
        for name, value in val_dict.items():
            self.log('val/'+name, float(value), prog_bar=True, sync_dist=True)
        
        self.all_pred_val = ""
        self.all_gold_val = ""
        
        
    def test_step(self, batch, batch_idx):
        # Get batch information
        labels = batch["labels"]
        sources = batch["source"]
        targets = batch["target"]

        # Delete labels, source, and target so now batch contains everything to be inputted into the model
        del batch["labels"]
        del batch["source"]
        del batch["target"]
            
        # Compute model forward
        outputs, _ = self.forward(**batch)
        outputs = torch.sigmoid(outputs)
        
        sorted_ids = (
            torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()
        ) 
            
        predictions = []
        
        if self.ref_summary:
            ref_summaries = targets
        else: 
            ref_summaries = []
            for idx, label in enumerate(labels):
                temp = ""
                for idy, l in enumerate(label):
                    if l:
                        temp += sources[idx][idy]
                        temp += '<q>'
                temp = temp[:-3]        
                ref_summaries.append(temp)
            
        
        # Get ROUGE scores for each (source, target) pair
        for idx, (source, source_ids, target) in enumerate(
            zip(sources, sorted_ids, ref_summaries)
        ):
            current_prediction = []
            pos = []
            for sent_idx, i in enumerate(source_ids):
                if i >= len(source):
                    continue

                candidate = source[i].strip()
                
                
                # If n-gram blocking is enabled and searching for matching n-gram finds no matches
                # then add the candidate to the current prediction list.
                # During the predicting process, N-gram Blocking is used to reduce redundancy. Given
                # selected summary S and a candidate sentence c, then will skip c if there exists a
                # n-gram overlapping between c and S.
                if (not self.hparams.no_test_block_ngrams) and (
                    not block_ngrams(candidate, current_prediction, self.hparams.n_gram_blocking)
                ):
                    current_prediction.append(candidate)
                    pos.append(i.item())

                if len(current_prediction) == self.hparams.test_k:
                    break
            pos.sort()
            current_prediction = "<q>".join([source[i] for i in pos])
            predictions.append(current_prediction)
            
        if self.hparams.no_use_token_type_ids:
            temp_token_type_ids = "no-token-type-ids"
        else:
            temp_token_type_ids = "use-token-type-ids"
        model_name_or_path = self.hparams.model_name_or_path.replace('/','-')
        self.save_file = "{}_{}_{}_{}_{}_{}_{}_{}".format(
            model_name_or_path, temp_token_type_ids,
            self.hparams.classifier_transformer_num_layers, self.hparams.seed, self.hparams.learning_rate,
            self.hparams.classifier_dropout, self.hparams.batch_size, self.epoch
            )    
        self.temp_test_gold = self.save_path_test + "/" + self.save_file + "_test_gold.txt"
        self.temp_test_pred = self.save_path_test + "/" + self.save_file + "_test_pred.txt"
        
        # Gather all summaries in single text file  
        os.makedirs(self.save_path_test, exist_ok=True) 
        
        for pred in predictions:
            self.all_pred_test += str(pred).strip() + "\n"
        for gold in ref_summaries:
            self.all_gold_test += str(gold).strip() + "\n"
        
        return None

    def test_epoch_end(self, outputs):
        with open(self.temp_test_pred, 'w') as save_pred, open(self.temp_test_gold, 'w') as save_gold:
            save_pred.write(self.all_pred_test)
            save_gold.write(self.all_gold_test)
        
        # ROUGE scoring 
        raw_rouge, rouge_score = compute_rouge_score(
            self.fold, self.epoch, self.save_file, self.temp_test_dir, self.temp_test_pred, self.temp_test_gold
            )
        results_dir = "results"
        os.makedirs(results_dir,exist_ok=True)
        type_ref = "_abstractive" if self.ref_summary else "_extractive"
        file = results_dir + "/" + self.save_file + type_ref + ".txt"
        with open(file,'w') as f:
            f.write("Precision\n")
            f.write(
                "ROUGE-1 : {} \nROUGE-2 : {} \nROUGE-L : {}\n\n".format(
                    rouge_score['precision-rouge-1'], 
                    rouge_score['precision-rouge-2'], 
                    rouge_score['precision-rouge-l']
                    )
                )
            f.write("Recall\n")
            f.write(
                "ROUGE-1 : {} \nROUGE-2 : {} \nROUGE-L : {}\n\n".format(
                    rouge_score['recall-rouge-1'], 
                    rouge_score['recall-rouge-2'], 
                    rouge_score['recall-rouge-l']
                    )
                )
            f.write("F1-Score\n")
            f.write(
                "ROUGE-1 : {} \nROUGE-2 : {} \nROUGE-L : {}\n\n".format(
                    rouge_score['f1-rouge-1'],
                    rouge_score['f1-rouge-2'],
                    rouge_score['f1-rouge-l']
                    )
                )
            

        test_dict = {
            **rouge_score, 
            **raw_rouge
            }
        
        # Generate logs
        for name, value in test_dict.items():
            self.log('test/'+name, float(value), prog_bar=True, sync_dist=True) 

        
        self.test_preds = []
        self.test_labels = []
        self.all_pred_test = ""
        self.all_gold_test = ""

    def predict(
        self,
        input_sentences: str,
        source: str,
        num_summary_sentences=3,
        raw_scores=True
    ):
        input_ids = self.bert.get_input_ids(input_sentences) 
        input_ids = torch.tensor(input_ids) 
        attention_mask = [1] * len(input_ids) 
        attention_mask = torch.tensor(attention_mask)

        sent_rep_token_ids = [
            i for i, t in enumerate(input_ids) if t == self.tokenizer.cls_token_id
        ]
        sent_rep_mask = torch.tensor([1] * len(sent_rep_token_ids)) 
         
        _segs = [-1] + [i for i, t in enumerate(input_ids) if t == self.tokenizer.sep_token_id]
        segs = [_segs[i] - _segs[i - 1] for i in range(1, len(_segs))]        
        token_type_ids = [] 
        for i, s in enumerate(segs):
            if (i % 2 == 0):
                token_type_ids += s * [0]
            else:
                token_type_ids += s * [1]   
        token_type_ids = torch.tensor(token_type_ids)

        input_ids.unsqueeze_(0) 
        attention_mask.unsqueeze_(0)
        sent_rep_mask.unsqueeze_(0)
        token_type_ids.unsqueeze_(0)
        
        self.eval()
        
        # No gradient calculated during prediction
        with torch.no_grad():
            outputs, _ = self.forward(
                input_ids,
                attention_mask,
                sent_rep_mask=sent_rep_mask,
                sent_rep_token_ids=sent_rep_token_ids,
                token_type_ids=token_type_ids
            )
            outputs = torch.sigmoid(outputs)

        if raw_scores:
            sent_scores = list(zip(input_sentences, outputs.tolist()[0]))
            sent_scores = [f"{sentence} | {score}" for sentence, score in sent_scores]
            return json.dumps(sent_scores)

        sorted_ids = (
            torch.argsort(outputs, dim=1, descending=True).detach().cpu().numpy()
        )

        selected_ids = sorted_ids[0, :num_summary_sentences]
        selected_ids = np.sort(selected_ids)
        selected_sents = []
        for i in selected_ids:
            selected_sents.append(input_sentences[i])
        
        # Save selected sentences in text file
        saved_path = 'outputs/prediction_' + source
        dir_path = saved_path.split('/')[:2]
        dir_path = '/'.join(dir_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            
        with open(saved_path, 'w') as f:
            f.write('\n'.join(selected_sents))
        
        return json.dumps(selected_sents)


    @staticmethod
    def add_model_specific_args(parent_parser):
        """Arguments specific to this model"""
        parser = ArgumentParser(parents=[parent_parser],description='Extractive Summarization on Indonesian News using BERT')
        parser.add_argument("--save_dir", type=str, default="./bert_checkpoints", help="Directory path to save model")
        parser.add_argument("--ref_summary", type=int, default=0, help="Reference summary for scoring evaluation (0 = gold label, 1 = gold summary)")  
        parser.add_argument(
            "--model_name_or_path",
            type=str,
            default="bert-base-multilingual-uncased",
            help="Path to pre-trained model or shortcut name.",
        )
        parser.add_argument(
            "--model_type",
            type=str,
            default="bert",
            help="Used model type.",
        )
        parser.add_argument(
            "--max_seq_length",
            type=int,
            default=512,
            help="The maximum sequence length of BERT and transformer model.",
        )
        parser.add_argument(
            "--data_path", type=str, default='./prepared_data_basic/indo_news_', help="Directory containing used data."
        )
        parser.add_argument(
            "--pooling_mode",
            type=str,
            default="sent_rep_tokens",
            help="Convert word vectors to sentence embeddings.",
        )
        parser.add_argument(
            "--num_frozen_steps",
            type=int,
            default=0,
            help="Freeze (don't train) the word embedding model for this many steps.",
        )
        parser.add_argument(
            "--batch_size",
            default=8,
            type=int,
            help="Batch size per GPU/CPU for training/evaluation/testing.",
        )
        parser.add_argument(
            "--dataloader_num_workers",
            default=2,
            type=int,
            help="""The number of workers to use when loading data. A general place to 
            start is to set num_workers equal to the number of CPU cores used machine.""",
        )
        parser.add_argument(
            "--no_use_token_type_ids",
            action="store_true",
            help="Set to not train with `token_type_ids`.",
        )
        parser.add_argument(
            "--classifier",
            type=str,
            default="transformer_position",
            help="""Which classifier/encoder to use to reduce the hidden dimension of the sentence vectors""",
        )
        parser.add_argument(
            "--classifier_dropout",
            type=float,
            default=0.1,
            help="The value for the dropout layers in the classifier.",
        )
        parser.add_argument(
            "--classifier_transformer_num_layers",
            type=int,
            default=2,
            help='The number of layers for the `transformer` classifier.',
        )
        parser.add_argument(
            "--train_name",
            type=str,
            default="train",
            help="name for set of training files on disk.",
        )
        parser.add_argument(
            "--val_name",
            type=str,
            default="dev",
            help="name for set of validation files on disk.",
        )
        parser.add_argument(
            "--test_name",
            type=str,
            default="test",
            help="name for set of testing files on disk.",
        )
        parser.add_argument(
            "--test_k",
            type=int,
            default=3,
            help="The `k` parameter to chose top k predictions from the model for evaluation scoring (default: 3)",
        )
        parser.add_argument(
            "--n_gram_blocking",
            type=int,
            default=3,
            help="number of n-gram blocking for testing"
        )
        parser.add_argument(
            "--no_test_block_ngrams",
            action="store_true",
            help="Disable n-gram blocking when calculating ROUGE scores during testing.",
        )

        return parser