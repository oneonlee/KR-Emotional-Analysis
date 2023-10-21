import os
import argparse
import json
import logging
import sys

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Sequence

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output-dir", type=str, required=True, help="output directory path to save artifacts")
g.add_argument("--model-path", type=str, default="klue/roberta-base", help="model file path")
g.add_argument("--tokenizer", type=str, default="klue/roberta-base", help="huggingface tokenizer path")
g.add_argument("--max-seq-len", type=int, default=128, help="max sequence length")
g.add_argument("--batch-size", type=int, default=32, help="training batch size")
g.add_argument("--valid-batch-size", type=int, default=64, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=10, help="the numnber of training epochs")
g.add_argument("--patience", type=int, default=10, help="number of early stop patience")
g.add_argument("--learning-rate", type=float, default=2e-4, help="max learning rate")
g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--gpu-num", type=int, default=0, help="gpu number for training")
g.add_argument("--cleansing", type=str, default="no", help="cleansing method for KcElectra")
g.add_argument("--removing-symbol", type=str, default="no", help="cleansing method for KcElectra")
g.add_argument("--removing-emoji", type=str, default="no", help="cleansing method for KcElectra")
g.add_argument("--removing-others", type=str, default="no", help="removing '&others&'")
g.add_argument("--num-folds", type=int, default=10, help="number of folds for K-Fold Cross Validation")

def save_dataset_to_jsonl(dataset, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for example in dataset:
            # 데이터 포맷을 정의하고 example을 이에 맞게 변환
            data_to_save = {
                "input": example["input"],
                "output": example["output"]
            }
            f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')

def main(args):
    logger = logging.getLogger("train")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
        logger.addHandler(handler)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguments ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Use Device: {args.gpu_num}")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_num)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    logger.info(f"[+] Set Random Seed to {args.seed}")
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore

    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        EvalPrediction,
        EarlyStoppingCallback
    )

#     class FocalLoss(BCEWithLogitsLoss):
#         def __init__(self, gamma=2.0, alpha=0.25, pos_weight=None, reduction='mean'):
#             super().__init__(pos_weight=pos_weight, reduction=reduction)
#             self.gamma = gamma
#             self.alpha = alpha

#         def forward(self, input_, target):
#             BCE_loss = super().forward(input_, target)
#             pt = torch.exp(-BCE_loss)
#             F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

#             if self.reduction == 'mean':
#                 return torch.mean(F_loss)
#             elif self.reduction == 'sum':
#                 return torch.sum(F_loss)
#             else:
#                 return F_loss
    
#     # Define trainer
#     class CustomTrainer(Trainer):
#         def __init__(self, *args, **kwargs):
#             super().__init__(*args, **kwargs)

#         def compute_loss(self, model, inputs, return_outputs=False):
#             labels = inputs.pop("labels")
#             outputs = model(**inputs)
#             logits = outputs.logits

#             loss_fct = FocalLoss(reduction='sum')

#             if len(labels.shape) > 1: # multi-label case
#                 # print(logits.shape) # torch.Size([Batch, num_class])
#                 # print(logits.view(-1).shape) # torch.Size([Batch * num_class])
#                 # print(labels.shape) # torch.Size([Batch, num_class])
#                 # print(labels.view(-1).shape) # torch.Size([Batch * num_class])
#                 loss = loss_fct(logits.view(-1), labels.float().view(-1))
                
#             else: # single-label case
#                 loss = loss_fct(logits.view(-1), labels.long().view(-1))

#             return (loss, outputs) if return_outputs else loss 


    # Reference: https://github.com/Alibaba-MIIL/ASL/blob/8c9e0bd8d5d450cf19093363fc08aa7244ad4408/src/loss_functions/losses.py#L5
    class ASLLoss(nn.Module):
        def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1):
            super(ASLLoss, self).__init__()

            self.gamma_pos = gamma_pos
            self.gamma_neg = gamma_neg
            self.eps = eps

        def forward(self, inputs, target):
            """inputs : N * C,
               target : N * C"""

            logit = torch.sigmoid(inputs)

            pos_part = target * ((1 - logit) ** self.gamma_pos) * torch.log(logit + self.eps)
            neg_part = (1 - target) * ((logit) ** self.gamma_neg) * torch.log(1 - logit + self.eps)

            return -(torch.sum(pos_part + neg_part)) / inputs.size()[0]
        
    class ASLCustomTrainer(Trainer):
        def __init__(self, model=None, args=None,
                     train_dataset=None,
                     eval_dataset=None,
                     compute_metrics=None,
                     tokenizer=None,
                     asl_loss_config={"gamma_pos": 0,"gamma_neg": 4,"eps": 0.1},
                     **kwargs):

            super().__init__(model=model,args=args,
                              train_dataset=train_dataset,
                              eval_dataset=eval_dataset,
                              compute_metrics=compute_metrics,
                              tokenizer=tokenizer,**kwargs)

            # Instantiate the loss function 
            self.loss_fct = ASLLoss(**asl_loss_config)
                
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            loss = self.loss_fct(logits.view(-1), labels.float().view(-1))

            return (loss, outputs) if return_outputs else loss 
        
            
    logger.info(f'[+] Load Tokenizer"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Load Dataset')
    dataset = Dataset.from_json("resource/data/nikluge-ea-2023-train+dev.jsonl")
    labels = list(dataset["output"][0].keys())
    id2label = {idx:label for idx, label in enumerate(labels)}
    label2id = {label:idx for idx, label in enumerate(labels)}
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f)

    logger.info(f'[+] Text Cleansing : {args.cleansing}')
    logger.info(f'[+] Removing "&others&" : {args.removing_others}')

    import re
    import emoji
    from soynlp.normalizer import repeat_normalize

    pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
    url_pattern = re.compile(
        r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

    def clean(x, removing_symbol, removing_emoji):
        if removing_symbol == "yes":
            x = pattern.sub(' ', x) # emoji 포함 특수문자 전부 삭제
        else:
            if removing_emoji == "yes":
                x = emoji.replace_emoji(x, replace='') # emoji 삭제
                
        x = url_pattern.sub('', x)
        x = x.strip()
        x = repeat_normalize(x, num_repeats=2)
        return x
        
    def preprocess_data(examples):
        # take a batch of texts
        text1 = examples["input"]["form"]
        text2 = examples["input"]["target"]["form"]
        
        if args.removing_others == "yes":
            if type(text1) is str:
                text1 = text1.replace("&others&", "")
            if type(text2) is str:
                text2 = text2.replace("&others&", "")
        if args.cleansing == "yes":
            if type(text1) is str:
                text1 = clean(text1, args.removing_symbol, args.removing_emoji)
            if type(text2) is str:
                text2 = clean(text2, args.removing_symbol, args.removing_emoji)
            
        # encode them
        encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=args.max_seq_len)

        # add labels
        encoding["labels"] = [0.0] * len(labels)
        for key, idx in label2id.items():
            if examples["output"]=='':
                encoding["labels"][idx] = 0.0
            if examples["output"][key] == 'True':
                encoding["labels"][idx] = 1.0
        
        return encoding


    kfold = MultilabelStratifiedKFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)

    form = []
    output = []
    for data in dataset:
        form.append(data["input"]["form"])
        result_list = [1 if value == 'True' else 0 for value in data["output"].values()]
        output.append(result_list)
        
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(form, output)):
        logger.info(f"===== Fold {fold + 1}/{args.num_folds} =====")

        train_ds = dataset.select(train_idx)
        valid_ds = dataset.select(valid_idx)

        # train 데이터셋 저장
        save_dataset_to_jsonl(train_ds, f"resource/data/folded/train_dataset_fold{fold}.jsonl")
        # valid 데이터셋 저장
        save_dataset_to_jsonl(valid_ds, f"resource/data/folded/valid_dataset_fold{fold}.jsonl")
        
        encoded_tds = train_ds.map(preprocess_data, remove_columns=train_ds.column_names)
        encoded_vds = valid_ds.map(preprocess_data, remove_columns=valid_ds.column_names)

        # Load model and set up trainer for each fold
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            problem_type="multi_label_classification",
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id
        )
        model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(num_added_token + tokenizer.vocab_size + 1)
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
    
        targs = TrainingArguments(
            output_dir=os.path.join(args.output_dir, f"fold_{fold}"),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.valid_batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
        )
        
        def multi_label_metrics(predictions, labels, threshold=0.5):
            #first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
            sigmoid = torch.nn.Sigmoid()
            probs = sigmoid(torch.Tensor(predictions))
            # next, use threshold to turn them into integer predictions
            y_pred = np.zeros(probs.shape)
            y_pred[np.where(probs >= threshold)] = 1
            # finally, compute metrics
            y_true = labels
            f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
            roc_auc = roc_auc_score(y_true, y_pred, average = 'micro')
            accuracy = accuracy_score(y_true, y_pred)
            # return as dictionary
            metrics = {'f1': f1_micro_average,
                    'roc_auc': roc_auc,
                    'accuracy': accuracy}
            return metrics

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            # print(preds)
            # print(preds.shape) # (4269, 8)
            # print(p.label_ids)
            # print(p.label_ids.shape) # (4269, 8)
            result = multi_label_metrics(predictions=preds, labels=p.label_ids)
            return result

        trainer = ASLCustomTrainer(
            model=model,
            args=targs,
            train_dataset=encoded_tds, # 학습데이터
            eval_dataset=encoded_vds,  # validation 데이터
            compute_metrics=compute_metrics, # 모델 평가 방식
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
        
        trainer.train()
        
        with open(os.path.join(os.path.join(args.output_dir, f"fold_{fold}"), "label2id.json"), "w") as f:
            json.dump(label2id, f)
# end main

if __name__ == "__main__":
    exit(main(parser.parse_args()))