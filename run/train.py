import os
import argparse
import json
import logging
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from typing import Optional, Sequence

import numpy as np
from datasets import Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
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
g.add_argument("--removing-others", type=str, default="no", help="removing '&others&'")
g.add_argument("--num-folds", type=int, default=10, help="number of folds for K-Fold Cross Validation")
    
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

    logger.info(" ====== Arguements ======")
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
    
    # Define trainer
    class CustomTrainer(Trainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def compute_loss(self, model, inputs, return_outputs=False):
            # forward pass
            labels = inputs.pop("labels").to(torch.int64)
            outputs = model(**inputs) # outputs[0]은 logits tensor입니다.
            logits = outputs[0] # torch.Size([B, 8])
            
            # joy_logit = logits[:, 0] # torch.Size([B])
            # anticipation_logit = logits[:, 1]
            # trust_logit = logits[:, 2]
            # surprise_logit = logits[:, 3]
            # disgust_logit = logits[:, 4]
            # fear_logit = logits[:, 5]
            # anger_logit = logits[:, 6]
            # sadness_logit = logits[:, 7]
    
            # ASLSingleLabel loss
            criterion = {
                'joy' : ASLSingleLabel().to(device),
                'anticipation' : ASLSingleLabel().to(device),
                'trust' : ASLSingleLabel().to(device),
                'surprise' : ASLSingleLabel().to(device),
                'disgust' : ASLSingleLabel().to(device),
                'fear' : ASLSingleLabel().to(device),
                'anger' : ASLSingleLabel().to(device),
                'sadness' : ASLSingleLabel().to(device),
            }
            
            loss = criterion['joy'](logits, labels[:, 0]) + \
                        criterion['anticipation'](logits, labels[:, 1]) + \
                        criterion['trust'](logits, labels[:, 2]) + \
                        criterion['surprise'](logits, labels[:, 3]) + \
                        criterion['disgust'](logits, labels[:, 4]) + \
                        criterion['fear'](logits, labels[:, 5]) + \
                        criterion['anger'](logits, labels[:, 6]) + \
                        criterion['sadness'](logits, labels[:, 7])
            
            
            # preds = F.softmax(logits, dim=0) # torch.Size([B, 8])
            # preds_binary = torch.where(preds >= 0.5, torch.tensor(1), torch.tensor(0)) # torch.Size([256, 8])
            
            outputs = None, \
                        logits
                        # logits[:, 0], \
                        # logits[:, 1], \
                        # logits[:, 2], \
                        # logits[:, 3], \
                        # logits[:, 4], \
                        # logits[:, 5], \
                        # logits[:, 6], \
                        # logits[:, 7]
            
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

    def clean(x): 
        x = pattern.sub(' ', x)
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
                text1 = clean(text1)
            if type(text2) is str:
                text2 = clean(text2)
            
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


    kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=args.seed)
    
    form = []
    output = []
    
    for data in dataset:
        form.append(data["input"]["form"])
        output.append(data["output"])
        
    for fold, (train_idx, valid_idx) in enumerate(kfold.split(form, output)):
        logger.info(f"===== Fold {fold + 1}/{args.num_folds} =====")

        train_ds = dataset.select(train_idx)
        valid_ds = dataset.select(valid_idx)

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


        # trainer = CustomTrainer(
        #     model=model,
        #     args=targs,
        #     train_dataset=encoded_tds, # 학습데이터
        #     eval_dataset=encoded_vds,  # validation 데이터
        #     compute_metrics=compute_metrics, # 모델 평가 방식
        #     tokenizer=tokenizer,
        #     callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        # )
        
        trainer = Trainer(
            model,
            targs,
            train_dataset=encoded_tds,
            eval_dataset=encoded_vds,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
        
        trainer.train()
        
        with open(os.path.join(os.path.join(args.output_dir, f"fold_{fold}"), "label2id.json"), "w") as f:
            json.dump(label2id, f)
# end main

    
# Reference: https://github.com/Alibaba-MIIL/ASL/blob/8c9e0bd8d5d450cf19093363fc08aa7244ad4408/src/loss_functions/losses.py#L107
class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size,number_classes)
        "target" dimensions: - (batch_size)
        '''
        
        num_classes = inputs.size()[-1]
        
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg,
                                 self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:  # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


if __name__ == "__main__":
    exit(main(parser.parse_args()))