import os
import argparse
import json
import logging
import sys

import torch
from torch import nn
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
g.add_argument("--num-folds", type=int, default=5, help="number of folds for K-Fold Cross Validation")
    
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

            joy_logit, anticipation_logit, trust_logit, surprise_logit, disgust_logit, fear_logit, anger_logit, sadness_logit = model(**inputs)

            # focal loss
            criterion = {
                'joy' : FocalLoss().to(device),
                'anticipation' : FocalLoss().to(device),
                'trust' : FocalLoss().to(device),
                'surprise' : FocalLoss().to(device),
                'disgust' : FocalLoss().to(device),
                'fear' : FocalLoss().to(device),
                'anger' : FocalLoss().to(device),
                'sadness' : FocalLoss().to(device),
            }
            # labels = labels.type(torch.float).clone().detach()
            loss = criterion['joy'](type_logit, labels[::, 0]) + \
                    criterion['anticipation'](type_logit, labels[::, 1]) + \
                    criterion['trust'](type_logit, labels[::, 2]) + \
                    criterion['surprise'](type_logit, labels[::, 3]) + \
                    criterion['disgust'](type_logit, labels[::, 4]) + \
                    criterion['fear'](type_logit, labels[::, 5]) + \
                    criterion['anger'](type_logit, labels[::, 6]) + \
                    criterion['sadness'](type_logit, labels[::, 7])

            outputs = None, \
                        torch.argmax(joy_logit, dim = 1), \
                        torch.argmax(anticipation_logit, dim = 1), \
                        torch.argmax(trust_logit, dim = 1), \
                        torch.argmax(surprise_logit, dim = 1), \
                        torch.argmax(disgust_logit, dim = 1), \
                        torch.argmax(fear_logit, dim = 1), \
                        torch.argmax(anger_logit, dim = 1), \
                        torch.argmax(sadness_logit, dim = 1)

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
            result = multi_label_metrics(predictions=preds, labels=p.label_ids)
            return result

        # trainer = Trainer(
        #     model,
        #     targs,
        #     train_dataset=encoded_tds,
        #     eval_dataset=encoded_vds,
        #     tokenizer=tokenizer,
        #     compute_metrics=compute_metrics
        # )
        trainer = CustomTrainer(
            model=model,
            args=targs,
            train_dataset=encoded_tds, # 학습데이터
            eval_dataset=encoded_vds,  # validation 데이터
            compute_metrics=compute_metrics, # 모델 평가 방식
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        )
        
        trainer.train()
# end main

class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(self,
                 alpha: Optional[Tensor] = None,
                 gamma: float = 0.,
                 reduction: str = 'mean',
                 ignore_index: int = -100):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(
                'Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)

    def __repr__(self):
        arg_keys = ['alpha', 'gamma', 'ignore_index', 'reduction']
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f'{k}={v!r}' for k, v in zip(arg_keys, arg_vals)]
        arg_str = ', '.join(arg_strs)
        return f'{type(self).__name__}({arg_str})'

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss


def focal_loss(alpha: Optional[Sequence] = None,
               gamma: float = 0.,
               reduction: str = 'mean',
               ignore_index: int = -100,
               device='cpu',
               dtype=torch.float32) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha,
        gamma=gamma,
        reduction=reduction,
        ignore_index=ignore_index)
    return fl
                   
if __name__ == "__main__":
    exit(main(parser.parse_args()))
