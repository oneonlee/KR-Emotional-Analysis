# 한국어 텍스트 감정 분석 모델
본 리포지토리는 [2023 국립국어원 인공 지능 언어 능력 평가](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103) 중 [감정 분석(Emotional Analysis) 과제](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103)를 위한 모델 및 해당 모델의 재현을 위한 소스 코드를 포함하고 있습니다.

### Performance


| Model                                     | Test Micro-F1 | Batch Size | Epochs | Learing Rate | Weight Decay | Removing "&others&" | Cleansing symobls |
| :---------------------------------------- | :------------ | :--------- | :----- | :----------- | :----------- | :------------------ | :---------------- |
| beomi/KcELECTRA-base-v2022                | 87.2167629    | 64         | 14     | 2e-5         | 0.1          | True                | True              |
| beomi/KcELECTRA-base-v2022                | 86.9470899    | 64         | 14     | 2e-5         | 0.1          | False               | True              |
| klue/roberta-large                        | 86.5736336    | 64         | 14     | 2e-5         | 0.1          | False               | False             |
| klue/roberta-large                        | 86.4393056    | 64         | 14     | 2e-5         | 0.1          | False               | True              |
| klue/roberta-large                        | 86.1970606    | 64         | 14     | 2e-5         | 0.1          | True                | True              |
| klue/roberta-base                         | 85.3803773    | 128        | 14     | 4e-5         | 0.1          | False               | False             |
| klue/bert-base                            | 84.7908872    | 128        | 5      | 2e-5         | 0.1          | False               | False             |
| kykim/bert-kor-base                       | 85.8253457    | 128        | 4      | 2e-5         | 0.1          | False               | False             |
| kykim/bert-kor-base                       |               | 128        |        | 2e-5         | 0.1          | True                | False             |
| kykim/funnel-kor-base                     | 85.3276541    | 128        | 10     | 2e-5         | 0.1          | False               | False             |
| kykim/funnel-kor-base                     |               | 128        |        | 2e-5         | 0.1          | True                | False             |
| kykim/electra-kor-base                    | 85.0007948    | 128        | 14     | 2e-5         | 0.1          | False               | False             |
| kykim/electra-kor-base                    |               | 128        | 14     | 2e-5         | 0.1          | True                | False             |
| monologg/koelectra-base-v3-goemotions     |               |            |        |              |              |                     |                   |
| monologg/kocharelectra-base-discriminator |               |            |        |              |              |                     |                   |
| quantumaikr/KoreanLM                      |               |            |        |              |              |                     |                   |


## Directory Structue
```
resource
└── data

# Executable python script
run
├── infernece.py
└── train.py

# Python dependency file
requirements.txt
```

## Data Format
```
{
    "id": "nikluge-2023-ea-dev-000001",
    "input": {
        "form": "하,,,,내일 옥상다이브 하기 전에 표 구하길 기도해주세요",
        "target": {
            "form": "표",
            "begin": 20,
            "end": 21
        }
    },
    "output": {
        "joy": "False",
        "anticipation": "True",
        "trust": "False",
        "surprise": "False",
        "disgust": "False",
        "fear": "False",
        "anger": "False",
        "sadness": "False"
    }
}
```


## Enviroments
Docker Image
```
docker pull nvcr.io/nvidia/pytorch:22.08-py3 
```

Docker Run Script
```
docker run -dit --gpus all --shm-size=8G --name baseline_ea nvcr.io/nvidia/pytorch:22.08-py3
```

Install Python Dependency
```
pip install -r requirements.txt
```

## How to Run
### Train
```
python3 -m run train \
    --output-dir outputs/ \
    --seed 42 --epoch 14 \
    --learning-rate 2e-5 --weight-decay 0.1 \
    --batch-size 64 --valid-batch-size 64 \
    --model-path beomi/KcELECTRA-base-v2022 --tokenizer beomi/KcELECTRA-base-v2022 \
    --removing-others yes --cleansing yes \
    --gpu-num 0
```

### Inference
```
python3 -m run inference \
    --model-ckpt-path outputs/checkpoint-<XXXX> \
    --output-path test_output.jsonl \
    --batch-size 64 \
    --removing-others yes --cleansing yes \
    --device cuda:0
```

### Reference
- 국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)  
- teddysum/Korean_EA_2023 (https://github.com/teddysum/Korean_EA_2023)
- Beomi/KcELECTRA (https://github.com/Beomi/KcELECTRA)
- KLUE-benchmark/KLUE (https://github.com/KLUE-benchmark/KLUE)
- kiyoungkim1/LMkor (https://github.com/kiyoungkim1/LMkor)
- huggingface/transformers (https://github.com/huggingface/transformers)  
