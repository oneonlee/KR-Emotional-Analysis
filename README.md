# 한국어 텍스트 감정 분석 모델
본 리포지토리는 [2023 국립국어원 인공 지능 언어 능력 평가](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103) 중 [감정 분석(Emotional Analysis) 과제](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103)를 위한 모델 및 해당 모델의 재현을 위한 소스 코드를 포함하고 있습니다.

### Performance
| Model              | Test Micro-F1 | Batch Size | Epochs | Learing Rate | Weight Decay |
| :----------------- | :------------ | :--------- | :----- | :----------- | :----------- |
| klue/roberta-large | 86.5736336    | 128        | 14     | 2e-5         | 0.1          |
| klue/roberta-base  | 85.3803773    | 64         | 14     | 4e-5         | 0.1          |
| klue/bert-base     | 84.7908872    | 64         | 5      | 2e-5         | 0.1          |


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
	 --model-path klue/roberta-large --tokenizer klue/roberta-large \
    --gpu-num 0
```

### Inference
```
python3 -m run inference \
    --model-ckpt-path outputs/checkpoint-<XXXX> \
    --output-path test_output.jsonl \
    --batch-size 64 \
    --device cuda:0
```

### Reference
- teddysum/Korean_EA_2023 (https://github.com/teddysum/Korean_EA_2023)
- 국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)  
- transformers (https://github.com/huggingface/transformers)  
- KLUE (https://github.com/KLUE-benchmark/KLUE)
