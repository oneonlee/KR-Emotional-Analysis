# 한국어 텍스트 감정 분석 모델
본 리포지토리는 [2023 국립국어원 인공 지능 언어 능력 평가](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103) 중 [감정 분석(Emotional Analysis) 과제](https://corpus.korean.go.kr/taskOrdtm/taskList.do?taskOrdtmId=103)를 위한 모델 및 해당 모델의 재현을 위한 소스 코드를 포함하고 있습니다.

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

Install Python Dependency
```
pip install -r requirements.txt
```

## How to Run
### Train

```
python3 -m run train2 \
    --output-dir outputs/beomi-KcELECTRA-base-v2022/removing_all-stratified_10folds-ES_patience5-ASL_Loss-lr2_5-b64 \
    --seed 42 --epoch 50 --patience 5 --num-folds 10 \
    --learning-rate 2e-5 --weight-decay 0.1 \
    --max-seq-len 200 \
    --batch-size 64 --valid-batch-size 64 \
    --model-path beomi/KcELECTRA-base-v2022 \
    --tokenizer beomi/KcELECTRA-base-v2022 \
    --removing-symbol yes --removing-emoji yes \
    --removing-others yes --cleansing yes \
    --gpu-num 2
```

### Inference


```
python3 -m run inference \
    --model-ckpt-path outputs/beomi-KcELECTRA-base-v2022/removing_all-stratified_10folds-ES_patience5-ASL_Loss-lr2_5-b64/fold_9/e7 \
    --output-path outputs/beomi-KcELECTRA-base-v2022/removing_all-stratified_10folds-ES_patience5-ASL_Loss-lr2_5-b64/fold_9-test_output.jsonl \
    --max-seq-len 200 \
    --batch-size 128 \
    --removing-symbol yes --removing-emoji yes \
    --removing-others yes --cleansing yes \
    --device cuda:3
```

## Reference
- 국립국어원 모두의말뭉치 (https://corpus.korean.go.kr/)  
- teddysum/Korean_EA_2023 (https://github.com/teddysum/Korean_EA_2023)
- Beomi/KcELECTRA (https://github.com/Beomi/KcELECTRA)
- huggingface/transformers (https://github.com/huggingface/transformers)  
