# DataMining-TrafficPrediction
PKNU 24Fall 데이터마이닝 과제입니다. 
[제주도 도로 교통량 예측 AI 경진대회](https://dacon.io/competitions/official/235985/overview/description)의 데이터를 이용해 도로 교통량을 예측해볼 것입니다.

해당 데이터는 StructeredData이며, 연속형 값은 교통량을 예측하는 문제로 회귀(Regression)으로 모델링합니다.

# Database
DACON에 회원가입 > 연습 > [해당 데이터](https://dacon.io/competitions/official/235985/data) 다운로드 후
`database`라는 폴더에 넣어 주세요. 

# Conda 가상환경
```bash
conda activate damine
```
을 통해 가상환경을 설정할 수 있습니다.

## 활성이 되지 않는 경우
아래의 코드를 입력해 주세요
```bash
conda env create -f environment.yaml
```