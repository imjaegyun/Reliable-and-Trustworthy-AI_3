# Marabou Assignment 3

신뢰할 수 있는 인공지능 과제 3 제출용 저장소입니다. Marabou를 사용해 작은
Iris 분류기의 local robustness를 검증합니다.

## 파일 구성

- `test.py`: 모델을 불러오고, 입력 perturbation bound와 출력 제약을 설정한 뒤
  Marabou를 실행합니다. 결과는 `results/verification_result.json`에 저장됩니다.
- `models/iris_tiny.nnet`: Marabou가 직접 읽을 수 있는 `.nnet` 형식의 작은
  완전연결 ReLU 모델입니다.
- `data/iris.csv`: UCI Machine Learning Repository에서 받은 전체 Iris
  데이터셋입니다. `test.py`는 이 파일에서 Setosa 샘플을 읽고 petal length와
  petal width를 정규화해 검증 입력으로 사용합니다.
- `resources_exploration.md`: Marabou `resources` 디렉터리 조사 요약입니다.
- `report.pdf`: 과제 보고서입니다.
- `environment.yml`: conda 환경 정의 파일입니다.
- `run_all.sh`: 환경 생성/재사용, 의존성 설치, Python 문법 검사, Marabou 검증을
  한 번에 실행하는 스크립트입니다.

## 실행 환경

conda 환경 이름은 기본적으로 `assignment3-marabou`입니다. 직접 만들려면 다음을
실행합니다.

```bash
conda env create -f environment.yml
conda activate assignment3-marabou
```

또는 아래 스크립트를 실행하면 conda 환경을 자동으로 만들거나 기존 환경을
재사용합니다.

```bash
./run_all.sh
```

## 실행 방법

기본 검증:

```bash
./run_all.sh
```

epsilon과 timeout을 지정하는 경우:

```bash
./run_all.sh --epsilon 0.04 --timeout 30
```

## 검증 내용

기준 입력은 Setosa 샘플이며, 입력 특성은 정규화된 petal length와 petal width
두 개입니다. 스크립트는 `L_inf` 반경 `epsilon = 0.04` 안에서 다음 두 반례가
존재하는지 각각 확인합니다.

- Versicolor logit이 Setosa logit 이상이 되는 입력
- Virginica logit이 Setosa logit 이상이 되는 입력

두 query가 모두 `UNSAT`이면 해당 perturbation 영역 안에서 Setosa 예측이 유지되어
local robustness가 검증됩니다.

## 참고

실행 중 TensorFlow 또는 ONNX parser가 설치되어 있지 않다는 warning이 나올 수
있습니다. 이번 실험은 `.nnet` 모델을 사용하므로 해당 warning은 실행 결과에
영향을 주지 않습니다.
