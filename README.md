# CardioCare — End-to-End ML System for Heart Disease Prediction

> **CardioCare 는 심장 전문의의 의사결정을 보조하는 도구이며, 절대 단독으로 임상 결정을 내리는 시스템이 아닙니다.**

---

## 프로젝트 구조

```
.
├── data/
│   ├── cleveland.csv          ← 첫 실행 시 ucimlrepo로 자동 다운로드
│   └── sample_batch.csv       ← Docker 추론 테스트용 샘플
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py       ← sklearn Pipeline (IQRClipper, ColumnTransformer)
│   ├── train.py               ← MLflow 실험 추적, 4개 모델, CV, 하이퍼파라미터 튜닝
│   ├── inference.py           ← 추론 엔트리포인트, 임상 범위 검증, logging
│   └── monitor.py             ← KS 드리프트 탐지, 성능 비교, 시계열 모니터링
├── tests/
│   └── test_pipeline.py       ← unittest 4개 (shape / proba / 임상범위 / 결정론성)
├── logs/
│   ├── inference.log          ← inference.py 실행 시 생성
│   └── monitor.log            ← monitor.py 실행 시 생성
├── mlruns/                    ← MLflow 아티팩트 (python src/train.py 실행 후 생성)
├── models/                    ← best_model.pkl (python src/train.py 실행 후 생성)
├── report.pdf                 ← 최종 보고서
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

---

## 전체 재현 절차

### 0. 의존성 설치

```bash
git clone https://github.com/chg1004510-cmd/Final_exam_MuchineLearning.git
cd Final_exam_MuchineLearning
pip install -r requirements.txt
```

### 1. EDA 노트북 실행 (선택 — 브라우저)

```bash
jupyter notebook notebooks/01_eda_preprocessing.ipynb
```

### 2. 모델 학습 & MLflow 기록

```bash
python src/train.py
# → mlruns/ 에 3~4개 계열 모델 실험 기록
# → models/best_model.pkl 저장
```

MLflow UI 확인:
```bash
# Linux/macOS
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri mlruns/

# Windows PowerShell
$env:MLFLOW_ALLOW_FILE_STORE="true"; mlflow ui --backend-store-uri mlruns/

# 브라우저: http://localhost:5000
```

### 3. Docker 빌드 & 실행

```bash
# 빌드 (models/best_model.pkl 은 레포에 포함 → clone 직후 빌드 가능)
docker build -t cardiocare:1.0 .

# 실행 (sample_batch.csv 로 추론)
docker run --rm cardiocare:1.0

# 커스텀 입력
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  cardiocare:1.0 \
  --input data/sample_batch.csv \
  --output /tmp/predictions.csv
```

### 4. 단위 테스트

```bash
python -m unittest discover -s tests -v
# 4개 테스트 모두 OK
```

### 5. 드리프트 모니터링

```bash
python src/monitor.py
# → 로그: logs/monitor.log
# → 시각화 4개: data/fig_monitor*.png
```

---

## 데이터셋

**UCI Heart Disease — Cleveland Clinic Foundation**  
출처: https://archive.ics.uci.edu/dataset/45/heart+disease  
버전: Cleveland Clinic 데이터 (303행 × 13 특성 + 타깃)  
타깃 이진화: `target = (num > 0).astype(int)` (0=정상 / 1=심장병)

**다운로드 방법 (두 가지 중 하나):**

```bash
# 방법 A — train.py 실행 시 자동 취득 (ucimlrepo API 사용, 권장)
python src/train.py          # data/cleveland.csv 가 없으면 자동 다운로드 후 저장

# 방법 B — 직접 취득
python -c "
from src.preprocessing import load_raw_data
df = load_raw_data()
df.to_csv('data/cleveland.csv', index=False)
print(df.shape)              # (303, 14)
"
```

`data/cleveland.csv` 가 이미 있으면 `train.py` / `monitor.py` 모두 로컬 파일을 그대로 사용합니다.  
인터넷 연결이 필요한 것은 최초 1회 다운로드뿐이며, 이후 재현은 오프라인으로 가능합니다.

---

## 필수 도구 스택

| 도구 | 버전 | 용도 |
|------|------|------|
| Python | 3.10+ | 런타임 |
| scikit-learn | 1.5.1 | 파이프라인·모델 |
| pandas / numpy | 2.2.3 / 1.26.4 | 데이터 처리 |
| mlflow | 3.13.0 | 실험 추적 |
| scipy | 1.13.1 | KS 드리프트 검정 |
| joblib | 1.4.2 | 모델 직렬화 |
| ucimlrepo | 0.0.7 | UCI 데이터셋 API 로드 |
| unittest | 표준 라이브러리 | 단위 테스트 |
| Docker | — | 패키징 |
| GitHub Actions | — | CI |

---

## 재현성 보장

- 랜덤 시드 고정: `RANDOM_SEED = 42` (train_test_split·CV·RF·SelectFromModel 모두 적용)
- 의존성 버전 고정: `requirements.txt` 참조
- 결정론성 검증: `tests/test_pipeline.py::test_pipeline_determinism`

---

## 데이터 누수 방지

모든 변환(Imputer, IQRClipper, StandardScaler, OneHotEncoder, SelectFromModel)은  
`build_preprocessing_pipeline()` / `build_full_pipeline()` 내에 캡슐화되며,  
`train_test_split` **이후** `X_train` 에만 `.fit()` 합니다.

---

## §5.3 피처 스토어 & 모델 레지스트리 (서술)

### 피처 스토어에 등록해야 할 피처: `chol` (혈청 콜레스테롤)

`chol` 은 식이 변화·약물 복용 이력에 따라 시간적으로 변동하는 특성으로,  
외부 전자의무기록(EMR) 시스템에서 실시간으로 가져와야 합니다.  
Feast 같은 피처 스토어에 등록해 두면 학습·서빙 시점 모두 동일한  
버전의 특성값을 재사용할 수 있어 **학습-서빙 스큐(skew)** 를 방지합니다.

### 모델 레지스트리에 기록해야 할 메타데이터: `balanced_accuracy` + `recall`

임상 도구의 모델은 단순 accuracy 만으로는 위험 평가가 불충분합니다.  
`balanced_accuracy` (클래스 불균형 보정) 와 `recall` (FN 최소화) 을  
MLflow Model Registry 에 기록하면, 차기 모델 등록 시 이 두 지표가  
이전 버전보다 낮으면 **자동 차단(gate)** 정책을 설정할 수 있습니다.

---

## CI 상태

GitHub Actions (`push` 트리거) → `python -m unittest discover -s tests -v`  
`.github/workflows/ci.yml` 참조

---

## AI 도구 사용 공개 (§8 요구사항)

이 프로젝트의 보일러플레이트 코드 작성과 디버깅 과정에서  
**Claude (Anthropic, claude-sonnet-4-6)** 를 활용하였습니다.  
