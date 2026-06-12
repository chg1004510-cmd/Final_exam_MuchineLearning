"""
src/preprocessing.py — CardioCare 전처리 파이프라인

[누수(Leakage) 방지 원칙]
  Imputer, IQRClipper, Scaler, OneHotEncoder 등 데이터로부터 학습되는
  모든 변환은 반드시 X_train 에만 .fit() 한 뒤, X_val / X_test 에는
  .transform() 만 적용해야 합니다.
  => build_preprocessing_pipeline() 이 반환하는 ColumnTransformer를
     train_test_split 이후에 fit 하는 것으로 이 원칙을 강제합니다.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── 특성 그룹 정의 ──────────────────────────────────────────────────────────
# EDA 결과:
#   연속형(age, trestbps, chol, thalach, oldpeak) → 중앙값 대치 + IQR 클리핑 + 표준화
#   순서형(ca: 0-3)                               → 중앙값 대치 + IQR 클리핑 + 표준화
#   범주형(cp, restecg, slope, thal)              → 최빈값 대치 + OHE
#   이진(sex, fbs, exang)                         → 최빈값 대치 + OHE
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COL = "target"
RANDOM_SEED = 42

# UCI Cleveland 원본 컬럼 순서
_CLEVELAND_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num",
]


# ── 커스텀 트랜스포머 ──────────────────────────────────────────────────────
class IQROutlierClipper(BaseEstimator, TransformerMixin):
    """
    IQR 방식으로 이상치를 클리핑한다.

    fit()  : 학습 데이터의 Q1 / Q3를 기억 (학습 분포만 참조 → 누수 없음)
    transform(): [Q1 - k*IQR, Q3 + k*IQR] 범위 밖 값을 경계값으로 클리핑

    Parameters
    ----------
    k : float, default 1.5
        IQR 배율. 1.5 = Tukey fence (표준), 3.0 = 극단값만 제거.
    """

    def __init__(self, k: float = 1.5):
        self.k = k

    def fit(self, X, y=None):
        X_arr = np.array(X, dtype=float)
        q1 = np.nanpercentile(X_arr, 25, axis=0)
        q3 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        return self

    def transform(self, X, y=None):
        X_arr = np.array(X, dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        n = len(self.lower_) if hasattr(self, "lower_") else 0
        return np.array([f"x{i}" for i in range(n)], dtype=object)


# ── 파이프라인 빌더 ────────────────────────────────────────────────────────
def build_preprocessing_pipeline(impute_strategy: str = "median") -> ColumnTransformer:
    """
    sklearn ColumnTransformer 기반 전처리 파이프라인을 반환한다.

    Parameters
    ----------
    impute_strategy : {'median', 'mean', 'knn'}
        연속형 특성 결측치 대치 전략.
        - 'median' : 중앙값 (이상치에 강건, 기본값) — EDA 상 chol, ca, thal에 적합
        - 'mean'   : 평균값
        - 'knn'    : KNN 대치 (결측이 2% 미만인 ca/thal에 유리하나 속도 느림)

    Returns
    -------
    ColumnTransformer
        반드시 train split 이후에 .fit(X_train) 을 호출하여 데이터 누수를 방지.

    Pipeline 구성
    -------------
    연속형: SimpleImputer(median) → IQROutlierClipper(k=1.5) → StandardScaler
    범주형: SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown='ignore')

    Notes
    -----
    StandardScaler 를 SVC / Logistic Regression 등 거리·그래디언트 기반 모델에
    사용한다. Tree 계열(Random Forest, XGBoost)은 스케일 불변이지만
    동일 파이프라인을 재사용하므로 포함한다.
    """
    # 연속형 결측치 대치기 선택
    if impute_strategy == "knn":
        num_imputer = KNNImputer(n_neighbors=5)
    elif impute_strategy == "mean":
        num_imputer = SimpleImputer(strategy="mean")
    else:
        num_imputer = SimpleImputer(strategy="median")

    numeric_pipeline = Pipeline(steps=[
        ("imputer", num_imputer),
        ("clipper", IQROutlierClipper(k=1.5)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ── 데이터 로더 ────────────────────────────────────────────────────────────
def load_raw_data(path: str = None) -> pd.DataFrame:
    """
    UCI Heart Disease (Cleveland) 데이터를 ucimlrepo로 로드하고 타깃을 이진화한다.

    Returns
    -------
    pd.DataFrame
        컬럼: age, sex, cp, trestbps, chol, fbs, restecg,
               thalach, exang, oldpeak, slope, ca, thal, target
        target = 0 (정상), 1 (심장병) — 원본 num > 0 이면 1
    """
    from ucimlrepo import fetch_ucirepo  # lazy import: Docker/추론 환경에서 불필요

    heart = fetch_ucirepo(id=45)
    df = heart.data.features.copy()
    df[TARGET_COL] = (heart.data.targets.iloc[:, 0] > 0).astype(int)

    # 빈 컬럼(모두 NaN) 제거
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    if empty_cols:
        df = df.drop(columns=empty_cols)

    # 중복 행 제거
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_before:
        import warnings
        warnings.warn(
            f"[load_raw_data] 중복 행 {n_before - len(df)}개 제거 "
            f"(전: {n_before} → 후: {len(df)})"
        )

    return df
