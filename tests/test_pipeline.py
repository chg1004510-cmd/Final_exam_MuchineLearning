"""
tests/test_pipeline.py — CardioCare 파이프라인 단위 테스트

실행:
    python -m unittest tests/test_pipeline.py -v
    python -m unittest discover -s tests -v

테스트 목록:
    1. test_prediction_output_shape   — 예측 shape 가 입력 shape 와 일치하는지
    2. test_predict_proba_range_sum   — 확률이 [0,1] 범위이고 행합 ≈ 1 인지
    3. test_clinical_feature_ranges   — 임상 범위 위반 입력에 ValueError 발생 여부
    4. test_pipeline_determinism      — 동일 입력·시드에서 동일 출력이 나오는지

설계 원칙:
    - 합성 데이터를 사용하여 CI 에서 인터넷 연결·사전 학습 모델 없이도 실행 가능.
    - 파이프라인은 setUpClass 에서 X_train 에만 fit → 누수 없음.
    - 각 테스트는 독립적이며 실제 버그를 잡을 수 있도록 의미 있는 assertion 포함.
"""

import sys
import os
import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 프로젝트 루트를 sys.path 에 추가 (tests/ 에서 src/ import 가능하게)
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.preprocessing import ALL_FEATURES, RANDOM_SEED
from src.train import build_full_pipeline
from src.inference import CLINICAL_BOUNDS, validate_input_ranges


# ──────────────────────────────────────────────────────────────────────────────
# 합성 데이터 생성 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def _make_synthetic_data(n: int = 150, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    UCI Heart Disease 데이터셋과 같은 컬럼 구조·값 범위를 가진 합성 데이터를 생성한다.
    CI 에서 실제 데이터 다운로드 없이 테스트를 실행하기 위해 사용한다.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        # 연속형
        "age":      rng.integers(29, 78, n).astype(float),
        "trestbps": rng.integers(94, 201, n).astype(float),
        "chol":     rng.integers(126, 421, n).astype(float),
        "thalach":  rng.integers(71, 203, n).astype(float),
        "oldpeak":  np.round(rng.uniform(0.0, 5.0, n), 1),
        "ca":       rng.integers(0, 4, n).astype(float),
        # 범주형 / 이진
        "sex":      rng.integers(0, 2, n).astype(float),
        "cp":       rng.integers(1, 5, n).astype(float),      # 1-4
        "fbs":      rng.integers(0, 2, n).astype(float),
        "restecg":  rng.integers(0, 3, n).astype(float),
        "exang":    rng.integers(0, 2, n).astype(float),
        "slope":    rng.integers(1, 4, n).astype(float),
        "thal":     rng.choice([3.0, 6.0, 7.0], n),
    })
    # 약 5% 결측치 삽입 (실제 데이터의 특성 반영)
    for col in ["ca", "thal"]:
        mask = rng.random(n) < 0.05
        df.loc[mask, col] = np.nan

    df["target"] = rng.integers(0, 2, n)
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 테스트 클래스
# ──────────────────────────────────────────────────────────────────────────────
class TestCardioCare(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        """
        합성 데이터로 파이프라인을 한 번 훈련시켜 모든 테스트에서 공유한다.
        - train_test_split → X_train 에만 fit (누수 없음)
        - RandomForestClassifier(n_estimators=30) : 빠른 테스트를 위해 트리 수 최소화
        """
        df = _make_synthetic_data(n=160, seed=RANDOM_SEED)
        X  = df[ALL_FEATURES].copy()
        y  = df["target"].copy()

        cls.X_train, cls.X_test, cls.y_train, cls.y_test = train_test_split(
            X, y,
            test_size=0.25,
            random_state=RANDOM_SEED,
            stratify=y,
        )

        # 학습 데이터에만 fit
        cls.pipeline = build_full_pipeline(
            RandomForestClassifier(n_estimators=30, random_state=RANDOM_SEED)
        )
        cls.pipeline.fit(cls.X_train, cls.y_train)

    # ── Test 1 ────────────────────────────────────────────────────────────────
    def test_prediction_output_shape(self) -> None:
        """
        예측 결과의 shape 가 입력 shape 와 일치하는지 검증한다.

        검증 항목:
            - len(y_pred) == len(X_test)  (샘플 수 일치)
            - y_pred.ndim == 1            (1차원 배열)
            - 예측값이 {0, 1} 집합의 부분집합  (이진 분류기)
        """
        y_pred = self.pipeline.predict(self.X_test)

        self.assertEqual(
            y_pred.shape[0], len(self.X_test),
            msg="예측 샘플 수가 입력 샘플 수와 다릅니다.",
        )
        self.assertEqual(
            y_pred.ndim, 1,
            msg="predict() 결과는 1차원 배열이어야 합니다.",
        )
        self.assertTrue(
            set(y_pred.tolist()).issubset({0, 1}),
            msg="이진 분류기의 예측값은 0 또는 1 만 포함해야 합니다.",
        )

        # 단일 샘플 추론도 동일하게 동작해야 함
        single = self.X_test.iloc[[0]]
        y_single = self.pipeline.predict(single)
        self.assertEqual(y_single.shape, (1,), msg="단일 샘플 예측 shape 오류")

    # ── Test 2 ────────────────────────────────────────────────────────────────
    def test_predict_proba_range_and_sum(self) -> None:
        """
        predict_proba() 반환값이 유효한 확률 분포인지 검증한다.

        검증 항목:
            - 모든 확률값 ∈ [0.0, 1.0]
            - 각 행의 합 ≈ 1.0 (atol=1e-6)
            - 반환 shape == (n_samples, n_classes=2)
        """
        proba = self.pipeline.predict_proba(self.X_test)

        # [0, 1] 범위
        self.assertTrue(
            np.all(proba >= 0.0) and np.all(proba <= 1.0),
            msg="확률값이 [0.0, 1.0] 범위를 벗어났습니다.",
        )

        # 행합 ≈ 1.0
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums,
            np.ones(len(row_sums)),
            atol=1e-6,
            err_msg="각 샘플의 클래스 확률 합이 1.0 이 아닙니다.",
        )

        # shape
        self.assertEqual(
            proba.shape,
            (len(self.X_test), 2),
            msg="predict_proba() shape 은 (n_samples, 2) 이어야 합니다.",
        )

        # predict() 는 argmax(proba) 를 사용하므로 동일 방식으로 비교
        # (트리 수가 짝수일 때 50/50 동률 → argmax=0, >=0.5=1 으로 불일치 가능)
        y_pred       = self.pipeline.predict(self.X_test)
        y_from_proba = np.argmax(proba, axis=1)
        np.testing.assert_array_equal(
            y_pred, y_from_proba,
            err_msg="predict() 와 argmax(predict_proba()) 의 결과가 불일치합니다.",
        )

    # ── Test 3 ────────────────────────────────────────────────────────────────
    def test_clinical_feature_ranges(self) -> None:
        """
        임상적으로 범위가 정해진 특성에 대해 입력값 범위 검증 함수를 테스트한다.

        검증 항목:
            - 정상 입력 → 예외 없음
            - chol > 600   → ValueError 발생
            - age < 1      → ValueError 발생
            - oldpeak < 0  → ValueError 발생 (음수 불가)
            - NaN 입력     → 예외 없음 (Imputer 가 처리할 것이므로 통과)
        """
        # 임상 범위 안의 정상 입력 — 예외 없어야 함
        valid_row = pd.DataFrame([{
            "age":      55.0,
            "trestbps": 140.0,
            "chol":     250.0,
            "thalach":  150.0,
            "oldpeak":  1.5,
            "ca":       1.0,
        }])
        try:
            validate_input_ranges(valid_row)
        except ValueError as exc:
            self.fail(f"정상 입력에서 ValueError 발생: {exc}")

        # chol = 700 → 범위 [0, 600] 위반
        bad_chol = valid_row.copy()
        bad_chol["chol"] = 700.0
        with self.assertRaises(ValueError,
                               msg="chol=700 은 ValueError 를 발생시켜야 합니다."):
            validate_input_ranges(bad_chol)

        # age = -5 → 범위 [1, 120] 위반
        bad_age = valid_row.copy()
        bad_age["age"] = -5.0
        with self.assertRaises(ValueError,
                               msg="age=-5 는 ValueError 를 발생시켜야 합니다."):
            validate_input_ranges(bad_age)

        # oldpeak = -1.0 → 범위 [0, 10] 위반
        bad_oldpeak = valid_row.copy()
        bad_oldpeak["oldpeak"] = -1.0
        with self.assertRaises(ValueError,
                               msg="oldpeak=-1.0 은 ValueError 를 발생시켜야 합니다."):
            validate_input_ranges(bad_oldpeak)

        # NaN 값 → 검증을 건너뛰어야 함 (예외 없음)
        nan_row = valid_row.copy()
        nan_row["chol"] = np.nan
        try:
            validate_input_ranges(nan_row)
        except ValueError as exc:
            self.fail(f"NaN 입력에서 ValueError 발생 (NaN 은 Imputer 가 처리): {exc}")

        # CLINICAL_BOUNDS 의 모든 하한·상한이 올바르게 정의되어 있는지 메타 검증
        for col, (lo, hi) in CLINICAL_BOUNDS.items():
            self.assertLess(lo, hi,
                            msg=f"CLINICAL_BOUNDS['{col}'] 에서 lo >= hi 오류")

    # ── Test 4 ────────────────────────────────────────────────────────────────
    def test_pipeline_determinism(self) -> None:
        """
        고정 시드에서 파이프라인이 결정론적인지 검증한다.
        (동일 입력 → 동일 출력 / 동일 확률)

        비결정론적 파이프라인은 재현성 요건을 위반하며,
        Imputer·Clipper·StandardScaler·SelectFromModel·RF 모두
        random_state 또는 결정론적 알고리즘을 사용해야 한다.
        """
        n_estimators = 30

        # 두 개의 독립적인 파이프라인 인스턴스 — 동일 seed
        pipe_a = build_full_pipeline(
            RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED)
        )
        pipe_b = build_full_pipeline(
            RandomForestClassifier(n_estimators=n_estimators, random_state=RANDOM_SEED)
        )

        # 동일 훈련 데이터로 각각 fit
        pipe_a.fit(self.X_train, self.y_train)
        pipe_b.fit(self.X_train, self.y_train)

        # 동일 테스트 데이터로 예측
        pred_a  = pipe_a.predict(self.X_test)
        pred_b  = pipe_b.predict(self.X_test)
        proba_a = pipe_a.predict_proba(self.X_test)
        proba_b = pipe_b.predict_proba(self.X_test)

        # 예측값 동일
        np.testing.assert_array_equal(
            pred_a, pred_b,
            err_msg="동일 seed 로 학습된 두 파이프라인의 predict() 결과가 다릅니다.",
        )
        # 확률값 동일 (부동소수점 허용 오차 내)
        np.testing.assert_allclose(
            proba_a, proba_b, atol=1e-10,
            err_msg="동일 seed 로 학습된 두 파이프라인의 predict_proba() 결과가 다릅니다.",
        )

        # 같은 인스턴스로 같은 입력을 두 번 예측해도 동일해야 함
        pred_c = pipe_a.predict(self.X_test)
        np.testing.assert_array_equal(
            pred_a, pred_c,
            err_msg="동일 파이프라인에서 같은 입력에 대한 반복 예측 결과가 다릅니다.",
        )


# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    unittest.main(verbosity=2)
