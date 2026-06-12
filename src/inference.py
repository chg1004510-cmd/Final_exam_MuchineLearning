"""
src/inference.py — CardioCare 추론 엔트리포인트

사용법:
    python src/inference.py --input data/sample_batch.csv
    python src/inference.py --input data/sample_batch.csv --output results.csv
    python src/inference.py --input data/sample_batch.csv --model models/best_model.pkl

추론 로그 (logs/inference.log):
    타임스탬프, 모델 버전, 입력 shape, 예측값, 예측 확률 평균 기록
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Windows CP949 터미널에서 한글/특수문자 출력 보장
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import joblib
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import ALL_FEATURES

# ── 모델 버전 & 경로 ────────────────────────────────────────────────────────
MODEL_VERSION      = "1.0"
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
LOG_DIR            = ROOT_DIR / "logs"
LOG_FILE           = LOG_DIR / "inference.log"

# ── 임상적 허용 범위 ─────────────────────────────────────────────────────────
# 이 범위를 벗어난 입력값은 측정 오류 또는 데이터 품질 문제로 간주한다.
# validate_input_ranges() 가 여기서 정의된 범위로 검증을 수행한다.
CLINICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "age":      (1.0,   120.0),   # 나이: 인간 생존 가능 범위
    "trestbps": (50.0,  300.0),   # 안정 혈압(mmHg): 50 미만은 측정 오류 의심
    "chol":     (0.0,   600.0),   # 혈청 콜레스테롤(mg/dl): 0은 결측 가능성
    "thalach":  (40.0,  250.0),   # 최대 심박수(bpm)
    "oldpeak":  (0.0,   10.0),    # 운동 유발 ST 하강: 음수 불가
    "ca":       (0.0,   4.0),     # 주요 혈관 수 0-3 (4는 경계값으로 허용)
}


# ── 로거 설정 ────────────────────────────────────────────────────────────────
def _setup_inference_logger() -> logging.Logger:
    """logs/inference.log 에 기록하는 전용 로거를 반환한다."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _logger = logging.getLogger("cardiocare.inference")
    if not _logger.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # 파일 핸들러 (추론 이력 보관)
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        # 콘솔 핸들러 (Docker stdout)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        _logger.addHandler(fh)
        _logger.addHandler(sh)
        _logger.setLevel(logging.INFO)
    return _logger


logger = _setup_inference_logger()


# ── 입력 검증 ─────────────────────────────────────────────────────────────────
def validate_input_ranges(df: pd.DataFrame) -> None:
    """
    임상적으로 정의된 범위를 벗어나는 특성값이 있으면 ValueError 를 발생시킨다.

    Parameters
    ----------
    df : pd.DataFrame
        추론 입력 데이터프레임. CLINICAL_BOUNDS 에 있는 컬럼만 검사한다.

    Raises
    ------
    ValueError
        범위 위반 컬럼과 위반값 목록이 메시지에 포함된다.

    Notes
    -----
    NaN 은 전처리 파이프라인의 Imputer 가 대치하므로 여기서는 건너뛴다.
    """
    violations: list[str] = []
    for col, (lo, hi) in CLINICAL_BOUNDS.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        out_of_range = series[(series < lo) | (series > hi)]
        if not out_of_range.empty:
            violations.append(
                f"'{col}': 허용 범위 [{lo}, {hi}] 위반 — 값: {out_of_range.tolist()[:5]}"
            )
    if violations:
        raise ValueError(
            "입력 데이터에 임상적 범위 위반이 발견되었습니다:\n  "
            + "\n  ".join(violations)
        )


# ── 추론 함수 ─────────────────────────────────────────────────────────────────
def predict(
    input_path: str | os.PathLike,
    model_path: str | os.PathLike | None = None,
    output_path: str | os.PathLike | None = None,
    skip_validation: bool = False,
) -> pd.DataFrame:
    """
    CSV 파일을 읽어 예측값·확률을 반환하고 추론 결과를 로깅한다.

    Parameters
    ----------
    input_path      : 입력 CSV 파일 경로 (ALL_FEATURES 컬럼 포함)
    model_path      : 모델 pkl 경로. None 이면 DEFAULT_MODEL_PATH 사용.
    output_path     : 결과 CSV 저장 경로. None 이면 저장하지 않음.
    skip_validation : True 이면 임상 범위 검증을 건너뜀.

    Returns
    -------
    pd.DataFrame
        입력 컬럼 + 'predicted' + 'prob_disease' 컬럼

    Logging
    -------
    logs/inference.log 에 다음을 기록한다:
        - 타임스탬프
        - 모델 버전 (MODEL_VERSION)
        - 입력 shape
        - 예측값 (처음 10개)
        - 예측 확률 평균
        - 실제 레이블 (target 컬럼이 있을 경우)
    """
    ts = datetime.now().isoformat(timespec="seconds")

    # 모델 로드
    resolved_model_path = Path(model_path or DEFAULT_MODEL_PATH)
    if not resolved_model_path.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {resolved_model_path}\n"
            "먼저 'python src/train.py' 를 실행하세요."
        )
    pipeline = joblib.load(resolved_model_path)
    logger.info("모델 로드 완료: %s  (version=%s)", resolved_model_path, MODEL_VERSION)

    # 입력 로드
    df_input = pd.read_csv(input_path)
    logger.info("ts=%s  model_version=%s  input_shape=%s",
                ts, MODEL_VERSION, str(df_input.shape))

    # 임상 범위 검증
    if not skip_validation:
        validate_input_ranges(df_input)
        logger.info("임상 범위 검증 통과")

    # 특성 추출 (target 컬럼이 있으면 분리)
    feat_cols = [c for c in ALL_FEATURES if c in df_input.columns]
    missing_cols = set(ALL_FEATURES) - set(feat_cols)
    if missing_cols:
        logger.warning("누락된 특성 컬럼: %s — NaN 으로 대치됩니다.", missing_cols)
        for col in missing_cols:
            df_input[col] = np.nan
        feat_cols = ALL_FEATURES

    X = df_input[feat_cols]
    y_true = df_input["target"] if "target" in df_input.columns else None

    # 예측
    y_pred = pipeline.predict(X)
    y_prob: np.ndarray | None = None
    clf = pipeline.named_steps.get("classifier", None)
    if clf is not None and hasattr(clf, "predict_proba"):
        y_prob = pipeline.predict_proba(X)[:, 1]

    # 추론 결과 로깅
    logger.info(
        "predictions (first 10): %s",
        y_pred[:10].tolist(),
    )
    if y_prob is not None:
        logger.info(
            "prob_disease — mean=%.4f  min=%.4f  max=%.4f",
            float(y_prob.mean()), float(y_prob.min()), float(y_prob.max()),
        )
    if y_true is not None:
        logger.info("actual_labels (first 10): %s", y_true.iloc[:10].tolist())

    # 결과 DataFrame 구성
    result = df_input.copy()
    result["predicted"]    = y_pred
    if y_prob is not None:
        result["prob_disease"] = np.round(y_prob, 4)

    # 저장
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_path, index=False)
        logger.info("결과 저장: %s", output_path)

    return result


# ── CLI 엔트리포인트 ───────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CardioCare — 심장병 예측 추론 스크립트"
    )
    p.add_argument("--input",           required=True,  help="입력 CSV 경로")
    p.add_argument("--output",          default=None,   help="결과 CSV 저장 경로")
    p.add_argument("--model",           default=None,   help="모델 pkl 경로 (기본: models/best_model.pkl)")
    p.add_argument("--skip-validation", action="store_true",
                   help="임상 범위 검증 건너뜀")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = predict(
        input_path=args.input,
        model_path=args.model,
        output_path=args.output,
        skip_validation=args.skip_validation,
    )
    # 주요 컬럼 출력
    show_cols = ["predicted"]
    if "prob_disease" in result.columns:
        show_cols.append("prob_disease")
    if "target" in result.columns:
        show_cols.append("target")
    print("\n[예측 결과]")
    print(result[show_cols].to_string(index=True))
    print(f"\n총 {len(result)}건  |  예측 심장병: {(result['predicted'] == 1).sum()}건")


if __name__ == "__main__":
    main()
