"""
src/monitor.py — CardioCare 데이터 드리프트 탐지 & 성능 모니터링

실행:
    python src/monitor.py              # 기본: models/best_model.pkl 사용
    python src/monitor.py --model models/best_model.pkl

§5.4 요구사항 충족 항목:
  1. 추론 경로 logging 계측 — 타임스탬프·모델 버전·입력 shape·예측값·실제 정답
  2. 연속형 특성 분포 인위적 이동 (chol +30 / trestbps +15 / oldpeak ×1.4)
  3. ks_2samp: 훈련 분포 vs 이동 분포, p-value 보고, p<0.05 플래그
  4. 원본 balanced_accuracy vs 드리프트 balanced_accuracy 비교·시각화
  5. 시간에 따른 지표 변화 시계열 그래프 (합성 타임스탬프)
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Windows CP949 터미널에서 한글/특수문자 출력 보장
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import joblib
from scipy.stats import ks_2samp
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

# ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    load_raw_data,
    ALL_FEATURES,
    TARGET_COL,
    RANDOM_SEED,
)
from src.inference import MODEL_VERSION

# ── 상수 ────────────────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
DATA_PATH          = ROOT_DIR / "data" / "cleveland.csv"
LOG_DIR            = ROOT_DIR / "logs"
LOG_FILE           = LOG_DIR / "monitor.log"
FIG_DIR            = ROOT_DIR / "data"

TEST_SIZE     = 0.20
KS_ALPHA      = 0.05        # 드리프트 판정 임계값

# 모니터링 대상 연속형 특성 (KS 검정 수행 대상)
CONT_FEATS = ["age", "trestbps", "chol", "thalach", "oldpeak"]

# 분포 이동 설정
# - mean_shift : 평균 이동량 (단위: 각 특성의 원래 단위)
# - std_mult   : 표준편차 배율 (1.0 = 변화 없음)
# 임상 맥락:
#   chol 상승 (+30 mg/dl) — 식이 변화·스크리닝 대상 인구 변화 시 발생 가능
#   trestbps 상승 (+15 mmHg) — 고령화·고혈압 유병률 증가 시 발생 가능
#   oldpeak 산포 증가 (×1.4) — 운동 강도 프로토콜 변경 시 발생 가능
SHIFT_CFG: dict[str, dict] = {
    "chol":     {"mean_shift": 30.0, "std_mult": 1.5},
    "trestbps": {"mean_shift": 15.0, "std_mult": 1.3},
    "oldpeak":  {"mean_shift": 0.0,  "std_mult": 1.4},
}

# 시계열 시뮬레이션 — chol 을 점진적으로 이동
N_TIME_WINDOWS  = 12
CHOL_STEP       = 10.0      # 창 당 chol 평균 이동량 (+0, +10, +20, ..., +110)
SIM_START_DATE  = datetime(2025, 1, 1)
SIM_INTERVAL    = timedelta(weeks=2)


# ── 로거 설정 ─────────────────────────────────────────────────────────────────
def _setup_monitor_logger() -> logging.Logger:
    """logs/monitor.log 에 기록하는 전용 로거를 반환한다."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    _log = logging.getLogger("cardiocare.monitor")
    if not _log.handlers:
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        _log.addHandler(fh)
        _log.addHandler(sh)
        _log.setLevel(logging.INFO)
    return _log


logger = _setup_monitor_logger()


# ──────────────────────────────────────────────────────────────────────────────
# 1. 추론 경로 logging 계측
# ──────────────────────────────────────────────────────────────────────────────
def log_inference_batch(
    pipeline,
    X: pd.DataFrame,
    y_true: pd.Series | None = None,
    batch_id: str = "batch_0",
) -> tuple[np.ndarray, np.ndarray]:
    """
    배치 추론을 수행하고 필수 항목을 모두 logs/monitor.log 에 기록한다.

    Logging 항목:
        - 타임스탬프 (ISO 8601)
        - 모델 버전  (MODEL_VERSION)
        - 입력 shape
        - 예측값     (처음 10개)
        - 예측 확률 평균·최솟값·최댓값
        - 실제 정답  (y_true 제공 시)
        - balanced_accuracy (y_true 제공 시)

    Returns
    -------
    (y_pred, y_prob) : 예측 레이블, 양성 클래스 확률
    """
    ts = datetime.now().isoformat(timespec="seconds")
    y_pred = pipeline.predict(X)
    y_prob = pipeline.predict_proba(X)[:, 1]

    logger.info(
        "ts=%s  model_version=%s  batch_id=%s  input_shape=%s",
        ts, MODEL_VERSION, batch_id, str(X.shape),
    )
    logger.info(
        "predictions(first10)=%s  prob_mean=%.4f  prob_min=%.4f  prob_max=%.4f",
        y_pred[:10].tolist(),
        float(y_prob.mean()), float(y_prob.min()), float(y_prob.max()),
    )
    if y_true is not None:
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        logger.info(
            "actual(first10)=%s  balanced_accuracy=%.4f",
            y_true.iloc[:10].tolist(),
            bal_acc,
        )
    return y_pred, y_prob


# ──────────────────────────────────────────────────────────────────────────────
# 2. 분포 이동 (Distribution Shift)
# ──────────────────────────────────────────────────────────────────────────────
def apply_distribution_shift(
    X: pd.DataFrame,
    cfg: dict[str, dict] | None = None,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    연속형 특성의 분포를 인위적으로 이동시켜 데이터 드리프트를 시뮬레이션한다.

    각 특성에 대해:
        1. 평균을 mean_shift 만큼 이동
        2. 평균 중심화 후 표준편차를 std_mult 배 확장 → 분산 증가

    Parameters
    ----------
    X   : 원본 테스트 데이터프레임
    cfg : {특성명: {"mean_shift": float, "std_mult": float}} 형태
    """
    X_drifted = X.copy()
    cfg = cfg or SHIFT_CFG
    rng = np.random.default_rng(seed)

    for feat, params in cfg.items():
        if feat not in X_drifted.columns:
            continue
        mean_shift = params.get("mean_shift", 0.0)
        std_mult   = params.get("std_mult",   1.0)

        col = X_drifted[feat].copy()
        col = col + mean_shift                          # 평균 이동
        col_mean = col.mean()
        col = (col - col_mean) * std_mult + col_mean   # 분산 확대

        # 소량의 노이즈 추가 → 실제 드리프트 특성 반영
        noise = rng.normal(0, col.std() * 0.05, size=len(col))
        col   = col + noise
        X_drifted[feat] = col

    return X_drifted


# ──────────────────────────────────────────────────────────────────────────────
# 3. KS 드리프트 탐지
# ──────────────────────────────────────────────────────────────────────────────
def run_ks_drift_detection(
    X_train: pd.DataFrame,
    X_test_drifted: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    """
    훈련 분포 vs 이동된 테스트 분포에 대해 ks_2samp 를 수행한다.

    KS 검정 선택 근거:
        - 분포 형태에 대한 가정 없음 (비모수 검정)
        - 연속형 특성에 적합 (이산형에는 카이제곱 등이 더 적합)
        - scipy.stats.ks_2samp: 두 샘플이 같은 분포에서 왔다는 귀무가설 검정
        - p < α (0.05): 두 분포가 유의미하게 다름 → 드리프트로 판정

    Returns
    -------
    pd.DataFrame
        컬럼: feature, ks_stat, p_value, drifted(bool), flag(str)
    """
    feats = features or CONT_FEATS
    rows: list[dict] = []

    for feat in feats:
        train_vals   = X_train[feat].dropna().values
        drifted_vals = X_test_drifted[feat].dropna().values
        ks_stat, p_val = ks_2samp(train_vals, drifted_vals)
        flagged = bool(p_val < KS_ALPHA)

        rows.append({
            "feature": feat,
            "ks_stat": round(float(ks_stat), 4),
            "p_value": round(float(p_val),   6),
            "mean_train":   round(float(np.nanmean(train_vals)),   2),
            "mean_drifted": round(float(np.nanmean(drifted_vals)), 2),
            "mean_delta":   round(float(np.nanmean(drifted_vals) - np.nanmean(train_vals)), 2),
            "drifted":      flagged,
            "flag":         "DRIFT" if flagged else "OK",
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────────────
# 4. 시계열 드리프트 시뮬레이션
# ──────────────────────────────────────────────────────────────────────────────
def simulate_timeseries_drift(
    pipeline,
    X_train: pd.DataFrame,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
) -> pd.DataFrame:
    """
    chol 드리프트를 점진적으로 증가시키며 시간에 따른 성능 변화를 시뮬레이션한다.

    각 시간 창(time window):
        - chol 에 i × CHOL_STEP 만큼 평균 이동 (i=0~N_TIME_WINDOWS-1)
        - KS 통계량·p-value 기록
        - balanced_accuracy 기록

    합성 타임스탬프를 사용하여 시계열 그래프를 생성한다.
    """
    records: list[dict] = []

    for i in range(N_TIME_WINDOWS):
        chol_shift = float(i * CHOL_STEP)
        ts = SIM_START_DATE + i * SIM_INTERVAL

        # chol 만 이동
        X_shifted = X_test.copy()
        X_shifted["chol"] = X_shifted["chol"] + chol_shift

        # KS 검정
        ks_stat, p_val = ks_2samp(
            X_train["chol"].dropna().values,
            X_shifted["chol"].dropna().values,
        )

        # 성능 평가
        y_pred   = pipeline.predict(X_shifted)
        bal_acc  = balanced_accuracy_score(y_test, y_pred)

        # 추론 로깅 (시계열 각 창마다 기록)
        y_prob = pipeline.predict_proba(X_shifted)[:, 1]
        logger.info(
            "[TimeSeries] ts=%s  window=%02d  chol_shift=+%.0f  "
            "ks_stat=%.4f  p_val=%.6f  balanced_accuracy=%.4f",
            ts.strftime("%Y-%m-%d"), i, chol_shift,
            ks_stat, p_val, bal_acc,
        )

        records.append({
            "timestamp":         ts,
            "window":            i,
            "chol_shift":        chol_shift,
            "balanced_accuracy": round(bal_acc, 4),
            "ks_stat":           round(ks_stat, 4),
            "ks_p_value":        round(p_val,   6),
            "drift_flagged":     bool(p_val < KS_ALPHA),
        })

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────────────
# 시각화 함수들
# ──────────────────────────────────────────────────────────────────────────────
def plot_distribution_shift(
    X_train: pd.DataFrame,
    X_test_orig: pd.DataFrame,
    X_test_drift: pd.DataFrame,
    features: list[str],
    save_path: str,
) -> None:
    """훈련·원본 테스트·드리프트 테스트의 분포를 KDE 로 비교한다."""
    n = len(features)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, feat in zip(axes, features):
        X_train[feat].dropna().plot.kde(
            ax=ax, label="Train",          color="#4C72B0", lw=2, linestyle="-"
        )
        X_test_orig[feat].dropna().plot.kde(
            ax=ax, label="Test (orig)",    color="#55A868", lw=2, linestyle="--"
        )
        X_test_drift[feat].dropna().plot.kde(
            ax=ax, label="Test (drifted)", color="#DD8452", lw=2, linestyle=":"
        )
        ax.set_title(feat, fontsize=11)
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)

    plt.suptitle("Distribution Shift: Train vs Original Test vs Drifted Test",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_ks_results(ks_df: pd.DataFrame, save_path: str) -> None:
    """KS 검정 결과 — 특성별 p-value 막대 그래프 (p=0.05 기준선 포함)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    colors_p = ["#DD8452" if d else "#4C72B0" for d in ks_df["drifted"]]
    colors_k = ["#DD8452" if d else "#4C72B0" for d in ks_df["drifted"]]

    # 왼쪽: p-value
    ax0 = axes[0]
    bars = ax0.bar(ks_df["feature"], ks_df["p_value"], color=colors_p, edgecolor="black")
    ax0.axhline(KS_ALPHA, color="red", linestyle="--", lw=1.5,
                label=f"α = {KS_ALPHA} (drift threshold)")
    ax0.set_title("KS Test — p-value per Feature", fontsize=11)
    ax0.set_ylabel("p-value")
    ax0.set_ylim(0, max(ks_df["p_value"].max() * 1.3, KS_ALPHA * 3))
    ax0.legend(fontsize=9)
    for bar, val in zip(bars, ks_df["p_value"]):
        ax0.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=8)

    # 오른쪽: KS 통계량
    ax1 = axes[1]
    bars2 = ax1.bar(ks_df["feature"], ks_df["ks_stat"], color=colors_k, edgecolor="black")
    ax1.set_title("KS Statistic per Feature (higher = more drift)", fontsize=11)
    ax1.set_ylabel("KS Statistic")
    for bar, val in zip(bars2, ks_df["ks_stat"]):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005, f"{val:.4f}",
                 ha="center", va="bottom", fontsize=8)

    # 범례 설명
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#DD8452", label="DRIFT (p < 0.05)"),
        Patch(facecolor="#4C72B0", label="OK    (p ≥ 0.05)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center",
               ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.suptitle("KS Drift Detection Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_accuracy_comparison(
    orig_metrics: dict,
    drift_metrics: dict,
    save_path: str,
) -> None:
    """원본 테스트셋 vs 드리프트 테스트셋의 주요 지표를 나란히 비교한다."""
    metric_keys  = ["balanced_accuracy", "recall", "precision", "f1"]
    metric_labels = ["Balanced\nAccuracy", "Recall\n(Sensitivity)", "Precision", "F1-Score"]

    orig_vals  = [orig_metrics.get(k, 0)  for k in metric_keys]
    drift_vals = [drift_metrics.get(k, 0) for k in metric_keys]

    x = np.arange(len(metric_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, orig_vals,  width, label="Original Test",
                   color="#4C72B0", edgecolor="black", alpha=0.85)
    bars2 = ax.bar(x + width / 2, drift_vals, width, label="Drifted Test",
                   color="#DD8452", edgecolor="black", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Model Performance: Original vs Drifted Test Set", fontsize=12)
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="gray", linestyle=":", lw=1)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, color="#2c3e50")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{bar.get_height():.3f}", ha="center", fontsize=9, color="#c0392b")

    # 성능 하락 주석
    drop = orig_vals[0] - drift_vals[0]
    ax.annotate(
        f"BalAcc drop: {drop:+.3f}",
        xy=(x[0] + width / 2, drift_vals[0]),
        xytext=(x[0] + 0.7, drift_vals[0] + 0.12),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9, color="red",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_timeseries(ts_df: pd.DataFrame, save_path: str) -> None:
    """
    시간에 따른 balanced_accuracy 와 KS 통계량의 변화를 이중 축 그래프로 표시한다.
    드리프트 감지 구간은 배경색으로 강조한다.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    # 드리프트 구간 배경 강조
    for _, row in ts_df.iterrows():
        if row["drift_flagged"]:
            ax1.axvspan(
                row["timestamp"] - SIM_INTERVAL / 2,
                row["timestamp"] + SIM_INTERVAL / 2,
                alpha=0.12, color="tomato",
            )

    # balanced_accuracy 선
    line1, = ax1.plot(
        ts_df["timestamp"], ts_df["balanced_accuracy"],
        color="#4C72B0", marker="o", linewidth=2.0, markersize=6,
        label="Balanced Accuracy",
    )
    ax1.set_xlabel("Date (synthetic)", fontsize=11)
    ax1.set_ylabel("Balanced Accuracy", fontsize=11, color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.set_ylim(0.3, 1.05)
    ax1.axhline(0.5, color="#4C72B0", linestyle=":", lw=1, alpha=0.5)

    # KS 통계량 선
    line2, = ax2.plot(
        ts_df["timestamp"], ts_df["ks_stat"],
        color="#DD8452", marker="s", linewidth=2.0, markersize=5,
        linestyle="--", label="KS Statistic (chol)",
    )
    ax2.axhline(0.3, color="#DD8452", linestyle=":", lw=1, alpha=0.5,
                label="KS alert level (0.3)")
    ax2.set_ylabel("KS Statistic", fontsize=11, color="#DD8452")
    ax2.tick_params(axis="y", labelcolor="#DD8452")
    ax2.set_ylim(0, 1.0)

    # chol shift 주석
    for _, row in ts_df.iterrows():
        ax1.annotate(
            f"+{row['chol_shift']:.0f}",
            xy=(row["timestamp"], row["balanced_accuracy"]),
            xytext=(0, 8), textcoords="offset points",
            fontsize=7, ha="center", color="#2c3e50",
        )

    # 범례 통합
    lines  = [line1, line2]
    labels = [l.get_label() for l in lines]
    from matplotlib.patches import Patch
    lines.append(Patch(facecolor="tomato", alpha=0.3, label="Drift Flagged (p<0.05)"))
    labels.append("Drift Flagged (p<0.05)")
    ax1.legend(lines, labels, loc="lower left", fontsize=9)

    fig.suptitle(
        "Time-Series Monitoring: Performance Degradation under chol Drift\n"
        "(synthetic timestamps — annotations show chol mean shift amount)",
        fontsize=11, fontweight="bold",
    )
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", save_path)


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main(model_path: str | Path | None = None) -> None:
    logger.info("=" * 64)
    logger.info("CardioCare — 드리프트 탐지 & 성능 모니터링 시작")
    logger.info("=" * 64)

    # ── 1. 모델 & 데이터 로드 ────────────────────────────────────────────────
    resolved_model = Path(model_path or DEFAULT_MODEL_PATH)
    if not resolved_model.exists():
        logger.error(
            "모델 파일을 찾을 수 없습니다: %s\n"
            "먼저 'python src/train.py' 를 실행하세요.", resolved_model
        )
        sys.exit(1)

    pipeline = joblib.load(resolved_model)
    logger.info("모델 로드: %s  (version=%s)", resolved_model, MODEL_VERSION)

    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        logger.info("UCI 에서 다운로드 중 ...")
        df = load_raw_data()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y,
    )
    logger.info("데이터 분할 — Train: %s  Test: %s", X_train.shape, X_test.shape)

    # ── 2. 원본 테스트셋 추론 & 로깅 ────────────────────────────────────────
    print("\n" + "=" * 64)
    print("§1  원본 테스트셋 추론 (logging 계측)")
    print("=" * 64)
    y_pred_orig, y_prob_orig = log_inference_batch(
        pipeline, X_test, y_test, batch_id="original_test"
    )

    from sklearn.metrics import precision_score, recall_score, f1_score
    orig_metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_orig),
        "precision":  precision_score(y_test, y_pred_orig, zero_division=0),
        "recall":     recall_score(y_test, y_pred_orig, zero_division=0),
        "f1":         f1_score(y_test, y_pred_orig, zero_division=0),
    }
    print(f"  Balanced Accuracy : {orig_metrics['balanced_accuracy']:.4f}")
    print(f"  Recall            : {orig_metrics['recall']:.4f}")
    print(f"  Precision         : {orig_metrics['precision']:.4f}")
    print(f"  F1-Score          : {orig_metrics['f1']:.4f}")

    # ── 3. 분포 이동 적용 ───────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("§2  분포 이동 적용 (Distribution Shift)")
    print("=" * 64)
    X_test_drifted = apply_distribution_shift(X_test, cfg=SHIFT_CFG)

    print("  이동 전후 평균 비교:")
    print(f"  {'Feature':<12} {'Mean(orig)':>12} {'Mean(drift)':>12} {'Delta':>10}")
    print("  " + "-" * 50)
    for feat in CONT_FEATS:
        m_orig  = X_test[feat].mean()
        m_drift = X_test_drifted[feat].mean()
        print(f"  {feat:<12} {m_orig:>12.2f} {m_drift:>12.2f} {m_drift - m_orig:>+10.2f}")

    # ── 4. 드리프트 테스트셋 추론 & 로깅 ────────────────────────────────────
    print("\n" + "=" * 64)
    print("§3  드리프트 테스트셋 추론 (logging 계측)")
    print("=" * 64)
    y_pred_drift, y_prob_drift = log_inference_batch(
        pipeline, X_test_drifted, y_test, batch_id="drifted_test"
    )

    drift_metrics = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred_drift),
        "precision":  precision_score(y_test, y_pred_drift, zero_division=0),
        "recall":     recall_score(y_test, y_pred_drift, zero_division=0),
        "f1":         f1_score(y_test, y_pred_drift, zero_division=0),
    }
    print(f"  Balanced Accuracy : {drift_metrics['balanced_accuracy']:.4f}")
    print(f"  Recall            : {drift_metrics['recall']:.4f}")
    print(f"  Precision         : {drift_metrics['precision']:.4f}")
    print(f"  F1-Score          : {drift_metrics['f1']:.4f}")

    # ── 5. 성능 비교 ────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("§4  원본 vs 드리프트 성능 비교")
    print("=" * 64)
    delta_bal = drift_metrics["balanced_accuracy"] - orig_metrics["balanced_accuracy"]
    delta_rec = drift_metrics["recall"]            - orig_metrics["recall"]
    print(f"  {'Metric':<22} {'Original':>10} {'Drifted':>10} {'Delta':>10}")
    print("  " + "-" * 55)
    for k in ["balanced_accuracy", "recall", "precision", "f1"]:
        orig_v  = orig_metrics[k]
        drift_v = drift_metrics[k]
        delta   = drift_v - orig_v
        flag    = " ← 주의" if abs(delta) > 0.05 else ""
        print(f"  {k:<22} {orig_v:>10.4f} {drift_v:>10.4f} {delta:>+10.4f}{flag}")

    logger.info(
        "성능 비교 — orig_bal_acc=%.4f  drift_bal_acc=%.4f  delta=%.4f",
        orig_metrics["balanced_accuracy"], drift_metrics["balanced_accuracy"], delta_bal
    )

    # ── 6. KS 드리프트 탐지 ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("§5  KS 드리프트 탐지 (훈련 분포 vs 이동 분포)")
    print("=" * 64)
    ks_df = run_ks_drift_detection(X_train, X_test_drifted, CONT_FEATS)

    print(f"  {'Feature':<12} {'KS Stat':>9} {'p-value':>12} {'Mean△':>9} {'Status':>12}")
    print("  " + "-" * 58)
    for _, row in ks_df.iterrows():
        print(
            f"  {row['feature']:<12} {row['ks_stat']:>9.4f}"
            f" {row['p_value']:>12.6f} {row['mean_delta']:>+9.2f}"
            f" {'[DRIFT]' if row['drifted'] else '[OK]':>12}"
        )

    flagged = ks_df[ks_df["drifted"]]["feature"].tolist()
    print(f"\n  드리프트 플래그 특성 (p < {KS_ALPHA}): {flagged if flagged else '없음'}")
    logger.info("KS 드리프트 플래그 특성: %s", flagged)

    # ── 7. 시계열 드리프트 시뮬레이션 ────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"§6  시계열 드리프트 시뮬레이션 ({N_TIME_WINDOWS} windows)")
    print("=" * 64)
    ts_df = simulate_timeseries_drift(pipeline, X_train, X_test, y_test)

    print(f"  {'Date':<12} {'chol+':>7} {'BalAcc':>8} {'KS Stat':>9} {'p-val':>10} {'Flag':>8}")
    print("  " + "-" * 60)
    for _, row in ts_df.iterrows():
        flag_str = "[DRIFT]" if row["drift_flagged"] else "[OK]"
        print(
            f"  {row['timestamp'].strftime('%Y-%m-%d'):<12}"
            f" {row['chol_shift']:>+7.0f}"
            f" {row['balanced_accuracy']:>8.4f}"
            f" {row['ks_stat']:>9.4f}"
            f" {row['ks_p_value']:>10.6f}"
            f" {flag_str:>8}"
        )

    first_drift_window = ts_df[ts_df["drift_flagged"]]["window"].min()
    if not np.isnan(first_drift_window):
        first_drift_row = ts_df[ts_df["window"] == first_drift_window].iloc[0]
        print(f"\n  첫 드리프트 감지: window={int(first_drift_window)}"
              f"  chol_shift=+{first_drift_row['chol_shift']:.0f}"
              f"  p={first_drift_row['ks_p_value']:.6f}")

    # ── 8. 시각화 저장 ─────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("§7  시각화 저장")
    print("=" * 64)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    p_dist  = str(FIG_DIR / "fig_monitor1_distribution_shift.png")
    p_ks    = str(FIG_DIR / "fig_monitor2_ks_results.png")
    p_comp  = str(FIG_DIR / "fig_monitor3_accuracy_comparison.png")
    p_ts    = str(FIG_DIR / "fig_monitor4_timeseries.png")

    plot_distribution_shift(X_train, X_test, X_test_drifted, CONT_FEATS, p_dist)
    plot_ks_results(ks_df, p_ks)
    plot_accuracy_comparison(orig_metrics, drift_metrics, p_comp)
    plot_timeseries(ts_df, p_ts)

    print(f"  {p_dist}")
    print(f"  {p_ks}")
    print(f"  {p_comp}")
    print(f"  {p_ts}")

    # ── 9. 최종 요약 ───────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("모니터링 요약")
    print("=" * 64)
    print(f"  드리프트 플래그 특성       : {flagged}")
    print(f"  원본 테스트 Balanced Acc   : {orig_metrics['balanced_accuracy']:.4f}")
    print(f"  드리프트 테스트 Balanced Acc: {drift_metrics['balanced_accuracy']:.4f}")
    print(f"  성능 변화 (Δ)              : {delta_bal:+.4f}")
    print(f"  Recall 변화 (Δ)            : {delta_rec:+.4f}")
    if not np.isnan(first_drift_window):
        print(f"  처음 드리프트 감지 시점    : chol +{first_drift_row['chol_shift']:.0f} mg/dl")
    else:
        print("  처음 드리프트 감지 시점    : 없음 (모든 창에서 p ≥ 0.05)")
    print()
    print("재학습 권고:")
    if abs(delta_bal) > 0.05 or len(flagged) > 0:
        print("  [권고] 드리프트 감지 및 성능 저하 확인 → 재학습 트리거 고려")
        print("  [Human-in-the-loop] 심장 전문의 검토 후 재학습 여부 최종 결정")
    else:
        print("  [정상] 드리프트 미감지 — 모니터링 계속")
    print("=" * 64)
    logger.info("모니터링 완료")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CardioCare 드리프트 모니터링")
    parser.add_argument("--model", default=None, help="모델 pkl 경로")
    args = parser.parse_args()
    main(model_path=args.model)
