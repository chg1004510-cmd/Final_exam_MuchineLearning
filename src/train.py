"""
src/train.py — CardioCare 모델 학습, 특성 선택, MLflow 실험 추적

실행:
    python src/train.py

산출물:
    - mlruns/  : MLflow 실험 아티팩트 (파라미터·지표·혼동행렬·모델)
    - models/best_model.pkl : 최종 선택 모델 (inference.py 가 로드)

누수(Leakage) 방지 설계:
    모든 sklearn Pipeline 은 build_full_pipeline() 으로 생성하며,
    전처리·특성선택·스케일러가 전부 Pipeline 내부에 캡슐화된다.
    train_test_split 이후 X_train 에만 .fit() 하므로 테스트셋 정보가
    어떤 단계에서도 학습 과정에 유입되지 않는다.
"""

import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# MLflow 3.x 에서 파일 스토어 사용을 허용 (유지 관리 모드 경고 우회)
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

# Windows CP949 터미널에서 한글/특수문자 출력 보장
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")   # GUI 없는 환경에서 플롯 저장
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import joblib
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# ── 경로 설정 ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    ALL_FEATURES,
    TARGET_COL,
    RANDOM_SEED,
    build_preprocessing_pipeline,
    load_raw_data,
)

# ── 상수 ────────────────────────────────────────────────────────────────────
DATA_PATH            = ROOT_DIR / "data" / "cleveland.csv"
MODELS_DIR           = ROOT_DIR / "models"
MLFLOW_TRACKING_URI  = (ROOT_DIR / "mlruns").as_uri()  # Windows: file:///F:/... 형식으로 변환
EXPERIMENT_NAME      = "CardioCare_HeartDisease"
BEST_MODEL_PATH      = MODELS_DIR / "best_model.pkl"

# 분할 비율: 80/20
# - 소규모 데이터(303행)에서 test=20%는 약 60행 — 통계적으로 의미 있는 최소 크기.
# - stratify=y 로 클래스 비율을 양 셋에서 동일하게 유지한다.
TEST_SIZE     = 0.20
CV_FOLDS      = 5
TUNING_N_ITER = 20            # RandomizedSearchCV 탐색 횟수
PRIMARY_SCORE = "balanced_accuracy"   # 주 평가 지표

# ── 후보 모델 정의 ──────────────────────────────────────────────────────────
# class_weight='balanced': 소수 클래스 가중치 자동 보정 — FN 비용을 줄이는 기본 설정.
# probability=True (SVC): ROC-AUC 계산과 임상적 임계값 조정을 위해 활성화.
CANDIDATE_MODELS: dict = {
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=1000,
        random_state=RANDOM_SEED, class_weight="balanced",
    ),
    "SVC": SVC(
        C=1.0, kernel="rbf", probability=True,
        random_state=RANDOM_SEED, class_weight="balanced",
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1,
    ),
    "KNN": KNeighborsClassifier(
        n_neighbors=9, weights="distance",
    ),
}

# RF 하이퍼파라미터 탐색 공간
# RandomizedSearchCV 를 선택한 이유:
#   - GridSearch 대비 동일 n_iter 에서 더 넓은 공간을 커버.
#   - 소규모 데이터셋에서 과도한 그리드 탐색은 과적합 위험이 있음.
RF_PARAM_DIST: dict = {
    "classifier__n_estimators":      randint(50, 400),
    "classifier__max_depth":         [None, 5, 10, 15, 20],
    "classifier__min_samples_split": randint(2, 12),
    "classifier__min_samples_leaf":  randint(1, 6),
    "classifier__max_features":      ["sqrt", "log2", 0.4, 0.6],
}

# ── 로거 ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("cardiocare.train")


# ──────────────────────────────────────────────────────────────────────────────
# 파이프라인 빌더
# ──────────────────────────────────────────────────────────────────────────────
def build_full_pipeline(model) -> Pipeline:
    """
    전처리(ColumnTransformer) → 특성 선택(SelectFromModel) → 분류기
    로 구성된 단일 sklearn Pipeline 을 반환한다.

    특성 선택 설계:
        내부 RF 의 feature importances 로 threshold='median' 이상인
        특성만 유지한다. 비선형 관계를 포착하므로 LR·SVC 같은 선형
        모델의 전처리 단계로도 효과적이다.
        파이프라인 내부에서 fit → 학습 fold 에만 적용 → 누수 없음.
    """
    preprocessor = build_preprocessing_pipeline(impute_strategy="median")
    selector = SelectFromModel(
        estimator=RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1,
        ),
        threshold="median",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("selector",     selector),
        ("classifier",   model),
    ])


def get_selected_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """피팅된 파이프라인에서 SelectFromModel 이 선택한 특성 이름 목록을 반환한다."""
    try:
        prep  = fitted_pipeline.named_steps["preprocessor"]
        sel   = fitted_pipeline.named_steps["selector"]
        names = np.array(prep.get_feature_names_out())
        mask  = sel.get_support()
        if len(names) != len(mask):
            logger.warning(
                "get_selected_feature_names: shape mismatch names=%d mask=%d",
                len(names), len(mask),
            )
            return []
        return names[mask].tolist()
    except Exception as exc:
        logger.warning("get_selected_feature_names 실패: %s", exc)
        return []


# ──────────────────────────────────────────────────────────────────────────────
# 지표 & 시각화 헬퍼
# ──────────────────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """balanced_accuracy / precision / recall / f1 / confusion matrix 값을 반환한다."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "cm_tp": int(tp), "cm_fp": int(fp),
        "cm_fn": int(fn), "cm_tn": int(tn),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            pass
    return metrics


def _save_cm_figure(y_true, y_pred, title: str, save_dir: str) -> str:
    """혼동행렬 그림을 저장하고 경로를 반환한다."""
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_true, y_pred),
        display_labels=["Normal(0)", "Disease(1)"],
    ).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    path = os.path.join(save_dir, f"cm_{title.replace(' ', '_')}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ──────────────────────────────────────────────────────────────────────────────
# MLflow 단일 실험 실행
# ──────────────────────────────────────────────────────────────────────────────
def mlflow_run_model(
    model_name: str,
    pipeline: Pipeline,
    X_train, y_train,
    X_test,  y_test,
    extra_tags: dict | None = None,
) -> dict:
    """
    단일 모델을 학습·평가하고 모든 결과를 MLflow 에 기록한다.

    MLflow 기록 내용:
        Tags    : model_family, dataset, run_type
        Params  : test_size, seed, 모델 하이퍼파라미터, 선택된 특성 수
        Metrics : balanced_accuracy, precision, recall, f1, roc_auc,
                  cm_tp/fp/fn/tn
        Artifacts: 혼동행렬 이미지, classification report, 선택 특성 목록, 모델
    """
    with mlflow.start_run(run_name=model_name) as run:
        run_id = run.info.run_id

        # 태그
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("dataset",      "UCI_Cleveland")
        mlflow.set_tag("run_type",     "baseline")
        if extra_tags:
            for k, v in extra_tags.items():
                mlflow.set_tag(k, str(v))

        # 학습 — X_train 에만 fit (누수 없음)
        pipeline.fit(X_train, y_train)

        # 예측
        y_pred = pipeline.predict(X_test)
        y_prob = None
        clf = pipeline.named_steps["classifier"]
        if hasattr(clf, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]

        # 선택된 특성
        selected_feats = get_selected_feature_names(pipeline)
        n_selected     = len(selected_feats)

        # 파라미터 로깅
        params: dict = {
            "test_size":        TEST_SIZE,
            "random_seed":      RANDOM_SEED,
            "selector":         "SelectFromModel(RF, threshold=median)",
            "n_selected_feats": n_selected,
        }
        _clf_param_keys = ["C", "kernel", "n_estimators", "max_depth",
                           "min_samples_split", "n_neighbors", "penalty"]
        for key in _clf_param_keys:
            val = getattr(clf, key, None)
            if val is not None:
                params[key] = str(val)
        mlflow.log_params(params)

        # 지표 로깅
        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(
            {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        )

        # 아티팩트 저장
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1) 혼동행렬 이미지
            cm_path = _save_cm_figure(y_test, y_pred, model_name, tmpdir)
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")

            # 2) classification report 텍스트
            report_txt  = classification_report(
                y_test, y_pred, target_names=["Normal(0)", "Disease(1)"]
            )
            report_path = os.path.join(tmpdir, f"report_{model_name}.txt")
            Path(report_path).write_text(
                f"Model: {model_name}\n\n{report_txt}", encoding="utf-8"
            )
            mlflow.log_artifact(report_path, artifact_path="classification_reports")

            # 3) 선택된 특성 목록
            feats_path = os.path.join(tmpdir, f"feats_{model_name}.json")
            Path(feats_path).write_text(
                json.dumps(
                    {"model": model_name, "n_selected": n_selected,
                     "features": selected_feats},
                    indent=2, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            mlflow.log_artifact(feats_path, artifact_path="feature_selection")

        # 모델 아티팩트 (fitted pipeline 전체 저장)
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    logger.info(
        "[%s] run=%s  bal_acc=%.4f  recall=%.4f  f1=%.4f  n_feats=%d",
        model_name, run_id[:8],
        metrics["balanced_accuracy"], metrics["recall"],
        metrics["f1"], n_selected,
    )
    metrics["selected_features"] = selected_feats
    metrics["run_id"]            = run_id
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# 5-Fold 교차 검증
# ──────────────────────────────────────────────────────────────────────────────
def run_cross_validation(
    model_name: str,
    pipeline: Pipeline,
    X_train, y_train,
) -> dict:
    """
    StratifiedKFold CV 를 수행하고 집계 지표를 MLflow 에 기록한다.

    cross_validate() 가 각 fold 에서 pipeline.fit() 을 독립적으로 수행하므로
    fold 간 데이터 누수가 없다.
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring={
            "balanced_accuracy": "balanced_accuracy",
            "precision":         "precision",
            "recall":            "recall",
            "f1":                "f1",
        },
        return_train_score=False,
        n_jobs=-1,
    )

    summary: dict = {}
    with mlflow.start_run(run_name=f"{model_name}_CV"):
        mlflow.set_tag("model_family", model_name)
        mlflow.set_tag("run_type",     "cross_validation")
        mlflow.log_param("cv_folds",   CV_FOLDS)
        mlflow.log_param("model_name", model_name)

        for key, scores in cv_results.items():
            if not key.startswith("test_"):
                continue
            metric_name = key.replace("test_", "")
            mean_v, std_v = float(scores.mean()), float(scores.std())
            summary[f"cv_{metric_name}_mean"] = mean_v
            summary[f"cv_{metric_name}_std"]  = std_v
            mlflow.log_metric(f"cv_{metric_name}_mean", mean_v)
            mlflow.log_metric(f"cv_{metric_name}_std",  std_v)

    logger.info(
        "[CV %s]  bal_acc=%.4f±%.4f  recall=%.4f±%.4f",
        model_name,
        summary.get("cv_balanced_accuracy_mean", 0),
        summary.get("cv_balanced_accuracy_std",  0),
        summary.get("cv_recall_mean", 0),
        summary.get("cv_recall_std",  0),
    )
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# 하이퍼파라미터 튜닝 (Random Forest)
# ──────────────────────────────────────────────────────────────────────────────
def tune_random_forest(X_train, y_train) -> tuple[Pipeline, dict]:
    """
    RandomizedSearchCV 로 Random Forest 하이퍼파라미터를 탐색하고
    최적 파이프라인과 파라미터를 반환한다.

    refit=True: 탐색 종료 후 전체 X_train 으로 최적 파라미터를 재학습.
    scoring=balanced_accuracy: FN/FP 비대칭 비용을 반영한 일관된 기준.
    """
    base_pipeline = build_full_pipeline(
        RandomForestClassifier(
            random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1,
        )
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=RF_PARAM_DIST,
        n_iter=TUNING_N_ITER,
        cv=cv,
        scoring=PRIMARY_SCORE,
        random_state=RANDOM_SEED,
        refit=True,
        n_jobs=-1,
        verbose=0,
    )

    logger.info("[Tuning] RandomForest — RandomizedSearchCV(n_iter=%d, cv=%d) 시작 ...",
                TUNING_N_ITER, CV_FOLDS)
    search.fit(X_train, y_train)

    best_params    = dict(search.best_params_)
    best_cv_score  = float(search.best_score_)
    logger.info("[Tuning] 최적 CV balanced_accuracy: %.4f", best_cv_score)
    logger.info("[Tuning] 최적 파라미터: %s", best_params)

    with mlflow.start_run(run_name="RandomForest_Tuned"):
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("run_type",     "hyperparameter_tuning")
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter",        TUNING_N_ITER)
        mlflow.log_param("scoring",       PRIMARY_SCORE)
        mlflow.log_params(
            {k.replace("classifier__", ""): str(v) for k, v in best_params.items()}
        )
        mlflow.log_metric("best_cv_balanced_accuracy", best_cv_score)
        mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")

    return search.best_estimator_, best_params


# ──────────────────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────────────────
def main() -> None:
    logger.info("=" * 62)
    logger.info("CardioCare — 모델 학습 파이프라인 시작")
    logger.info("=" * 62)

    # ── 1. 데이터 로드 ────────────────────────────────────────────────────────
    if DATA_PATH.exists():
        logger.info("로컬 데이터 로드: %s", DATA_PATH)
        df = pd.read_csv(DATA_PATH)
    else:
        logger.info("UCI 아카이브에서 다운로드 중 ...")
        df = load_raw_data()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        logger.info("저장 완료: %s", DATA_PATH)

    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COL].copy()
    logger.info("데이터 shape: X=%s, 양성 비율=%.2f%%", X.shape, y.mean() * 100)

    # ── 2. 분할 ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    logger.info(
        "Train: %s (양성 %.2f%%)  |  Test: %s (양성 %.2f%%)",
        X_train.shape, y_train.mean() * 100,
        X_test.shape,  y_test.mean() * 100,
    )

    # ── 3. MLflow 환경 설정 ────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info("MLflow URI: %s  |  Experiment: %s",
                MLFLOW_TRACKING_URI, EXPERIMENT_NAME)

    # ── 4. 특성 선택 사전 보고 ────────────────────────────────────────────────
    # 실제 학습과 독립적으로, 선택되는 특성 집합을 미리 확인·보고하기 위한 분석.
    # 각 모델 파이프라인은 내부에서 독립적으로 selector 를 fit 하므로 누수 없음.
    _analysis_pipe = build_full_pipeline(
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    )
    _analysis_pipe.fit(X_train, y_train)
    preview_feats = get_selected_feature_names(_analysis_pipe)

    print("\n" + "=" * 62)
    print(f"[특성 선택] SelectFromModel(RF, threshold=median)")
    print(f"선택된 특성 수: {len(preview_feats)}")
    for f in preview_feats:
        print(f"  • {f}")
    print("=" * 62)

    # ── 5. 전체 모델 학습 & MLflow 기록 ──────────────────────────────────────
    all_results: dict = {}

    print(f"\n{'Model':<24} {'BalAcc':>7} {'Prec':>7} {'Recall':>7}"
          f" {'F1':>7} {'AUC':>7}")
    print("-" * 62)

    for model_name, model in CANDIDATE_MODELS.items():
        pipeline = build_full_pipeline(model)
        results  = mlflow_run_model(
            model_name=model_name,
            pipeline=pipeline,
            X_train=X_train, y_train=y_train,
            X_test=X_test,   y_test=y_test,
        )
        all_results[model_name] = results
        auc_str = f"{results.get('roc_auc', float('nan')):.4f}"
        print(
            f"{model_name:<24}"
            f" {results['balanced_accuracy']:>7.4f}"
            f" {results['precision']:>7.4f}"
            f" {results['recall']:>7.4f}"
            f" {results['f1']:>7.4f}"
            f" {auc_str:>7}"
        )

    print("-" * 62)

    # ── 6. 5-Fold 교차 검증 ───────────────────────────────────────────────────
    print("\n[5-Fold Stratified CV]")
    cv_summary: dict = {}
    for model_name, model in CANDIDATE_MODELS.items():
        pipeline = build_full_pipeline(model)   # 반드시 새 인스턴스
        cv_res   = run_cross_validation(model_name, pipeline, X_train, y_train)
        cv_summary[model_name] = cv_res
        print(
            f"  {model_name:<24}"
            f"  BalAcc={cv_res.get('cv_balanced_accuracy_mean', 0):.4f}"
            f" (±{cv_res.get('cv_balanced_accuracy_std', 0):.4f})"
            f"  Recall={cv_res.get('cv_recall_mean', 0):.4f}"
        )

    # ── 7. 하이퍼파라미터 튜닝 (Random Forest) ────────────────────────────────
    print("\n[하이퍼파라미터 튜닝 — Random Forest (RandomizedSearchCV)]")
    best_rf_pipeline, best_rf_params = tune_random_forest(X_train, y_train)

    # 튜닝된 RF 의 테스트셋 최종 평가
    y_pred_tuned = best_rf_pipeline.predict(X_test)
    y_prob_tuned = best_rf_pipeline.predict_proba(X_test)[:, 1]
    tuned_metrics = compute_metrics(y_test, y_pred_tuned, y_prob_tuned)
    all_results["RandomForest_Tuned"] = tuned_metrics

    # 테스트 평가 결과도 MLflow 에 기록
    with mlflow.start_run(run_name="RandomForest_Tuned_TestEval"):
        mlflow.set_tag("model_family", "RandomForest")
        mlflow.set_tag("run_type",     "tuned_test_evaluation")
        mlflow.log_params(
            {k.replace("classifier__", ""): str(v) for k, v in best_rf_params.items()}
        )
        mlflow.log_metrics(
            {k: v for k, v in tuned_metrics.items() if isinstance(v, (int, float))}
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = _save_cm_figure(
                y_test, y_pred_tuned, "RandomForest_Tuned", tmpdir
            )
            mlflow.log_artifact(cm_path, artifact_path="confusion_matrices")
        mlflow.sklearn.log_model(best_rf_pipeline, artifact_path="model")

    print(f"\n  최적 파라미터: {best_rf_params}")
    print(f"  테스트  balanced_accuracy : {tuned_metrics['balanced_accuracy']:.4f}")
    print(f"  테스트  recall             : {tuned_metrics['recall']:.4f}")
    print(f"  테스트  f1                 : {tuned_metrics['f1']:.4f}")
    print(f"  테스트  roc_auc            : {tuned_metrics.get('roc_auc', float('nan')):.4f}")

    # ── 8. 최종 모델 선택 ─────────────────────────────────────────────────────
    # 선택 기준:
    #   1순위 — balanced_accuracy (클래스 불균형 보정, 전반적 성능)
    #   2순위 — recall            (FN 최소화 — 심장병 환자를 놓치면 안 됨)
    best_name = max(
        all_results,
        key=lambda k: (
            all_results[k]["balanced_accuracy"],
            all_results[k]["recall"],
        ),
    )
    best_m = all_results[best_name]
    logger.info(
        "최종 선택: %s  BalAcc=%.4f  Recall=%.4f  F1=%.4f",
        best_name, best_m["balanced_accuracy"], best_m["recall"], best_m["f1"],
    )

    # 최종 모델 로컬 저장
    if best_name == "RandomForest_Tuned":
        final_pipeline = best_rf_pipeline
    else:
        final_pipeline = build_full_pipeline(CANDIDATE_MODELS[best_name])
        final_pipeline.fit(X_train, y_train)

    joblib.dump(final_pipeline, BEST_MODEL_PATH)
    logger.info("모델 저장: %s", BEST_MODEL_PATH)

    # ── 9. 최종 보고서 출력 ───────────────────────────────────────────────────
    sep = "=" * 62
    print(f"\n{sep}")
    print("최종 모델 선택 결과")
    print(sep)
    print(f"  선택 모델         : {best_name}")
    print(f"  Balanced Accuracy : {best_m['balanced_accuracy']:.4f}")
    print(f"  Recall (Sensitivity): {best_m['recall']:.4f}")
    print(f"  Precision          : {best_m['precision']:.4f}")
    print(f"  F1-Score           : {best_m['f1']:.4f}")
    print(f"  ROC-AUC            : {best_m.get('roc_auc', float('nan')):.4f}")
    cm_vals = (best_m["cm_tp"], best_m["cm_fp"], best_m["cm_fn"], best_m["cm_tn"])
    print(f"  Confusion Matrix   : TP={cm_vals[0]} FP={cm_vals[1]}"
          f" FN={cm_vals[2]} TN={cm_vals[3]}")
    print()
    print("임상적 선택 근거:")
    print(
        "  심장병 예측 시스템에서 가장 위험한 오류는 False Negative —\n"
        "  실제 심장병 환자를 '정상'으로 잘못 분류하는 경우이다.\n"
        "  FN 은 치료 지연·심각한 합병증으로 이어질 수 있어 FP 보다\n"
        "  임상적 비용이 훨씬 크다. 따라서 Recall 을 보조 기준으로,\n"
        "  Balanced Accuracy 를 주 기준으로 삼아 모델을 선택했다.\n"
        "  단, CardioCare 는 심장 전문의의 의사결정을 보조하는 도구이며,\n"
        "  이 시스템의 출력만으로 임상 결정을 내려서는 절대 안 된다."
    )
    print(sep)
    print(f"\nMLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
    print(f"저장된 모델: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
