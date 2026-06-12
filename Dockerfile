# ──────────────────────────────────────────────────────────────────────────────
# CardioCare — Dockerfile
#
# 빌드:  docker build -t cardiocare:1.0 .
# 실행:  docker run --rm cardiocare:1.0
# 커스텀 입력:
#   docker run --rm -v "$(pwd)/data:/app/data" cardiocare:1.0 \
#     --input data/my_batch.csv --output data/predictions.csv
#
# [사전 조건]
#   빌드 전 반드시 로컬에서 python src/train.py 를 실행하여
#   models/best_model.pkl 이 존재해야 합니다.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.10-slim

LABEL maintainer="CardioCare"
LABEL version="1.0"
LABEL description="Heart Disease Prediction Inference Service"

WORKDIR /app

# ── 1. 의존성 설치 (소스 코드 변경과 레이어 분리 → 캐시 효율) ────────────────
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# ── 2. 소스 코드 복사 ─────────────────────────────────────────────────────────
COPY src/ ./src/

# ── 3. 사전 학습된 모델 복사 (python src/train.py 실행 후 생성됨) ──────────────
COPY models/ ./models/

# ── 4. 샘플 입력 데이터 복사 ──────────────────────────────────────────────────
COPY data/sample_batch.csv ./data/sample_batch.csv

# ── 5. 비루트 사용자 생성 + 로그 디렉터리 (소유권까지 한 번에) ─────────────────
RUN useradd --create-home appuser \
 && mkdir -p logs \
 && chown -R appuser:appuser logs

USER appuser

# ── 6. 추론 엔트리포인트 ──────────────────────────────────────────────────────
# 기본 동작: 샘플 배치 파일로 추론 실행
# 오버라이드 예시: docker run ... --input data/custom.csv --output /tmp/out.csv
ENTRYPOINT ["python", "src/inference.py"]
CMD ["--input", "data/sample_batch.csv", "--output", "/tmp/predictions.csv"]
