FROM python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    DO_NOT_TRACK=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY dashboard/app.py ./dashboard/app.py
RUN python -m pip install --no-deps . \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --create-home app \
    && mkdir /runtime \
    && chown app:app /runtime

USER 10001:10001
EXPOSE 8000 5000 8501
ENTRYPOINT ["steel-energy"]
CMD ["serve", "--host", "0.0.0.0", "--workspace", "/runtime/mlflow", "--data", "/runtime/data/processed"]
