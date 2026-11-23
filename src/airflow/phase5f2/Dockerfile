# Separate disposable Airflow orchestration image; never replace historical stack.
FROM apache/airflow:3.3.1

# uv downloads an independent CPython 3.13.7 interpreter instead of changing
# Airflow's own Python 3.13.14 interpreter or its installed dependencies.
USER root
RUN mkdir -p /opt/lgmma-verified-python && chown airflow:0 /opt/lgmma-verified-python
USER airflow

RUN python -m pip install --no-cache-dir uv && \
    uv python install 3.13.7 && \
    uv venv --python 3.13.7 /opt/lgmma-verified-python && \
    uv pip install --python /opt/lgmma-verified-python/bin/python --no-cache \
      numpy==2.3.2 \
      scikit-learn==1.8.0 \
      tensorflow==2.20.0 \
      keras==3.11.3 \
      pandas pyarrow joblib arch==8.0.0 statsmodels matplotlib

# Airflow itself continues to run with the original image's Python.
