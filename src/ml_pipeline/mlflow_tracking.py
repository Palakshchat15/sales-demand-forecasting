"""
Optional MLflow experiment tracking for the training scripts (logging only).

A training script calls log_run() once, AFTER it has written all of its outputs, so tracking
never changes the model, seeds, data or outputs. Every step is guarded: if
MLFLOW_TRACKING_URI is unset, the mlflow package is missing, the server does not answer, or
any MLflow call fails, a WARNING is printed and the script carries on (the Airflow task still
succeeds). The same file is used in the Flight, Fraud and Sales projects.

Tags on every run: airflow_dag_id / airflow_run_id / airflow_task_id / airflow_try_number
(Airflow exports AIRFLOW_CTX_* into every task's environment; "manual" when run by hand) plus
the caller's tags (data_version, ...).
"""
import hashlib
import os
import re
import urllib.request

# Fail fast instead of MLflow's default of 7 retries with exponential backoff (minutes).
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
# The Airflow image has no git checkout of the project: stop GitPython warning about it.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

HEALTH_TIMEOUT_S = 5
PARAM_MAX_CHARS = 500
_KEY_BAD_CHARS = re.compile(r"[^0-9A-Za-z_\-. /]")  # MLflow key alphabet; anything else -> "_"
_AIRFLOW_CTX = {
    "airflow_dag_id": "AIRFLOW_CTX_DAG_ID",
    "airflow_run_id": "AIRFLOW_CTX_DAG_RUN_ID",
    "airflow_task_id": "AIRFLOW_CTX_TASK_ID",
    "airflow_try_number": "AIRFLOW_CTX_TRY_NUMBER",
}


def warn(msg):
    print(f"WARNING [mlflow]: {msg}; continuing without it (training outputs are unaffected).", flush=True)


def safe_key(key):
    return _KEY_BAD_CHARS.sub("_", str(key))[:250]


def tracking_uri():
    """The tracking URI if it is set and (for http/https) the server answers /health, else None."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "").strip()
    if not uri:
        warn("MLFLOW_TRACKING_URI is not set")
        return None
    if uri.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=HEALTH_TIMEOUT_S) as resp:
                resp.read()
        except Exception as e:  # noqa: BLE001 - any failure means "no tracking"
            warn(f"tracking server {uri} is unreachable ({type(e).__name__}: {e})")
            return None
    return uri


def airflow_tags():
    tags = {tag: os.environ[env] for tag, env in _AIRFLOW_CTX.items() if os.environ.get(env)}
    tags.setdefault("airflow_run_id", "manual")
    return tags


def file_sha256(path, chunk=1 << 20):
    """sha256 of a file (streamed, constant memory), or None if it cannot be read."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def flatten_metrics(obj, prefix=""):
    """Every numeric leaf of a nested dict/list as {"a.b.0.c": float}. Lists are indexed (a
    confusion matrix becomes cm.0.0 .. cm.1.1); strings, booleans and None are skipped."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten_metrics(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out.update(flatten_metrics(v, f"{prefix}.{i}" if prefix else str(i)))
    elif isinstance(obj, bool) or obj is None or isinstance(obj, str):
        pass
    else:
        try:
            out[safe_key(prefix)] = float(obj)
        except (TypeError, ValueError):
            pass
    return out


def log_run(experiment, run_name, params=None, metrics=None, artifacts=(), tags=None, log_model=None):
    """Logs one run. `artifacts`: file or directory paths (missing ones are skipped).
    `log_model`: optional callable(mlflow) that logs (and registers) the model inside the run.
    Returns the MLflow run id, or None if tracking was skipped or failed."""
    uri = tracking_uri()
    if uri is None:
        return None
    try:
        import mlflow
    except Exception as e:  # noqa: BLE001
        warn(f"the mlflow package cannot be imported ({type(e).__name__}: {e})")
        return None
    try:
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        all_tags = {**airflow_tags(), **{k: str(v) for k, v in (tags or {}).items() if v is not None}}
        with mlflow.start_run(run_name=run_name, tags=all_tags) as run:
            if params:
                mlflow.log_params({safe_key(k): str(v)[:PARAM_MAX_CHARS] for k, v in params.items()})
            if metrics:
                mlflow.log_metrics(metrics)  # the client sends these in batches of <= 1000
            for path in artifacts:
                path = str(path)
                if os.path.isdir(path):
                    mlflow.log_artifacts(path, artifact_path=os.path.basename(path.rstrip("/\\")))
                elif os.path.isfile(path):
                    mlflow.log_artifact(path)
                else:
                    print(f"[mlflow] artifact not found, skipped: {path}", flush=True)
            if log_model is not None:
                try:
                    log_model(mlflow)
                except Exception as e:  # noqa: BLE001 - keep the run, params and metrics
                    warn(f"model logging/registration failed ({type(e).__name__}: {e})")
        print(f"[mlflow] logged run {run.info.run_id} ('{run_name}') to experiment '{experiment}' at {uri}",
              flush=True)
        return run.info.run_id
    except Exception as e:  # noqa: BLE001
        warn(f"logging failed ({type(e).__name__}: {e})")
        return None
