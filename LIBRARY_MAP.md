# LIBRARY_MAP.md — Librerie target e strategia di inventario

Questo file è il “routing”: per ogni libreria definisce:
- import name
- unità “core” (dove partire per l’inventario)
- note speciali (CLI-first, SDK generato, lazy/eager, ecc.)
- suggerimento split (se enorme)

> Nota: l’inventario finale NON va hardcodato qui. Qui si definisce la strategia.
> L’inventario reale va costruito runtime (vedi INVENTORY_AND_COVERAGE.md).

---

## numpy
- Import: `import numpy as np`
- Core surfaces:
  - `numpy` top-level + submodule: `numpy.linalg`, `numpy.random`, `numpy.fft`
  - Classe: `numpy.ndarray` (metodi pubblici)
  - Random: `numpy.random.Generator` (metodi)
- Split: probabile (ndarray + top-level + linalg + random)

## pandas
- Import: `import pandas as pd`
- Core surfaces:
  - `pandas` top-level
  - classi: `DataFrame`, `Series`, `Index`, `MultiIndex`, `Categorical`
  - groupby/resample/rolling accessors e metodi chiave
  - IO: `read_*`, `to_*`
- Split: quasi certo (DataFrame/Series enorme)

## polars
- Import: `import polars as pl`
- Core surfaces:
  - top-level `polars`
  - classi: `DataFrame`, `LazyFrame`, `Series`, `Expr`
  - namespace (es. `pl.col`, `pl.all`, funzioni expression)
- Split: moderato (Expr è vasto)

## pyarrow
- Import: `import pyarrow as pa`
- Core surfaces:
  - top-level + `pyarrow.compute`
  - classi: `Array`, `ChunkedArray`, `Table`, `RecordBatch`, `Schema`
  - dataset/parquet/csv dove applicabile
- Split: possibile (compute è ampio)

## scipy
- Import: `import scipy`
- Core surfaces:
  - subpackage per area: `scipy.optimize`, `scipy.stats`, `scipy.sparse`, `scipy.integrate`, `scipy.signal`
- Split: sì (per subpackage)

## scikit-learn
- Import: `import sklearn`
- Core surfaces:
  - “estimator API”: `fit`, `predict`, `transform`, `score`
  - moduli: `model_selection`, `metrics`, `preprocessing`, `pipeline`, `compose`
- Strategia: per estimator generati in massa, documentare:
  - base classes + pattern + i metodi standard
  - e inventario degli estimator principali (criterio esplicito)
- Split: sì

## xgboost
- Import: `import xgboost as xgb`
- Core surfaces:
  - `xgboost.train`, `xgboost.cv`
  - classi: `Booster`, `DMatrix`
  - sklearn wrapper: `XGBClassifier`, `XGBRegressor`
- Split: no/forse

## lightgbm
- Import: `import lightgbm as lgb`
- Core surfaces:
  - `train`, `cv`
  - classi: `Booster`, `Dataset`
  - sklearn wrapper: `LGBMClassifier`, `LGBMRegressor`
- Split: no/forse

## catboost
- Import: `from catboost import CatBoostClassifier, CatBoostRegressor, Pool`
- Core surfaces:
  - classi: `CatBoost*`, `Pool`
  - metodi: fit/predict, feature_importances, evals
- Split: no/forse

## tensorflow
- Import: `import tensorflow as tf`
- Core surfaces:
  - `tf` top-level + `tf.data`, `tf.keras` (se non separato)
  - per API enorme: definire “surface contract”:
    - tensori, ops principali, data pipeline, training loop, saving/loading
- Split: sì (quasi certo)

## keras
- Import: `import keras` (o `from tensorflow import keras` a seconda setup)
- Core surfaces:
  - `Model`, `Layer`, `optimizers`, `losses`, `metrics`, `callbacks`
- Split: sì

## torch
- Import: `import torch`
- Core surfaces:
  - `torch` top-level ops
  - `torch.Tensor` metodi pubblici
  - `torch.nn`, `torch.optim`, `torch.utils.data`
- Split: sì

## pytorch-lightning
- Import: `import lightning as L` o `import pytorch_lightning as pl`
- Core surfaces:
  - `LightningModule`, `Trainer`, `Callback`, `LightningDataModule`
- Split: no/forse

## transformers
- Import: `import transformers`
- Core surfaces:
  - `pipeline`
  - `AutoTokenizer`, `AutoModel*`
  - `Trainer`, `TrainingArguments`
- Strategia: API enorme → surface contract per “user-facing core”
- Split: sì

## opencv-python
- Import: `import cv2`
- Core surfaces:
  - funzioni top-level `cv2.*`
  - classi principali (VideoCapture, etc.) se esposte
- Split: sì (per aree: imgproc, video, features, etc.)

## statsmodels
- Import: `import statsmodels.api as sm`
- Core surfaces:
  - `sm` API + submodule principali (regression, tsa, stats)
- Split: sì

## prophet
- Import: `from prophet import Prophet`
- Core surfaces:
  - `Prophet` metodi (fit, predict, plot…)
- Split: no

## darts
- Import: `from darts import TimeSeries`
- Core surfaces:
  - `TimeSeries` + modelli forecasting principali (criterio esplicito)
- Split: sì (per modelli)

## pyspark
- Import: `from pyspark.sql import SparkSession`
- Core surfaces:
  - `SparkSession`, `DataFrame`, `Column`, `functions` (F), `types`
- Split: sì (DataFrame/Column enormi)

## delta-spark
- Import: `from delta.tables import DeltaTable`
- Core surfaces:
  - `DeltaTable` + builder/merge API
- Split: no/forse

## dask
- Import: `import dask`, `import dask.dataframe as dd`
- Core surfaces:
  - array/dataframe + compute/scheduler
- Split: sì

## fastapi
- Import: `from fastapi import FastAPI`
- Core surfaces:
  - `FastAPI` + routing/deps + request/response models
- Split: no/forse

## uvicorn
- Import: CLI-first + python entrypoint
- Surface: comandi/opzioni + `uvicorn.run`
- Tipo unità: cli-command + config-option + function
- Split: no

## sqlalchemy
- Import: `import sqlalchemy as sa`
- Core surfaces:
  - `create_engine`, `Engine`, `Session` (orm), `select`, `Table`, `Column`
- Split: sì (core vs orm)

## pyodbc
- Import: `import pyodbc`
- Core surfaces:
  - connect + Connection/Cursor metodi
- Split: no

## requests
- Import: `import requests`
- Core surfaces:
  - `get/post/...`, `Session`, `Response`
- Split: no

## httpx
- Import: `import httpx`
- Core surfaces:
  - `get/post/...`, `Client`, `AsyncClient`, `Response`
- Split: no

## pydantic
- Import: `from pydantic import BaseModel`
- Core surfaces:
  - `BaseModel` + validation + settings (v2 differisce: gestire via introspezione)
- Split: sì

## mlflow
- Import: `import mlflow`
- Core surfaces:
  - tracking (start_run, log_*), models, pyfunc
- Split: sì

## docker (Python SDK + CLI notes)
- Import: `import docker`
- Core surfaces:
  - `docker.from_env`, `DockerClient` e sub-client (containers/images)
- Tipo unità: methods + (appendix CLI essentials)
- Split: no/forse

## kubernetes (Python client)
- Import: `from kubernetes import client, config`
- Core surfaces:
  - `config.load_*`
  - API clients principali (CoreV1Api, AppsV1Api, etc.) e i loro metodi chiamabili
- SDK enorme: surface contract per “client methods” (evitare DTO generati in massa salvo criterio)
- Split: sì

## great_expectations
- Import: `import great_expectations as gx` (varia)
- Core surfaces:
  - context/suite/validator + expectations
- Strategia: expectations sono molte → criterio esplicito (documenta pattern + elenco expectations)
- Split: sì

## evidently
- Import: `import evidently`
- Core surfaces:
  - report/metrics/pipeline
- Split: no/forse

## azure-storage-blob
- Import: `from azure.storage.blob import BlobServiceClient`
- Core surfaces:
  - client objects + metodi principali (container/blob)
- Split: no

## azure-identity
- Import: `from azure.identity import DefaultAzureCredential`
- Core surfaces:
  - credential classes + get_token
- Split: no

## azure-keyvault-secrets
- Import: `from azure.keyvault.secrets import SecretClient`
- Core surfaces:
  - `SecretClient` methods
- Split: no

## azure-ai-ml
- Import: `from azure.ai.ml import MLClient`
- Core surfaces:
  - MLClient + operations (jobs, models, endpoints) — surface contract
- Split: sì

## azure-servicebus
- Import: `from azure.servicebus import ServiceBusClient`
- Core surfaces:
  - sender/receiver + message types
- Split: no

## azure-eventhub
- Import: `from azure.eventhub import EventHubProducerClient`
- Core surfaces:
  - producer/consumer + event data
- Split: no

## chromadb
- Import: `import chromadb`
- Core surfaces:
  - client + collection operations
- Split: no

## qdrant-client
- Import: `from qdrant_client import QdrantClient`
- Core surfaces:
  - client methods (collections, points, search) — surface contract
- Split: no

## faiss
- Import: `import faiss`
- Core surfaces:
  - index classes + add/search/train
- Split: sì (indici diversi)

## pytest (tool + API)
- Import: `import pytest`
- Surface: fixture, mark, raises, parametrize + CLI appendix
- Tipo unità: python API + config-option (pyproject/ini)
- Split: no/forse

## ruff (CLI-first)
- Surface: comandi, regole, config (pyproject)
- Tipo unità: cli-command + config-option
- Split: no

## black (CLI-first)
- Surface: comandi/opzioni + config
- Tipo unità: cli-command + config-option
- Split: no

## mypy (CLI-first)
- Surface: comandi/opzioni + config + typing patterns
- Tipo unità: cli-command + config-option
- Split: no
