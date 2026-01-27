# EEGNetReplication data-science pipeline

Replication of results of the original EEGNet paper. We are focused on the SMR test replication specifically

In this project following steps were done for the replication process:
- fetches data: BCI Competition IV; Dataset 2a,
- pre-process fetched data,
- trains a classifier (CNN Model) to predict SMR actions,
- logs everything to `app.log`.

## Quickstart

```bash
python -m venv .eegnetenv
# Windows: .eegnetenv\Scripts\activate
# macOS/Linux: source .eegnetenv/bin/activate

pip install -e ".[ds,test,lint]"
```

1) Fetch data (cached into `data/raw/`) from kaggle

```bash
python -m eegnet_repl.fetch --src kaggle
```

Alternative:
```bash
python -m eegnet_repl.fetch --src moabb
```


2) Preprocess data (cached into `data/processed/`)

```bash
python -m eegnet_repl.dataset --src kaggle
```

3) Train model & Report generation

```bash
python -m eegnet_repl.train --trainingType Within-Subject --generateReport True
```

4) Run UI

```bash
python -m eegnet_repl.ui
```


## Unit tests

1) Unit test for functions in dataset.py

```bash
python -m pytest tests/test_dataset.py -v
```
2) Unit test for functions in model.py

```bash
python -m pytest tests/test_model.py -v
```


