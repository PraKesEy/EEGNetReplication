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


2) Build dataset

```bash
python -m eegnet_repl.dataset --src kaggle
```

3) Train model

```bash
python -m eegnet_repl.train --test-size 0.2 --seed 42
```

4) Run UI

```bash
python -m eegnet_repl.ui
```

## What students should implement (good Git issues)

- Add 2 more engineered features (e.g., log-transform, earth-only close approaches)
- Add a second model (e.g., RandomForest) and compare results
- Add a saved confusion-matrix figure under `reports/`
- Add one more test (e.g., “dataset has no negative diameters”, “model predicts probabilities in [0,1]”)

## Notes

- `DEMO_KEY` is fine for small experiments but has low rate limits; students can generate their own API keys on NASA Open APIs.
