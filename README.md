# Occupancy prediction models

This folder contains a cleaned, GitHub-ready version of the occupancy prediction code.

The public code starts from an already prepared 81-column `pandas.DataFrame`. The DataFrame must use two-level columns in the format `(room_name, sensor_name)`. Database access, raw-data export, missing-value handling, and normalization are intentionally excluded.

## Files

- `occupancy_models.py`: shared utilities, room-graph construction, position coding, dataset builders, model definitions, training, evaluation, and plotting.
- `demo_train_eight_models.ipynb`: a runnable demo that trains and compares eight model variants.
- `requirements.txt`: minimal Python dependencies.

## Model variants

1. CNN
2. CNN + position coding
3. LSTM
4. LSTM + position coding
5. GNN + CNN
6. GNN + CNN + position coding
7. GNN + LSTM
8. GNN + LSTM + position coding

## Expected input

The real input should look like this:

```python
import pandas as pd
import occupancy_models as om

# Replace this with your own cleaned DataFrame.
df = pd.read_parquet("your_clean_81_column_dataframe.parquet")

om.validate_input_dataframe(
    df,
    expected_columns=om.DEFAULT_COLUMNS,
    target_rooms=om.TARGET_ROOMS,
)
```

The demo notebook uses `om.make_synthetic_dataframe(...)` only as a public smoke test because the real data cannot be shared.

## Full experiment settings

The notebook uses small settings for a fast demo. For the original paper-scale setting, use:

```python
INPUT_LEN = 72
PRED_LEN = 36
NUM_WINDOWS = 5
START_WINDOW = 2
EPOCHS = 100
PATIENCE = 10
MIN_EPOCHS = 15
BATCH_SIZE = 16
GNN_HIDDEN = 64
LSTM_HIDDEN = 128
```

## Notes

The GNN implementation uses a small dense graph-attention layer implemented directly in PyTorch. This avoids requiring `torch_geometric` in the public demo while preserving the same room-graph and temporal graph modeling structure.
