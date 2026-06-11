"""Utilities for occupancy prediction experiments.

The module assumes that the input data already exists as a cleaned 81-column
pandas DataFrame. The expected column format is a two-level MultiIndex:
(room_name, sensor_name). No database access or raw-data preprocessing is
included here.
"""

from __future__ import annotations

import copy
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


TARGET_ROOMS: List[str] = ["Badkamer", "Eetkamer", "Keuken", "Living", "Hal boven"]

ROOM_GRAPH_EDGES: List[Tuple[str, str]] = [
    ("Hal boven", "Badkamer"),
    ("Hal boven", "slaapkamer 1"),
    ("Hal boven", "slaapkamer 2"),
    ("Hal boven", "slaapkamer 3"),
    ("Hal boven", "Hal beneden"),
    ("Hal boven", "Wasmachine"),
    ("Hal beneden", "Living"),
    ("Eetkamer", "Keuken"),
    ("Keuken", "Koelkast"),
    ("Living", "Eetkamer"),
    ("Living", "TV"),
]

DEFAULT_COLUMNS: List[Tuple[str, str]] = [
    ("Badkamer", "Pirstatus"),
    ("Badkamer", "Pirsum"),
    ("Badkamer", "MotorPosition"),
    ("Badkamer", "Temperature"),
    ("Badkamer", "TemperatureSet"),
    ("Badkamer", "Lightlevel"),
    ("Eetkamer", "Pirstatus"),
    ("Eetkamer", "Pirsum"),
    ("Eetkamer", "Lightlevel"),
    ("Hal beneden", "AccMotion"),
    ("Hal beneden", "Digital"),
    ("Hal beneden", "MotorPosition"),
    ("Hal beneden", "Temperature"),
    ("Hal beneden", "TemperatureSet"),
    ("Hal beneden", "X"),
    ("Hal beneden", "Y"),
    ("Hal beneden", "Z"),
    ("Hal boven", "Pirstatus"),
    ("Hal boven", "Pirsum"),
    ("Hal boven", "Lightlevel"),
    ("Keuken", "Pirstatus"),
    ("Keuken", "Pirsum"),
    ("Keuken", "MotorPosition"),
    ("Keuken", "Temperature"),
    ("Keuken", "TemperatureSet"),
    ("Keuken", "Lightlevel"),
    ("Koelkast", "ActivePower"),
    ("Koelkast", "Current"),
    ("Koelkast", "TotalActiveEnergy"),
    ("Koelkast", "Voltage"),
    ("Living", "Pirstatus"),
    ("Living", "Pirsum"),
    ("Living", "Co2"),
    ("Living", "Humidity"),
    ("Living", "Pressure"),
    ("Living", "Temperature"),
    ("Living", "Tvoc"),
    ("Living", "Lightlevel"),
    ("TV", "ActivePower"),
    ("TV", "Current"),
    ("TV", "TotalActiveEnergy"),
    ("TV", "Voltage"),
    ("Wasmachine", "ActivePower"),
    ("Wasmachine", "Current"),
    ("Wasmachine", "TotalActiveEnergy"),
    ("Wasmachine", "Voltage"),
    ("digitale meter", "GasKuub"),
    ("digitale meter", "NegativeActivePower"),
    ("digitale meter", "PhaseACurrent"),
    ("digitale meter", "PhaseAPositiveActivePower"),
    ("digitale meter", "PositiveActivePower"),
    ("slaapkamer 1", "AccMotion"),
    ("slaapkamer 1", "Digital"),
    ("slaapkamer 1", "MotorPosition"),
    ("slaapkamer 1", "Temperature"),
    ("slaapkamer 1", "TemperatureSet"),
    ("slaapkamer 1", "X"),
    ("slaapkamer 1", "Y"),
    ("slaapkamer 1", "Z"),
    ("slaapkamer 2", "AccMotion"),
    ("slaapkamer 2", "Digital"),
    ("slaapkamer 2", "MotorPosition"),
    ("slaapkamer 2", "Temperature"),
    ("slaapkamer 2", "TemperatureSet"),
    ("slaapkamer 2", "X"),
    ("slaapkamer 2", "Y"),
    ("slaapkamer 2", "Z"),
    ("slaapkamer 3", "AccMotion"),
    ("slaapkamer 3", "Digital"),
    ("slaapkamer 3", "MotorPosition"),
    ("slaapkamer 3", "Temperature"),
    ("slaapkamer 3", "TemperatureSet"),
    ("slaapkamer 3", "X"),
    ("slaapkamer 3", "Y"),
    ("slaapkamer 3", "Z"),
    ("watermeter", "Humidity"),
    ("watermeter", "Temperature"),
    ("Time", "hour_sin"),
    ("Time", "hour_cos"),
    ("Time", "dow_sin"),
    ("Time", "dow_cos"),
]


@dataclass
class SplitIndices:
    """Train, validation, and test indices for one expanding-window split."""

    window_id: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


@dataclass
class GraphInfo:
    """Metadata needed by graph-based models."""

    room_names: List[str]
    sensor_keys: List[str]
    adjacency: np.ndarray
    room_to_idx: Dict[str, int]


class ArrayDataset(Dataset):
    """A thin Dataset wrapper for numpy arrays."""

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[idx], self.y[idx]


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return a CUDA device when available, otherwise CPU."""

    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def validate_input_dataframe(
    df: pd.DataFrame,
    expected_columns: Optional[Sequence[Tuple[str, str]]] = None,
    target_rooms: Sequence[str] = TARGET_ROOMS,
) -> None:
    """Validate that the input DataFrame has the required shape and target labels."""

    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("The input DataFrame must use MultiIndex columns: (room_name, sensor_name).")
    if df.columns.nlevels != 2:
        raise ValueError("The input DataFrame must have exactly two column levels.")
    if expected_columns is not None:
        missing = [col for col in expected_columns if col not in df.columns]
        if missing:
            raise ValueError(f"The input DataFrame is missing expected columns: {missing[:10]}")
    missing_targets = [(room, "Pirstatus") for room in target_rooms if (room, "Pirstatus") not in df.columns]
    if missing_targets:
        raise ValueError(f"The input DataFrame is missing target occupancy columns: {missing_targets}")
    if df.shape[1] != 81:
        raise ValueError(f"Expected an 81-column DataFrame, but received {df.shape[1]} columns.")


def get_room_names(df: pd.DataFrame, include_time: bool = False) -> List[str]:
    """Return sorted room names from the first column level."""

    excluded = set() if include_time else {"Time", "Meta"}
    room_names = sorted({str(col[0]) for col in df.columns if str(col[0]) not in excluded})
    return room_names


def make_synthetic_dataframe(
    n_steps: int = 900,
    columns: Sequence[Tuple[str, str]] = DEFAULT_COLUMNS,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a synthetic 81-column DataFrame for a runnable smoke test.

    This function is only for the public demo notebook. Replace its output with
    the real cleaned DataFrame when running the paper experiments.
    """

    rng = np.random.default_rng(seed)
    index = pd.date_range("2025-01-01", periods=n_steps, freq="10min")
    multi_columns = pd.MultiIndex.from_tuples(columns)
    data = np.zeros((n_steps, len(columns)), dtype=np.float32)

    room_phase = {room: rng.uniform(0, 2 * np.pi) for room, _ in columns}
    room_bias = {room: rng.uniform(-0.6, 0.6) for room, _ in columns}

    hours = np.arange(n_steps) / 6.0
    daily_cycle = np.sin(2 * np.pi * hours / 24.0)
    weekly_cycle = np.sin(2 * np.pi * hours / (24.0 * 7.0))

    occupancy_cache: Dict[str, np.ndarray] = {}
    rooms = sorted({room for room, _ in columns if room not in {"Time", "Meta"}})
    for room in rooms:
        signal = 0.7 * np.sin(2 * np.pi * hours / 24.0 + room_phase[room]) + 0.3 * weekly_cycle + room_bias[room]
        probability = 1.0 / (1.0 + np.exp(-signal))
        occupancy_cache[room] = rng.binomial(1, probability * 0.55).astype(np.float32)

    for j, (room, sensor) in enumerate(columns):
        if room == "Time" and sensor == "hour_sin":
            data[:, j] = np.sin(2 * np.pi * (index.hour + index.minute / 60.0) / 24.0)
        elif room == "Time" and sensor == "hour_cos":
            data[:, j] = np.cos(2 * np.pi * (index.hour + index.minute / 60.0) / 24.0)
        elif room == "Time" and sensor == "dow_sin":
            data[:, j] = np.sin(2 * np.pi * index.dayofweek / 7.0)
        elif room == "Time" and sensor == "dow_cos":
            data[:, j] = np.cos(2 * np.pi * index.dayofweek / 7.0)
        elif sensor == "Pirstatus":
            data[:, j] = occupancy_cache.get(room, rng.binomial(1, 0.15, size=n_steps))
        elif sensor == "Pirsum":
            data[:, j] = np.clip(occupancy_cache.get(room, 0) + rng.normal(0.15, 0.15, n_steps), 0, 1)
        else:
            base = 0.35 + 0.15 * daily_cycle + 0.05 * weekly_cycle + rng.normal(0, 0.08, n_steps)
            if room in occupancy_cache:
                base += 0.15 * occupancy_cache[room]
            data[:, j] = np.clip(base, 0, 1)

    return pd.DataFrame(data, index=index, columns=multi_columns)


def build_room_adjacency(
    room_names: Sequence[str],
    room_graph_edges: Sequence[Tuple[str, str]] = ROOM_GRAPH_EDGES,
    add_self_loops: bool = True,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build a dense adjacency matrix from the predefined room graph."""

    room_to_idx = {room: i for i, room in enumerate(room_names)}
    adjacency = np.zeros((len(room_names), len(room_names)), dtype=np.float32)
    for src, dst in room_graph_edges:
        if src in room_to_idx and dst in room_to_idx:
            i, j = room_to_idx[src], room_to_idx[dst]
            adjacency[i, j] = 1.0
            adjacency[j, i] = 1.0
    if add_self_loops:
        np.fill_diagonal(adjacency, 1.0)
    return adjacency, room_to_idx


def compute_distance_features(
    room_names_or_columns: Sequence,
    target_rooms: Sequence[str] = TARGET_ROOMS,
    room_graph_edges: Sequence[Tuple[str, str]] = ROOM_GRAPH_EDGES,
    max_distance: int = 10,
) -> np.ndarray:
    """Compute normalized shortest-path distances to each target room.

    The input can be a sequence of room names or a sequence of MultiIndex column
    tuples. The output shape is (len(input), len(target_rooms)).
    """

    graph: Dict[str, List[str]] = defaultdict(list)
    for src, dst in room_graph_edges:
        graph[src].append(dst)
        graph[dst].append(src)

    def bfs(start_room: str) -> Dict[str, int]:
        visited = set()
        queue = deque([(start_room, 0)])
        distances: Dict[str, int] = {}
        while queue:
            room, distance = queue.popleft()
            if room in visited:
                continue
            visited.add(room)
            distances[room] = distance
            for neighbor in graph[room]:
                if neighbor not in visited:
                    queue.append((neighbor, distance + 1))
        return distances

    distance_maps = {room: bfs(room) for room in target_rooms}
    normalized_distances = []
    for item in room_names_or_columns:
        room = item[0] if isinstance(item, tuple) else item
        normalized_distances.append(
            [distance_maps[target].get(room, max_distance) / max_distance for target in target_rooms]
        )
    return np.asarray(normalized_distances, dtype=np.float32)


def get_target_indices(df: pd.DataFrame, target_rooms: Sequence[str] = TARGET_ROOMS) -> List[int]:
    """Return column indices of target-room Pirstatus labels."""

    column_to_index = {col: idx for idx, col in enumerate(df.columns)}
    return [column_to_index[(room, "Pirstatus")] for room in target_rooms]


def make_future_label(
    raw_values: np.ndarray,
    t_start: int,
    input_len: int,
    pred_len: int,
    target_indices: Sequence[int],
) -> np.ndarray:
    """Return a multi-label target indicating future occupancy in the prediction window."""

    future = raw_values[t_start + input_len : t_start + input_len + pred_len, target_indices]
    return (np.nanmax(future, axis=0) > 0).astype(np.float32)


def pad_to_size(values: np.ndarray, max_size: int = 81) -> np.ndarray:
    """Pad or truncate the last dimension to max_size."""

    output = np.zeros((values.shape[0], max_size), dtype=np.float32)
    n = min(max_size, values.shape[1])
    output[:, :n] = values[:, :n]
    return output


def build_sequence_windows(
    df: pd.DataFrame,
    input_len: int = 72,
    pred_len: int = 36,
    max_size: int = 81,
    target_rooms: Sequence[str] = TARGET_ROOMS,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build dense temporal windows for CNN and LSTM baselines.

    Returns:
        x: shape (num_samples, input_len, max_size)
        y: shape (num_samples, num_target_rooms)
    """

    validate_input_dataframe(df, target_rooms=target_rooms)
    raw_values = df.values.astype(np.float32)
    input_values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)
    target_indices = get_target_indices(df, target_rooms)
    total_len = input_values.shape[0]
    num_samples = total_len - input_len - pred_len + 1
    if num_samples <= 0:
        raise ValueError("The DataFrame is too short for the requested input_len and pred_len.")

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Building sequence windows")
    for t in iterator:
        x_list.append(pad_to_size(input_values[t : t + input_len], max_size=max_size))
        y_list.append(make_future_label(raw_values, t, input_len, pred_len, target_indices))
    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32)


def build_cnn_position_windows(
    df: pd.DataFrame,
    input_len: int = 72,
    pred_len: int = 36,
    max_size: int = 81,
    target_rooms: Sequence[str] = TARGET_ROOMS,
    max_distance: int = 10,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build CNN input with distance channels.

    Returns x with shape (num_samples, input_len, 1 + num_target_rooms, max_size).
    Channel 0 stores the sensor values; the remaining channels store normalized
    graph distances from each column's room to each target room.
    """

    x_sequence, y = build_sequence_windows(df, input_len, pred_len, max_size, target_rooms, show_progress)
    room_columns = list(df.columns[:max_size])
    if len(room_columns) < max_size:
        room_columns = room_columns + [("", "")] * (max_size - len(room_columns))
    distances = compute_distance_features(room_columns, target_rooms, max_distance=max_distance).T
    x_position = np.zeros((x_sequence.shape[0], input_len, 1 + len(target_rooms), max_size), dtype=np.float32)
    x_position[:, :, 0, :] = x_sequence
    x_position[:, :, 1:, :] = distances[None, None, :, :]
    return x_position, y


def build_lstm_position_windows(
    df: pd.DataFrame,
    input_len: int = 72,
    pred_len: int = 36,
    max_size: int = 81,
    target_rooms: Sequence[str] = TARGET_ROOMS,
    max_distance: int = 10,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build LSTM input with flattened distance features.

    Returns x with shape (num_samples, input_len, max_size * (1 + num_target_rooms)).
    """

    x_sequence, y = build_sequence_windows(df, input_len, pred_len, max_size, target_rooms, show_progress)
    room_columns = list(df.columns[:max_size])
    if len(room_columns) < max_size:
        room_columns = room_columns + [("", "")] * (max_size - len(room_columns))
    distance_matrix = compute_distance_features(room_columns, target_rooms, max_distance=max_distance)
    flattened_distances = distance_matrix.reshape(-1).astype(np.float32)
    repeated_distances = np.broadcast_to(
        flattened_distances[None, None, :],
        (x_sequence.shape[0], input_len, flattened_distances.shape[0]),
    )
    x_position = np.concatenate([x_sequence, repeated_distances], axis=-1)
    return x_position.astype(np.float32), y


def build_graph_windows(
    df: pd.DataFrame,
    input_len: int = 72,
    pred_len: int = 36,
    target_rooms: Sequence[str] = TARGET_ROOMS,
    use_position: bool = False,
    max_distance: int = 10,
    room_graph_edges: Sequence[Tuple[str, str]] = ROOM_GRAPH_EDGES,
    show_progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray, GraphInfo]:
    """Build temporal graph windows.

    Returns:
        x: shape (num_samples, input_len, num_rooms, num_node_features)
        y: shape (num_samples, num_target_rooms)
        graph_info: room names, aligned sensor keys, and adjacency matrix
    """

    validate_input_dataframe(df, target_rooms=target_rooms)
    room_names = get_room_names(df, include_time=False)
    adjacency, room_to_idx = build_room_adjacency(room_names, room_graph_edges)

    room_sensor_keys: Dict[str, List[str]] = {}
    for room in room_names:
        room_columns = [sensor for r, sensor in df.columns if r == room]
        room_sensor_keys[room] = list(room_columns)
    sensor_keys = sorted({sensor for sensors in room_sensor_keys.values() for sensor in sensors})

    aligned_features: List[np.ndarray] = []
    distance_features = compute_distance_features(room_names, target_rooms, room_graph_edges, max_distance=max_distance)
    for room_idx, room in enumerate(room_names):
        room_df = pd.DataFrame(index=df.index)
        for sensor in sensor_keys:
            if (room, sensor) in df.columns:
                room_df[sensor] = df[(room, sensor)].astype(np.float32)
            else:
                room_df[sensor] = 0.0
        values = np.nan_to_num(room_df[sensor_keys].values.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if use_position:
            position = np.broadcast_to(distance_features[room_idx][None, :], (values.shape[0], len(target_rooms)))
            values = np.concatenate([values, position.astype(np.float32)], axis=-1)
        aligned_features.append(values)

    time_room_feature = np.stack(aligned_features, axis=1).astype(np.float32)
    raw_values = df.values.astype(np.float32)
    target_indices = get_target_indices(df, target_rooms)
    num_samples = time_room_feature.shape[0] - input_len - pred_len + 1
    if num_samples <= 0:
        raise ValueError("The DataFrame is too short for the requested input_len and pred_len.")

    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    iterator = range(num_samples)
    if show_progress:
        iterator = tqdm(iterator, desc="Building graph windows")
    for t in iterator:
        x_list.append(time_room_feature[t : t + input_len])
        y_list.append(make_future_label(raw_values, t, input_len, pred_len, target_indices))

    graph_info = GraphInfo(room_names=room_names, sensor_keys=sensor_keys, adjacency=adjacency, room_to_idx=room_to_idx)
    return np.stack(x_list).astype(np.float32), np.stack(y_list).astype(np.float32), graph_info


def build_all_model_inputs(
    df: pd.DataFrame,
    input_len: int = 72,
    pred_len: int = 36,
    max_size: int = 81,
    target_rooms: Sequence[str] = TARGET_ROOMS,
    show_progress: bool = True,
) -> Dict[str, Dict[str, object]]:
    """Build inputs for all eight model variants."""

    x_sequence, y = build_sequence_windows(df, input_len, pred_len, max_size, target_rooms, show_progress)
    x_cnn_position, _ = build_cnn_position_windows(df, input_len, pred_len, max_size, target_rooms, show_progress=False)
    x_lstm_position, _ = build_lstm_position_windows(df, input_len, pred_len, max_size, target_rooms, show_progress=False)
    x_graph, _, graph_info = build_graph_windows(df, input_len, pred_len, target_rooms, use_position=False, show_progress=show_progress)
    x_graph_position, _, graph_position_info = build_graph_windows(
        df, input_len, pred_len, target_rooms, use_position=True, show_progress=show_progress
    )

    return {
        "CNN": {"x": x_sequence, "y": y, "graph_info": None},
        "CNN + position coding": {"x": x_cnn_position, "y": y, "graph_info": None},
        "LSTM": {"x": x_sequence, "y": y, "graph_info": None},
        "LSTM + position coding": {"x": x_lstm_position, "y": y, "graph_info": None},
        "GNN + CNN": {"x": x_graph, "y": y, "graph_info": graph_info},
        "GNN + CNN + position coding": {"x": x_graph_position, "y": y, "graph_info": graph_position_info},
        "GNN + LSTM": {"x": x_graph, "y": y, "graph_info": graph_info},
        "GNN + LSTM + position coding": {"x": x_graph_position, "y": y, "graph_info": graph_position_info},
    }


class TemporalCNNPredictor(nn.Module):
    """Temporal CNN baseline for dense sequence inputs."""

    def __init__(self, input_channels: int = 1, seq_len: int = 72, output_dim: int = 5, dropout: float = 0.3):
        super().__init__()
        pooled_t = max(1, seq_len)
        pooled_w = 64
        self.conv2d = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pooled_t, pooled_w)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_t * pooled_w, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        elif x.ndim == 4:
            x = x.permute(0, 2, 1, 3)
        else:
            raise ValueError(f"TemporalCNNPredictor expects a 3D or 4D tensor, received shape {tuple(x.shape)}")
        return self.head(self.conv2d(x))


class LSTMOccupancyPredictor(nn.Module):
    """LSTM baseline for dense sequence inputs."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 5,
        bidirectional: bool = False,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim > 3:
            x = x.flatten(start_dim=2)
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


class DenseGATLayer(nn.Module):
    """A small dense graph-attention layer that avoids external PyG dependency."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 1,
        concat: bool = True,
        dropout: float = 0.3,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_dim, out_dim * heads, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, out_dim))
        self.att_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.negative_slope = negative_slope
        self.reset_parameters()

    @property
    def output_dim(self) -> int:
        return self.out_dim * self.heads if self.concat else self.out_dim

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, _ = x.shape
        h = self.linear(x).view(batch_size, num_nodes, self.heads, self.out_dim)
        src_scores = (h * self.att_src.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)
        dst_scores = (h * self.att_dst.view(1, 1, self.heads, self.out_dim)).sum(dim=-1)
        scores = src_scores[:, :, None, :] + dst_scores[:, None, :, :]
        scores = F.leaky_relu(scores, negative_slope=self.negative_slope)
        mask = adjacency.to(dtype=torch.bool, device=x.device).view(1, num_nodes, num_nodes, 1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attention = torch.softmax(scores, dim=2)
        attention = self.dropout(attention)
        output = torch.einsum("bijh,bjhd->bihd", attention, h)
        if self.concat:
            return output.reshape(batch_size, num_nodes, self.heads * self.out_dim)
        return output.mean(dim=2)


class DenseGATEncoder(nn.Module):
    """Two-layer graph-attention encoder applied independently at each time step."""

    def __init__(self, in_channels: int, hidden_dim: int, adjacency: np.ndarray, dropout: float = 0.3):
        super().__init__()
        self.register_buffer("adjacency", torch.as_tensor(adjacency, dtype=torch.float32))
        self.layer1 = DenseGATLayer(in_channels, hidden_dim, heads=4, concat=True, dropout=dropout)
        self.layer2 = DenseGATLayer(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
        self.out_channels = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_nodes, in_channels = x.shape
        h = x.reshape(batch_size * seq_len, num_nodes, in_channels)
        h = F.elu(self.layer1(h, self.adjacency))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = F.elu(self.layer2(h, self.adjacency))
        h = F.dropout(h, p=self.dropout, training=self.training)
        return h.reshape(batch_size, seq_len, num_nodes, self.out_channels)


class GATTemporalCNNPredictor(nn.Module):
    """Graph-attention encoder followed by a temporal-node CNN head."""

    def __init__(
        self,
        in_channels: int,
        gnn_hidden: int,
        num_nodes: int,
        seq_len: int,
        output_dim: int,
        adjacency: np.ndarray,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = DenseGATEncoder(in_channels, gnn_hidden, adjacency, dropout=dropout)
        pooled_t = max(1, seq_len // 2)
        pooled_n = max(1, num_nodes // 2)
        self.conv2d = nn.Sequential(
            nn.Conv2d(gnn_hidden, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((pooled_t, pooled_n)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * pooled_t * pooled_n, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        x = embeddings.permute(0, 3, 1, 2)
        return self.head(self.conv2d(x))


class GATTemporalLSTMPredictor(nn.Module):
    """Graph-attention encoder followed by a bidirectional LSTM head."""

    def __init__(
        self,
        in_channels: int,
        gnn_hidden: int,
        num_nodes: int,
        seq_len: int,
        output_dim: int,
        adjacency: np.ndarray,
        lstm_hidden: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.encoder = DenseGATEncoder(in_channels, gnn_hidden, adjacency, dropout=dropout)
        self.lstm = nn.LSTM(
            input_size=num_nodes * gnn_hidden,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = self.encoder(x)
        batch_size, seq_len, num_nodes, hidden_dim = embeddings.shape
        x = embeddings.reshape(batch_size, seq_len, num_nodes * hidden_dim)
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])


def create_model_factory(
    model_name: str,
    x_shape: Tuple[int, ...],
    output_dim: int,
    graph_info: Optional[GraphInfo] = None,
    gnn_hidden: int = 64,
    lstm_hidden: int = 128,
) -> Callable[[], nn.Module]:
    """Create a no-argument model factory for one of the eight model names."""

    seq_len = int(x_shape[1])
    if model_name == "CNN":
        return lambda: TemporalCNNPredictor(input_channels=1, seq_len=seq_len, output_dim=output_dim)
    if model_name == "CNN + position coding":
        input_channels = int(x_shape[2])
        return lambda: TemporalCNNPredictor(input_channels=input_channels, seq_len=seq_len, output_dim=output_dim)
    if model_name == "LSTM":
        input_dim = int(x_shape[-1])
        return lambda: LSTMOccupancyPredictor(input_dim=input_dim, hidden_dim=lstm_hidden, output_dim=output_dim)
    if model_name == "LSTM + position coding":
        input_dim = int(x_shape[-1])
        return lambda: LSTMOccupancyPredictor(input_dim=input_dim, hidden_dim=lstm_hidden, output_dim=output_dim)
    if graph_info is None:
        raise ValueError(f"graph_info is required for model: {model_name}")
    in_channels = int(x_shape[-1])
    num_nodes = int(x_shape[2])
    if model_name == "GNN + CNN":
        return lambda: GATTemporalCNNPredictor(
            in_channels, gnn_hidden, num_nodes, seq_len, output_dim, graph_info.adjacency
        )
    if model_name == "GNN + CNN + position coding":
        return lambda: GATTemporalCNNPredictor(
            in_channels, gnn_hidden, num_nodes, seq_len, output_dim, graph_info.adjacency
        )
    if model_name == "GNN + LSTM":
        return lambda: GATTemporalLSTMPredictor(
            in_channels, gnn_hidden, num_nodes, seq_len, output_dim, graph_info.adjacency, lstm_hidden=lstm_hidden
        )
    if model_name == "GNN + LSTM + position coding":
        return lambda: GATTemporalLSTMPredictor(
            in_channels, gnn_hidden, num_nodes, seq_len, output_dim, graph_info.adjacency, lstm_hidden=lstm_hidden
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_expanding_window_splits(
    num_samples: int,
    num_windows: int = 5,
    start_window: int = 2,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
) -> List[SplitIndices]:
    """Create expanding-window splits following the original notebooks."""

    if num_windows < start_window:
        raise ValueError("num_windows must be greater than or equal to start_window.")
    segment_len = num_samples // num_windows
    if segment_len <= 0:
        raise ValueError("Not enough samples for the requested number of windows.")

    splits: List[SplitIndices] = []
    for window_id in range(start_window, num_windows + 1):
        window_end = segment_len * window_id
        train_ratio = 1.0 - val_ratio - test_ratio
        train_end = int(window_end * train_ratio)
        val_end = train_end + int(window_end * val_ratio)
        val_end = min(val_end, window_end)
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(train_end, val_end)
        test_idx = np.arange(val_end, window_end)
        if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
            continue
        splits.append(SplitIndices(window_id=window_id, train=train_idx, val=val_idx, test=test_idx))
    return splits


def safe_macro_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute macro AUC while ignoring labels without both classes."""

    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    if not aucs:
        return float("nan")
    return float(np.mean(aucs))


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    min_epochs: int = 15,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """Train one model with BCEWithLogitsLoss and early stopping on validation AUC."""

    device = device or get_device()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = {"train_loss": [], "val_loss": [], "val_auc": []}
    best_auc = -math.inf
    best_state: Optional[Dict[str, torch.Tensor]] = None
    bad_epochs = 0

    iterator = range(epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Training epochs", leave=False)

    for epoch in iterator:
        model.train()
        total_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += float(loss.item())
        train_loss = total_train_loss / max(1, len(train_loader))

        model.eval()
        total_val_loss = 0.0
        y_true_batches = []
        y_prob_batches = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits = model(x_val)
                loss = criterion(logits, y_val)
                total_val_loss += float(loss.item())
                y_true_batches.append(y_val.detach().cpu().numpy())
                y_prob_batches.append(torch.sigmoid(logits).detach().cpu().numpy())

        val_loss = total_val_loss / max(1, len(val_loader))
        y_true = np.concatenate(y_true_batches, axis=0)
        y_prob = np.concatenate(y_prob_batches, axis=0)
        val_auc = safe_macro_auc(y_true, y_prob)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_auc"].append(val_auc)

        current_auc = -math.inf if np.isnan(val_auc) else val_auc
        if epoch + 1 < min_epochs:
            continue
        if current_auc > best_auc:
            best_auc = current_auc
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_probabilities(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return sigmoid probabilities for a data loader."""

    device = device or get_device()
    model.to(device)
    model.eval()
    predictions = []
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            predictions.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_rooms: Sequence[str] = TARGET_ROOMS,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Compute binary classification metrics per target room."""

    y_binary = (y_prob >= threshold).astype(np.int32)
    rows = []
    for i, room in enumerate(target_rooms):
        y_t = y_true[:, i]
        y_b = y_binary[:, i]
        y_p = y_prob[:, i]
        if len(np.unique(y_t)) < 2:
            auc_value = np.nan
        else:
            auc_value = roc_auc_score(y_t, y_p)
        rows.append(
            {
                "room": room,
                "accuracy": accuracy_score(y_t, y_b),
                "recall": recall_score(y_t, y_b, zero_division=0),
                "precision": precision_score(y_t, y_b, zero_division=0),
                "f1": f1_score(y_t, y_b, zero_division=0),
                "auc": auc_value,
            }
        )
    return pd.DataFrame(rows)


def run_expanding_window_experiment(
    x: np.ndarray,
    y: np.ndarray,
    model_factory: Callable[[], nn.Module],
    target_rooms: Sequence[str] = TARGET_ROOMS,
    num_windows: int = 5,
    start_window: int = 2,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    batch_size: int = 16,
    epochs: int = 100,
    patience: int = 10,
    min_epochs: int = 15,
    lr: float = 5e-4,
    weight_decay: float = 1e-4,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run one model across expanding-window splits."""

    device = device or get_device()
    splits = make_expanding_window_splits(len(x), num_windows, start_window, val_ratio, test_ratio)
    per_window_metrics = []
    histories = []

    for split in splits:
        train_loader = DataLoader(ArrayDataset(x[split.train], y[split.train]), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(ArrayDataset(x[split.val], y[split.val]), batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(ArrayDataset(x[split.test], y[split.test]), batch_size=batch_size, shuffle=False)

        model = model_factory()
        history = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            min_epochs=min_epochs,
            device=device,
            verbose=verbose,
        )
        y_prob = predict_probabilities(model, test_loader, device=device)
        metrics = evaluate_predictions(y[split.test], y_prob, target_rooms=target_rooms)
        metrics.insert(0, "window", split.window_id)
        per_window_metrics.append(metrics)
        histories.append(history)

    if per_window_metrics:
        per_window_df = pd.concat(per_window_metrics, ignore_index=True)
        average_metrics = per_window_df.groupby("room", as_index=False)[["accuracy", "recall", "precision", "f1", "auc"]].mean()
    else:
        per_window_df = pd.DataFrame()
        average_metrics = pd.DataFrame()

    return {
        "per_window_metrics": per_window_df,
        "average_metrics": average_metrics,
        "histories": histories,
    }


def summarize_results(results: Mapping[str, Mapping[str, object]]) -> pd.DataFrame:
    """Create one summary table with macro metrics for each model."""

    rows = []
    for model_name, result in results.items():
        average_metrics = result["average_metrics"]
        if average_metrics is None or len(average_metrics) == 0:
            continue
        metric_frame = average_metrics.copy()
        rows.append(
            {
                "model": model_name,
                "accuracy": float(metric_frame["accuracy"].mean()),
                "recall": float(metric_frame["recall"].mean()),
                "precision": float(metric_frame["precision"].mean()),
                "f1": float(metric_frame["f1"].mean()),
                "auc": float(metric_frame["auc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def plot_summary_bar(summary: pd.DataFrame, metric: str = "auc", figsize: Tuple[int, int] = (10, 4)):
    """Plot a bar chart for one summary metric."""

    import matplotlib.pyplot as plt

    if metric not in summary.columns:
        raise ValueError(f"Unknown metric: {metric}")
    ax = summary.sort_values(metric, ascending=True).plot.barh(x="model", y=metric, legend=False, figsize=figsize)
    ax.set_xlabel(metric.upper())
    ax.set_ylabel("Model")
    ax.set_title(f"Model comparison by {metric.upper()}")
    plt.tight_layout()
    return ax


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, target_rooms: Sequence[str] = TARGET_ROOMS):
    """Plot ROC curves for all target rooms."""

    import matplotlib.pyplot as plt

    auc_values = {}
    for i, room in enumerate(target_rooms):
        if len(np.unique(y_true[:, i])) < 2:
            auc_values[room] = np.nan
            continue
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_values[room] = roc_auc
        plt.plot(fpr, tpr, label=f"{room} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves per target room")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    return auc_values
