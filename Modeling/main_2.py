#!/usr/bin/env python3
"""
US New Car Market Nowcasting System - Columnar Storage Architecture
====================================================================
Estimates non-reporting brand car sales from reporting brand sales data.

Architecture: Columnar storage with dimension tables for efficient
hierarchical nowcasting and reconciliation at billion-record scale.

Installation:
    pip install pandas numpy scikit-learn lightgbm catboost xgboost --break-system-packages

Usage:
    python nowcasting_system.py
"""

import warnings
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, ElasticNet, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

# =============================================================================
# (0) LOGGING AND DEBUG
# =============================================================================

@dataclass
class DebugInfo:
    """Central storage for debug information."""
    errors: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_details: Dict[str, dict] = field(default_factory=dict)
    warnings: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fallbacks: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    fallback_details: Dict[str, str] = field(default_factory=dict)  # key -> description
    printed_errors: set = field(default_factory=set)
    printed_warnings: set = field(default_factory=set)
    logs: List[dict] = field(default_factory=list)
    model_results: Dict[str, Any] = field(default_factory=dict)
    reconciliation_results: List[Any] = field(default_factory=list)
    best_models: Dict[str, str] = field(default_factory=dict)

debug = DebugInfo()


@dataclass
class ResultEntry:
    """Single result entry with context for comparison."""
    level: str
    origin: str  # model name or reconciliation descriptor
    cutoff_train: pd.Timestamp
    cutoff_test: pd.Timestamp
    predictions: Optional['LevelData'] = None
    mae: float = 0.0
    mae_pct_vs_zero: float = 0.0  # 100 * mae / zero_mae


class ResultsRegistry:
    """Registry for all computed results with context."""
    
    def __init__(self):
        self.entries: Dict[str, Dict[str, ResultEntry]] = defaultdict(dict)  # level -> origin -> entry
        self.zero_maes: Dict[str, float] = {}  # level -> zero MAE
    
    def register_zero_mae(self, level: str, zero_mae: float):
        """Register zero model MAE for a level (baseline for comparison)."""
        self.zero_maes[level] = zero_mae
    
    def register(self, level: str, origin: str, predictions: 'LevelData',
                 cutoff_train: pd.Timestamp, cutoff_test: pd.Timestamp):
        """Register a result entry."""
        if predictions is None or predictions.actuals is None:
            return
        
        mae = float(np.mean(np.abs(predictions.values - predictions.actuals)))
        zero_mae = self.zero_maes.get(level, 1.0)
        mae_pct = (mae / zero_mae * 100) if zero_mae > 0 else 0.0
        
        entry = ResultEntry(
            level=level,
            origin=origin,
            cutoff_train=cutoff_train,
            cutoff_test=cutoff_test,
            predictions=predictions,
            mae=mae,
            mae_pct_vs_zero=mae_pct
        )
        self.entries[level][origin] = entry
    
    def get_level_entries(self, level: str) -> Dict[str, ResultEntry]:
        """Get all entries for a level."""
        return self.entries.get(level, {})
    
    def get_best_n(self, level: str, n: int = 3) -> List[ResultEntry]:
        """Get top N entries by MAE % (lowest is best)."""
        entries = list(self.entries.get(level, {}).values())
        sorted_entries = sorted(entries, key=lambda e: e.mae_pct_vs_zero)
        return sorted_entries[:n]
    
    def get_all_levels(self) -> List[str]:
        """Get all levels with registered entries."""
        return list(self.entries.keys())


results_registry = ResultsRegistry()


def log(msg: str, level: str = 'info'):
    debug.logs.append({'level': level, 'msg': msg, 'time': time.time()})


def log_error(key: str, msg: str, exc: Exception = None):
    debug.errors[key] += 1
    tb_str = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)) if exc else None
    if key not in debug.error_details:
        debug.error_details[key] = {'msg': msg, 'traceback': tb_str, 'count': 1}
    else:
        debug.error_details[key]['count'] += 1
    if key not in debug.printed_errors:
        print(f"[ERROR] {msg}")
        if tb_str:
            print(f"[TRACEBACK]\n{tb_str}")
        debug.printed_errors.add(key)


def log_warning(key: str, msg: str):
    """Record warning without printing. Will be shown in summary."""
    debug.warnings[key] += 1


def log_fallback(key: str, msg: str):
    """Record fallback usage without printing. Will be shown in summary."""
    debug.fallbacks[key] += 1
    if key not in debug.fallback_details:
        debug.fallback_details[key] = msg


def print_progress(msg: str):
    print(f"[PROGRESS] {msg}")


def print_error_summary():
    print("\n" + "=" * 60)
    print("ERROR/WARNING/FALLBACK SUMMARY")
    print("=" * 60)
    if debug.errors:
        print(f"Errors ({sum(debug.errors.values())} total):")
        for k, v in sorted(debug.errors.items()):
            print(f"  - {k}: {v}")
            if k in debug.error_details and debug.error_details[k]['traceback']:
                for line in debug.error_details[k]['traceback'].split('\n')[:5]:
                    if line.strip():
                        print(f"      {line}")
    else:
        print("No errors.")
    if debug.warnings:
        print(f"\nWarnings ({sum(debug.warnings.values())} total):")
        for k, v in sorted(debug.warnings.items()):
            print(f"  - {k}: {v}")
    else:
        print("No warnings.")
    if debug.fallbacks:
        print(f"\nFallbacks ({sum(debug.fallbacks.values())} total):")
        for k, v in sorted(debug.fallbacks.items()):
            desc = debug.fallback_details.get(k, '')
            print(f"  - {k}: {v} occurrences")
            if desc:
                print(f"      {desc}")
    else:
        print("No fallbacks.")
    print("=" * 60)


# =============================================================================
# (1) DIMENSION TABLES AND REGISTRY
# =============================================================================

@dataclass
class DimensionTable:
    """
    Single dimension with integer encoding and hierarchy mappings.
    
    Attributes:
        name: Dimension name (e.g., 'model', 'tract', 'date')
        values: Array of unique values in order of their integer IDs
        value_to_id: Mapping from value to integer ID
        parents: Dict mapping parent dimension name to ID mapping array
    """
    name: str
    values: np.ndarray  # Original values (strings, dates, etc.)
    value_to_id: Dict[Any, int] = field(default_factory=dict)
    parents: Dict[str, np.ndarray] = field(default_factory=dict)
    
    @classmethod
    def from_values(cls, name: str, values: np.ndarray) -> 'DimensionTable':
        """Create dimension table from array of values."""
        unique_vals = np.unique(values)
        value_to_id = {v: i for i, v in enumerate(unique_vals)}
        return cls(name=name, values=unique_vals, value_to_id=value_to_id)
    
    def encode(self, values: np.ndarray) -> np.ndarray:
        """Encode values to integer IDs. Returns -1 for unknown values."""
        result = np.zeros(len(values), dtype=np.int32)
        for i, v in enumerate(values):
            if v in self.value_to_id:
                result[i] = self.value_to_id[v]
            else:
                result[i] = -1
                log_fallback(f'encode_unknown_{self.name}', 
                             f'Unknown value in dimension {self.name}, encoded as -1')
        return result
    
    def decode(self, ids: np.ndarray) -> np.ndarray:
        """Decode integer IDs back to original values."""
        return self.values[ids]
    
    def add_parent(self, parent_name: str, mapping: np.ndarray):
        """Add parent dimension mapping. mapping[child_id] = parent_id."""
        self.parents[parent_name] = mapping.astype(np.int32)
    
    def map_to_parent(self, parent_name: str, child_ids: np.ndarray) -> np.ndarray:
        """Map child IDs to parent IDs."""
        return self.parents[parent_name][child_ids]
    
    def __len__(self) -> int:
        return len(self.values)


@dataclass
class DimensionRegistry:
    """Central registry of all dimensions."""
    dimensions: Dict[str, DimensionTable] = field(default_factory=dict)
    
    def add(self, dim: DimensionTable):
        self.dimensions[dim.name] = dim
    
    def get(self, name: str) -> DimensionTable:
        return self.dimensions[name]
    
    def encode(self, name: str, values: np.ndarray) -> np.ndarray:
        return self.dimensions[name].encode(values)
    
    def has(self, name: str) -> bool:
        return name in self.dimensions


# =============================================================================
# (2) LEVEL DEFINITIONS AND REGISTRY
# =============================================================================

@dataclass
class LevelSpec:
    """Specification of an evaluation level."""
    name: str
    dims: List[str]  # Dimension names at this level
    sparsity: str  # 'sparse', 'moderate', 'dense'
    
    # For aggregation: maps each dim to its aggregated form at this level
    # e.g., {'model': 'brand', 'tract': 'state', 'date': 'week'}
    # If dim maps to itself, it's kept as-is
    dim_sources: Dict[str, str] = field(default_factory=dict)


@dataclass
class LevelRegistry:
    """Registry of all evaluation levels."""
    levels: Dict[str, LevelSpec] = field(default_factory=dict)
    level_order: List[str] = field(default_factory=list)  # Finest to coarsest
    
    def add(self, level: LevelSpec):
        self.levels[level.name] = level
        if level.name not in self.level_order:
            self.level_order.append(level.name)
    
    def get(self, name: str) -> LevelSpec:
        return self.levels[name]
    
    def get_dims(self, name: str) -> List[str]:
        return self.levels[name].dims


# =============================================================================
# (3) LEVEL DATA - VALUES AT A SPECIFIC LEVEL
# =============================================================================

@dataclass
class LevelData:
    """
    Values stored at a specific evaluation level.
    
    Uses parallel arrays for memory efficiency:
    - keys: (N, D) int32 array of dimension IDs
    - values: (N,) float32 array of values
    """
    level: str
    dim_names: List[str]  # Names of dimensions in key columns
    keys: np.ndarray  # (N, D) int32 - dimension IDs
    values: np.ndarray  # (N,) float32
    
    # Optional fields
    actuals: Optional[np.ndarray] = None  # Ground truth values
    _key_index: Optional[Dict[tuple, int]] = None
    
    def __len__(self) -> int:
        return len(self.values)
    
    @property
    def n_dims(self) -> int:
        return len(self.dim_names)
    
    def build_index(self):
        """Build lookup index for random access."""
        self._key_index = {tuple(k): i for i, k in enumerate(self.keys)}
    
    def get_value(self, key: tuple) -> float:
        """Get value for a specific key tuple."""
        if self._key_index is None:
            self.build_index()
        idx = self._key_index.get(key)
        return self.values[idx] if idx is not None else 0.0
    
    def get_key_strings(self, dim_registry: DimensionRegistry) -> List[str]:
        """Get human-readable key strings for display."""
        result = []
        for row in self.keys:
            parts = []
            for i, dim_name in enumerate(self.dim_names):
                if dim_registry.has(dim_name):
                    dim = dim_registry.get(dim_name)
                    parts.append(str(dim.values[row[i]]))
                else:
                    parts.append(str(row[i]))
            result.append('|'.join(parts))
        return result
    
    def aggregate_to(self, target_level: str, target_dims: List[str],
                     dim_registry: DimensionRegistry) -> 'LevelData':
        """
        Aggregate this level's data to a coarser level.
        
        Args:
            target_level: Name of target level
            target_dims: Dimension names at target level
            dim_registry: Registry for dimension mappings
        """
        n_records = len(self.keys)
        n_target_dims = len(target_dims)
        
        # Build parent keys
        parent_keys = np.zeros((n_records, n_target_dims), dtype=np.int32)
        
        for i, target_dim in enumerate(target_dims):
            if target_dim in self.dim_names:
                # Same dimension, copy directly
                src_idx = self.dim_names.index(target_dim)
                parent_keys[:, i] = self.keys[:, src_idx]
            else:
                # Need to map through dimension hierarchy
                # Find which source dim maps to this target
                mapped = False
                for j, src_dim in enumerate(self.dim_names):
                    if dim_registry.has(src_dim):
                        dim_table = dim_registry.get(src_dim)
                        if target_dim in dim_table.parents:
                            parent_keys[:, i] = dim_table.map_to_parent(target_dim, self.keys[:, j])
                            mapped = True
                            break
                if not mapped:
                    # Constant dimension (e.g., ALL_BRANDS) - use 0
                    parent_keys[:, i] = 0
        
        # Aggregate using lexsort + reduceat for efficiency
        # First, get unique parent keys and their sums
        sort_idx = np.lexsort(parent_keys.T[::-1])
        sorted_keys = parent_keys[sort_idx]
        sorted_values = self.values[sort_idx]
        
        # Find group boundaries
        if len(sorted_keys) > 1:
            diff = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
            boundaries = np.concatenate([[0], np.where(diff)[0] + 1, [len(sorted_keys)]])
        else:
            boundaries = np.array([0, len(sorted_keys)])
        
        # Get unique keys and sum values
        unique_keys = sorted_keys[boundaries[:-1]]
        sums = np.add.reduceat(sorted_values, boundaries[:-1])
        
        return LevelData(
            level=target_level,
            dim_names=target_dims,
            keys=unique_keys,
            values=sums.astype(np.float32)
        )
    
    def disaggregate_from(self, parent_data: 'LevelData',
                          dim_registry: DimensionRegistry) -> 'LevelData':
        """
        Top-down disaggregation: distribute parent values to this level
        using self.values as reference distribution.
        """
        # Map this level's keys to parent level
        n_records = len(self.keys)
        parent_dims = parent_data.dim_names
        n_parent_dims = len(parent_dims)
        
        mapped_parent_keys = np.zeros((n_records, n_parent_dims), dtype=np.int32)
        
        for i, parent_dim in enumerate(parent_dims):
            if parent_dim in self.dim_names:
                src_idx = self.dim_names.index(parent_dim)
                mapped_parent_keys[:, i] = self.keys[:, src_idx]
            else:
                for j, src_dim in enumerate(self.dim_names):
                    if dim_registry.has(src_dim):
                        dim_table = dim_registry.get(src_dim)
                        if parent_dim in dim_table.parents:
                            mapped_parent_keys[:, i] = dim_table.map_to_parent(parent_dim, self.keys[:, j])
                            break
                else:
                    mapped_parent_keys[:, i] = 0
        
        # Build parent index
        parent_data.build_index()
        
        # Group children by parent key
        unique_parent_keys, inverse = np.unique(mapped_parent_keys, axis=0, return_inverse=True)
        
        disaggregated = np.zeros(n_records, dtype=np.float32)
        
        for parent_idx, parent_key in enumerate(unique_parent_keys):
            child_mask = (inverse == parent_idx)
            child_refs = self.values[child_mask]
            
            parent_value = parent_data.get_value(tuple(parent_key))
            
            total_ref = child_refs.sum()
            if total_ref > 0:
                disaggregated[child_mask] = parent_value * (child_refs / total_ref)
            else:
                n_children = child_mask.sum()
                if n_children > 0:
                    log_fallback('disaggregate_equal_split', 
                                 'No reference weights, distributing parent value equally')
                    disaggregated[child_mask] = parent_value / n_children
        
        return LevelData(
            level=self.level,
            dim_names=self.dim_names.copy(),
            keys=self.keys.copy(),
            values=disaggregated
        )
    
    def compute_mae(self) -> float:
        """Compute MAE between values and actuals."""
        if self.actuals is None:
            return 0.0
        # Handle shape mismatch by using index alignment
        if len(self.values) != len(self.actuals):
            # Need to align by keys
            return 0.0  # Can't compute without alignment
        return float(np.mean(np.abs(self.values - self.actuals)))
    
    def compute_correlation(self) -> Optional[float]:
        """Compute correlation between values and actuals."""
        if self.actuals is None or len(self.values) < 2:
            return None
        if np.std(self.values) == 0 or np.std(self.actuals) == 0:
            return None
        return np.corrcoef(self.values, self.actuals)[0, 1]


# =============================================================================
# (4) FACT TABLE - RAW DATA STORAGE
# =============================================================================

@dataclass
class FactTable:
    """
    Columnar storage for raw transaction data.
    
    Each column is a numpy array. Dimension columns store integer IDs.
    """
    # Dimension columns (int32 IDs)
    model_id: np.ndarray
    tract_id: np.ndarray
    date_id: np.ndarray
    brand_id: np.ndarray
    segment_id: np.ndarray
    county_id: np.ndarray
    state_id: np.ndarray
    week_id: np.ndarray
    
    # Measure columns
    vin_count: np.ndarray  # float32
    is_reporting: np.ndarray  # bool
    
    # Date columns for simulation
    ownshp_dt: Optional[np.ndarray] = None  # datetime64
    nvi_efctv_start_dt: Optional[np.ndarray] = None  # datetime64
    
    # Feature columns
    day_of_week: Optional[np.ndarray] = None  # int32
    day_of_month: Optional[np.ndarray] = None  # int32
    model_price: Optional[np.ndarray] = None  # float32
    segment_price: Optional[np.ndarray] = None  # float32
    brand_price: Optional[np.ndarray] = None  # float32
    
    def __len__(self) -> int:
        return len(self.vin_count)
    
    def filter(self, mask: np.ndarray) -> 'FactTable':
        """Return new FactTable filtered by boolean mask."""
        return FactTable(
            model_id=self.model_id[mask],
            tract_id=self.tract_id[mask],
            date_id=self.date_id[mask],
            brand_id=self.brand_id[mask],
            segment_id=self.segment_id[mask],
            county_id=self.county_id[mask],
            state_id=self.state_id[mask],
            week_id=self.week_id[mask],
            vin_count=self.vin_count[mask],
            is_reporting=self.is_reporting[mask],
            ownshp_dt=self.ownshp_dt[mask] if self.ownshp_dt is not None else None,
            nvi_efctv_start_dt=self.nvi_efctv_start_dt[mask] if self.nvi_efctv_start_dt is not None else None,
            day_of_week=self.day_of_week[mask] if self.day_of_week is not None else None,
            day_of_month=self.day_of_month[mask] if self.day_of_month is not None else None,
            model_price=self.model_price[mask] if self.model_price is not None else None,
            segment_price=self.segment_price[mask] if self.segment_price is not None else None,
            brand_price=self.brand_price[mask] if self.brand_price is not None else None
        )
    
    @property
    def reporting(self) -> 'FactTable':
        """Get reporting brand records only."""
        return self.filter(self.is_reporting)
    
    @property
    def nonreporting(self) -> 'FactTable':
        """Get non-reporting brand records only."""
        return self.filter(~self.is_reporting)
    
    def aggregate_to_level(self, level_spec: LevelSpec, 
                           dim_registry: DimensionRegistry) -> LevelData:
        """Aggregate fact table to specified level."""
        dim_names = level_spec.dims
        n_dims = len(dim_names)
        n_records = len(self)
        
        # Build keys array
        keys = np.zeros((n_records, n_dims), dtype=np.int32)
        
        dim_column_map = {
            'model_key_short': self.model_id,
            'tract': self.tract_id,
            'date': self.date_id,
            'brand': self.brand_id,
            'segment': self.segment_id,
            'county': self.county_id,
            'state': self.state_id,
            'week': self.week_id,
        }
        
        for i, dim_name in enumerate(dim_names):
            if dim_name in dim_column_map:
                keys[:, i] = dim_column_map[dim_name]
            elif dim_name.startswith('ALL_'):
                keys[:, i] = 0  # Constant dimension
            else:
                log_warning(f'unknown_dim_{dim_name}', f"Unknown dimension: {dim_name}")
                keys[:, i] = 0
        
        # Create LevelData and aggregate
        temp_data = LevelData(
            level=level_spec.name,
            dim_names=dim_names,
            keys=keys,
            values=self.vin_count.astype(np.float32)
        )
        
        # Aggregate to unique keys
        return temp_data.aggregate_to(level_spec.name, dim_names, dim_registry)


# =============================================================================
# (5) DATA LOADING AND PREPARATION
# =============================================================================

def load_and_encode_data(filepath: str, dim_registry: DimensionRegistry, force_min_date=None, force_max_date=None) -> Tuple[FactTable, pd.Timestamp, pd.Timestamp]:
    """
    Load CSV data and encode all dimensions.
    
    Returns:
        FactTable, min_date, max_date
    """
    print_progress(f"Loading data from {filepath}")
    
    try:
        df = pd.read_csv(filepath, low_memory=False)
    except Exception as e:
        log_error('file_load', f"Failed to load {filepath}: {e}", exc=e)
        return None, None, None
    
    print_progress(f"Loaded {len(df)} rows")
    
    # Convert dates
    df['OWNSHP_DT'] = pd.to_datetime(df['OWNSHP_DT'], errors='coerce')
    df['NVI_EFCTV_START_DT'] = pd.to_datetime(df['NVI_EFCTV_START_DT'], errors='coerce')
    df['NVI_SYS_CREATE_DT'] = pd.to_datetime(df['NVI_SYS_CREATE_DT'], errors='coerce')

    if force_min_date is not None:
        df = df[df['OWNSHP_DT'] >= pd.to_datetime(force_min_date)]
    if force_max_date is not None:
        df = df[df['OWNSHP_DT'] <= pd.to_datetime(force_max_date)]
    
    min_date = df['OWNSHP_DT'].min()
    max_date = df['OWNSHP_DT'].max()
    
    # Prepare columns
    df['sale_date'] = df['OWNSHP_DT'].dt.date.astype(str)
    df['week'] = df['OWNSHP_DT'].dt.isocalendar().week.astype(int)
    df['day_of_week'] = df['OWNSHP_DT'].dt.dayofweek
    df['day_of_month'] = df['OWNSHP_DT'].dt.day
    
    # Handle missing values - convert to string first to handle mixed types
    df['state'] = df['NVI_STATE_from_ct'].fillna(df.get('SLS_STATE_ABBRV', 'XX')).astype(str)
    df['COUNTY'] = df['COUNTY'].fillna('UNKNOWN').astype(str)
    df['CENSUS_TRACT'] = df['NVI_CENSUS_TRACT'].fillna('UNKNOWN').astype(str)
    df['SEGMENT_DESC'] = df['SEGMENT_DESC'].fillna('UNKNOWN').astype(str)
    df['MODEL_DESC'] = df['MODEL_DESC'].fillna('UNKNOWN').astype(str)
    df['MAKE_DESC'] = df['MAKE_DESC'].fillna('UNKNOWN').astype(str)
    df['MODEL_KEY_SHORT'] = df['MODEL_KEY_SHORT'].fillna('UNKNOWN').astype(str)
    
    # Create dimension tables
    # Model dimension with brand and segment parents
    model_dim = DimensionTable.from_values('model_key_short', df['MODEL_KEY_SHORT'].values)
    dim_registry.add(model_dim)
    
    brand_dim = DimensionTable.from_values('brand', df['MAKE_DESC'].values)
    dim_registry.add(brand_dim)
    
    segment_dim = DimensionTable.from_values('segment', df['SEGMENT_DESC'].values)
    dim_registry.add(segment_dim)
    
    # Build model -> brand and model -> segment mappings
    model_brand_map = np.zeros(len(model_dim), dtype=np.int32)
    model_segment_map = np.zeros(len(model_dim), dtype=np.int32)
    
    for _, row in df[['MODEL_KEY_SHORT', 'MAKE_DESC', 'SEGMENT_DESC']].drop_duplicates().iterrows():
        model_id = model_dim.value_to_id.get(row['MODEL_KEY_SHORT'], -1)
        brand_id = brand_dim.value_to_id.get(row['MAKE_DESC'], 0)
        segment_id = segment_dim.value_to_id.get(row['SEGMENT_DESC'], 0)
        if model_id >= 0:
            model_brand_map[model_id] = brand_id
            model_segment_map[model_id] = segment_id
    
    model_dim.add_parent('brand', model_brand_map)
    model_dim.add_parent('segment', model_segment_map)
    
    # Tract dimension with county and state parents
    tract_dim = DimensionTable.from_values('tract', df['CENSUS_TRACT'].values)
    dim_registry.add(tract_dim)
    
    county_dim = DimensionTable.from_values('county', df['COUNTY'].values)
    dim_registry.add(county_dim)
    
    state_dim = DimensionTable.from_values('state', df['state'].values)
    dim_registry.add(state_dim)
    
    # Build tract -> county and tract -> state mappings
    tract_county_map = np.zeros(len(tract_dim), dtype=np.int32)
    tract_state_map = np.zeros(len(tract_dim), dtype=np.int32)
    
    for _, row in df[['CENSUS_TRACT', 'COUNTY', 'state']].drop_duplicates().iterrows():
        tract_id = tract_dim.value_to_id.get(row['CENSUS_TRACT'], -1)
        county_id = county_dim.value_to_id.get(row['COUNTY'], 0)
        state_id = state_dim.value_to_id.get(row['state'], 0)
        if tract_id >= 0:
            tract_county_map[tract_id] = county_id
            tract_state_map[tract_id] = state_id
    
    tract_dim.add_parent('county', tract_county_map)
    tract_dim.add_parent('state', tract_state_map)
    
    # County -> state mapping
    county_state_map = np.zeros(len(county_dim), dtype=np.int32)
    for _, row in df[['COUNTY', 'state']].drop_duplicates().iterrows():
        county_id = county_dim.value_to_id.get(row['COUNTY'], -1)
        state_id = state_dim.value_to_id.get(row['state'], 0)
        if county_id >= 0:
            county_state_map[county_id] = state_id
    county_dim.add_parent('state', county_state_map)
    
    # Date dimension with week parent
    date_dim = DimensionTable.from_values('date', df['sale_date'].values)
    dim_registry.add(date_dim)
    
    week_dim = DimensionTable.from_values('week', df['week'].values)
    dim_registry.add(week_dim)
    
    # Build date -> week mapping
    date_week_map = np.zeros(len(date_dim), dtype=np.int32)
    for _, row in df[['sale_date', 'week']].drop_duplicates().iterrows():
        date_id = date_dim.value_to_id.get(row['sale_date'], -1)
        week_id = week_dim.value_to_id.get(row['week'], 0)
        if date_id >= 0:
            date_week_map[date_id] = week_id
    date_dim.add_parent('week', date_week_map)
    
    # Constant dimensions
    all_brands_dim = DimensionTable(name='ALL_BRANDS', values=np.array(['ALL_BRANDS']),
                                     value_to_id={'ALL_BRANDS': 0})
    dim_registry.add(all_brands_dim)
    
    all_states_dim = DimensionTable(name='ALL_STATES', values=np.array(['ALL_STATES']),
                                     value_to_id={'ALL_STATES': 0})
    dim_registry.add(all_states_dim)
    
    all_time_dim = DimensionTable(name='ALL_TIME', values=np.array(['ALL_TIME']),
                                   value_to_id={'ALL_TIME': 0})
    dim_registry.add(all_time_dim)
    
    # Encode all columns
    model_ids = model_dim.encode(df['MODEL_KEY_SHORT'].values)
    tract_ids = tract_dim.encode(df['CENSUS_TRACT'].values)
    date_ids = date_dim.encode(df['sale_date'].values)
    brand_ids = brand_dim.encode(df['MAKE_DESC'].values)
    segment_ids = segment_dim.encode(df['SEGMENT_DESC'].values)
    county_ids = county_dim.encode(df['COUNTY'].values)
    state_ids = state_dim.encode(df['state'].values)
    week_ids = week_dim.encode(df['week'].values)
    
    # Handle price columns - fill NaN with 0
    model_price = pd.to_numeric(df['model_price'], errors='coerce').fillna(0).values.astype(np.float32)
    segment_price = pd.to_numeric(df['segment_price'], errors='coerce').fillna(0).values.astype(np.float32)
    brand_price = pd.to_numeric(df['brand_price'], errors='coerce').fillna(0).values.astype(np.float32)
    
    # Create fact table
    fact_table = FactTable(
        model_id=model_ids,
        tract_id=tract_ids,
        date_id=date_ids,
        brand_id=brand_ids,
        segment_id=segment_ids,
        county_id=county_ids,
        state_id=state_ids,
        week_id=week_ids,
        vin_count=pd.to_numeric(df['NVI_VIN_COUNT'], errors='coerce').fillna(1).values.astype(np.float32),
        is_reporting=df['is_reporting_brand'].values.astype(bool),
        ownshp_dt=df['OWNSHP_DT'].values,
        nvi_efctv_start_dt=df['NVI_SYS_CREATE_DT'].values,  #also here: date when reg info becomes publicly available. #abc was NVI_EFCTV_START_DT, now NVI_SYS_CREATE_DT
        day_of_week=df['day_of_week'].values.astype(np.int32),
        day_of_month=df['day_of_month'].values.astype(np.int32),
        model_price=model_price,
        segment_price=segment_price,
        brand_price=brand_price
    )
    
    print_progress(f"Created fact table: {len(fact_table)} records")
    print_progress(f"  Dimensions: {len(model_dim)} models, {len(tract_dim)} tracts, {len(date_dim)} dates")
    print_progress(f"  Reporting: {fact_table.is_reporting.sum()}, Non-reporting: {(~fact_table.is_reporting).sum()}")
    
    return fact_table, min_date, max_date


def split_data_two_cutoff(fact_table: FactTable, 
                          cutoff_train: pd.Timestamp,
                          cutoff_test: pd.Timestamp) -> Tuple[FactTable, FactTable]:
    """
    Two-cutoff split for realistic simulation of registration delay.
    
    At simulation time t (cutoff_train):
      - Training features: reporting records where NVI_EFCTV_START_DT <= t
      - Training target: non-reporting records where OWNSHP_DT <= t AND NVI_EFCTV_START_DT > t
                         AND NVI_EFCTV_START_DT <= T (registered by test time, so observable)
        (Records that were "missing" at t but have since been registered by T)
    
    At simulation time T (cutoff_test):
      - Test features: reporting records where NVI_EFCTV_START_DT <= T
      - Test target: non-reporting records where OWNSHP_DT <= T AND NVI_EFCTV_START_DT > T
        (Records currently "missing" at T)
    
    Note: Training target excludes STILL_UNREG records (nvi_dt > T) because these
    would not be observable in a production setting at time T.
    
    Args:
        fact_table: Full fact table with all records
        cutoff_train: Training simulation time (t)
        cutoff_test: Test simulation time (T)
    
    Returns:
        (train_table, test_table)
    """
    # Convert datetime columns - handle numpy datetime64 arrays
    ownshp_dt = pd.DatetimeIndex(fact_table.ownshp_dt)
    nvi_dt = pd.DatetimeIndex(fact_table.nvi_efctv_start_dt)
    is_reporting = fact_table.is_reporting
    
    # Training data
    # Features: reporting, NVI_EFCTV_START_DT <= cutoff_train
    train_rep_mask = is_reporting & (nvi_dt <= cutoff_train)
    # Target: non-reporting, OWNSHP_DT <= cutoff_train AND NVI_EFCTV_START_DT > cutoff_train
    #         AND NVI_EFCTV_START_DT <= cutoff_test (must be registered by T to be observable)
    # (Was missing at cutoff_train, registered between t and T, so we can learn from it)
    train_nonrep_mask = ((~is_reporting) & (ownshp_dt <= cutoff_train) & 
                         (nvi_dt > cutoff_train) & (nvi_dt <= cutoff_test))
    train_mask = train_rep_mask | train_nonrep_mask
    
    # Test data
    # Features: reporting, NVI_EFCTV_START_DT <= cutoff_test
    test_rep_mask = is_reporting & (nvi_dt <= cutoff_test)
    # Target: non-reporting, OWNSHP_DT <= cutoff_test AND NVI_EFCTV_START_DT > cutoff_test
    # (Currently missing at cutoff_test)
    test_nonrep_mask = (~is_reporting) & (ownshp_dt <= cutoff_test) & (nvi_dt > cutoff_test)
    test_mask = test_rep_mask | test_nonrep_mask
    
    train_table = fact_table.filter(np.array(train_mask))
    test_table = fact_table.filter(np.array(test_mask))
    
    print_progress(f"Training simulation time: {cutoff_train}")
    print_progress(f"  Reporting (visible): {train_rep_mask.sum()}")
    print_progress(f"  Non-reporting (resolved between t and T): {train_nonrep_mask.sum()}")
    print_progress(f"Test simulation time: {cutoff_test}")
    print_progress(f"  Reporting (visible): {test_rep_mask.sum()}")
    print_progress(f"  Non-reporting (still missing at T): {test_nonrep_mask.sum()}")
    
    return train_table, test_table


def split_data(fact_table: FactTable, date_dim: DimensionTable,
               cutoff_dt: pd.Timestamp, df_original: pd.DataFrame) -> Tuple[FactTable, FactTable]:
    """
    Legacy split function - kept for backwards compatibility.
    
    Training: records with OWNSHP_DT < cutoff
    Test (for nowcasting): reporting from training period, non-reporting late registrations
    """
    # Need original dataframe for date filtering
    ownshp_dt = pd.to_datetime(df_original['OWNSHP_DT'])
    nvi_dt = pd.to_datetime(df_original['NVI_SYS_CREATE_DT']) #this is the date when registration information becomes public #abc #was NVI_EFCTV_START_DT, now NVI_SYS_CREATE_DT
    
    # Training mask
    train_rep_mask = (ownshp_dt < cutoff_dt) & fact_table.is_reporting
    train_nonrep_mask = (ownshp_dt < cutoff_dt) & (nvi_dt < cutoff_dt) & (~fact_table.is_reporting)
    train_mask = train_rep_mask | train_nonrep_mask
    
    # Test mask: reporting from training period + late registrations
    # Late registrations: OWNSHP_DT < cutoff but NVI_EFCTV_START_DT >= cutoff
    test_rep_mask = train_rep_mask  # Use training reporting as features
    test_nonrep_mask = (ownshp_dt < cutoff_dt) & (nvi_dt >= cutoff_dt) & (~fact_table.is_reporting)
    test_mask = test_rep_mask | test_nonrep_mask
    
    train_table = fact_table.filter(train_mask)
    test_table = fact_table.filter(test_mask)
    
    print_progress(f"Training: {len(train_table)} records (rep: {train_table.is_reporting.sum()}, nonrep: {(~train_table.is_reporting).sum()})")
    print_progress(f"Test: {len(test_table)} records (rep: {test_table.is_reporting.sum()}, nonrep: {(~test_table.is_reporting).sum()})")
    
    return train_table, test_table


# =============================================================================
# (6) FEATURE PREPARATION FOR MODELS
# =============================================================================

@dataclass
class PreparedData:
    """Prepared data for model training/inference."""
    # Feature matrix
    X: np.ndarray  # (N, F) float32
    feature_names: List[str]
    
    # Target
    y: np.ndarray  # (N,) float32 - non-reporting sales
    
    # Keys for result mapping
    keys: np.ndarray  # (N, D) int32
    dim_names: List[str]
    
    # Metadata
    reporting_sum: float
    nonreporting_sum: float
    
    # For tracking new bins
    group_keys: Optional[np.ndarray] = None  # Non-temporal group keys


def prepare_level_data(train_table: FactTable, test_table: FactTable,
                       level_spec: LevelSpec, dim_registry: DimensionRegistry) -> Tuple[PreparedData, PreparedData]:
    """
    Prepare training and test data for a specific level.
    
    Features include:
    - reporting_sales: sum of reporting vin_count
    - model_price: weighted avg price (by vin_count)
    - segment_price: weighted avg price
    - brand_price: weighted avg price
    - day_of_week: mean day of week (for levels with date dimension)
    - day_of_month: mean day of month (for levels with date dimension)
    
    Returns:
        (train_data, test_data) as PreparedData objects
    """
    dims = level_spec.dims
    
    # Identify feature dimensions (geo + time) vs vehicle dimensions
    vehicle_dims = {'model_key_short', 'brand', 'segment'}
    feature_dims = [d for d in dims if d not in vehicle_dims]
    if not feature_dims:
        feature_dims = ['state', 'ALL_TIME']
    
    # Check if level has date/week dimension (for day features)
    has_date_dim = 'date' in dims or 'week' in dims
    
    # Get dimension column mapping
    def get_dim_column(table: FactTable, dim: str) -> np.ndarray:
        col_map = {
            'model_key_short': table.model_id,
            'tract': table.tract_id,
            'date': table.date_id,
            'brand': table.brand_id,
            'segment': table.segment_id,
            'county': table.county_id,
            'state': table.state_id,
            'week': table.week_id,
        }
        if dim in col_map:
            return col_map[dim]
        elif dim.startswith('ALL_'):
            return np.zeros(len(table), dtype=np.int32)
        return np.zeros(len(table), dtype=np.int32)
    
    def prepare_single(table: FactTable) -> PreparedData:
        # Separate reporting and non-reporting
        rep_mask = table.is_reporting
        nonrep_mask = ~table.is_reporting
        
        # Build keys for all dimensions
        n_dims = len(dims)
        
        # Aggregate non-reporting by full dimensions (target)
        nonrep_table = table.filter(nonrep_mask)
        nonrep_keys = np.column_stack([get_dim_column(nonrep_table, d) for d in dims])
        nonrep_values = nonrep_table.vin_count
        
        if len(nonrep_keys) > 0:
            # Aggregate to unique keys
            sort_idx = np.lexsort(nonrep_keys.T[::-1])
            sorted_keys = nonrep_keys[sort_idx]
            sorted_values = nonrep_values[sort_idx]
            
            if len(sorted_keys) > 1:
                diff = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
                boundaries = np.concatenate([[0], np.where(diff)[0] + 1, [len(sorted_keys)]])
            else:
                boundaries = np.array([0, len(sorted_keys)])
            
            unique_nonrep_keys = sorted_keys[boundaries[:-1]]
            nonrep_sums = np.add.reduceat(sorted_values, boundaries[:-1])
        else:
            unique_nonrep_keys = np.zeros((0, n_dims), dtype=np.int32)
            nonrep_sums = np.zeros(0, dtype=np.float32)
        
        # Aggregate reporting by feature dimensions only
        valid_feature_dims = [d for d in feature_dims if d in dims]
        if len(valid_feature_dims) == 0:
            valid_feature_dims = dims
        
        rep_table = table.filter(rep_mask)
        rep_feature_keys = np.column_stack([get_dim_column(rep_table, d) for d in valid_feature_dims])
        rep_values = rep_table.vin_count
        
        # Get feature arrays from reporting table
        rep_model_price = rep_table.model_price if rep_table.model_price is not None else np.zeros(len(rep_table))
        rep_segment_price = rep_table.segment_price if rep_table.segment_price is not None else np.zeros(len(rep_table))
        rep_brand_price = rep_table.brand_price if rep_table.brand_price is not None else np.zeros(len(rep_table))
        rep_day_of_week = rep_table.day_of_week if rep_table.day_of_week is not None else np.zeros(len(rep_table))
        rep_day_of_month = rep_table.day_of_month if rep_table.day_of_month is not None else np.zeros(len(rep_table))
        
        # Build reporting lookup with all features
        # rep_lookup[key] = (sum_vin, sum_vin*model_price, sum_vin*segment_price, sum_vin*brand_price, sum_dow, sum_dom, count, dow_0..dow_6)
        if len(rep_feature_keys) > 0:
            sort_idx = np.lexsort(rep_feature_keys.T[::-1])
            sorted_keys = rep_feature_keys[sort_idx]
            sorted_values = rep_values[sort_idx]
            sorted_model_price = rep_model_price[sort_idx]
            sorted_segment_price = rep_segment_price[sort_idx]
            sorted_brand_price = rep_brand_price[sort_idx]
            sorted_dow = rep_day_of_week[sort_idx]
            sorted_dom = rep_day_of_month[sort_idx]
            
            if len(sorted_keys) > 1:
                diff = np.any(sorted_keys[1:] != sorted_keys[:-1], axis=1)
                boundaries = np.concatenate([[0], np.where(diff)[0] + 1, [len(sorted_keys)]])
            else:
                boundaries = np.array([0, len(sorted_keys)])
            
            unique_rep_keys = sorted_keys[boundaries[:-1]]
            rep_sums = np.add.reduceat(sorted_values, boundaries[:-1])
            # Weighted sums for prices (weight = vin_count)
            rep_model_price_weighted = np.add.reduceat(sorted_values * sorted_model_price, boundaries[:-1])
            rep_segment_price_weighted = np.add.reduceat(sorted_values * sorted_segment_price, boundaries[:-1])
            rep_brand_price_weighted = np.add.reduceat(sorted_values * sorted_brand_price, boundaries[:-1])
            # Sum for day features (will divide by count later)
            rep_dow_sum = np.add.reduceat(sorted_dow.astype(np.float32), boundaries[:-1])
            rep_dom_sum = np.add.reduceat(sorted_dom.astype(np.float32), boundaries[:-1])
            rep_counts = np.diff(boundaries)
            
            # Per-weekday vin_count sums for one-hot encoding (7 arrays for days 0-6)
            dow_sums_per_day = []
            if USE_1HOT_WEEKDAYS:
                for day in range(7):
                    # vin_count where day_of_week == day, else 0
                    day_mask = (sorted_dow == day).astype(np.float32)
                    day_values = sorted_values * day_mask
                    dow_sums_per_day.append(np.add.reduceat(day_values, boundaries[:-1]))
            
            # Build lookup dict
            rep_lookup = {}
            for i, key in enumerate(unique_rep_keys):
                key_tuple = tuple(key)
                rep_lookup[key_tuple] = {
                    'vin_sum': rep_sums[i],
                    'model_price_weighted': rep_model_price_weighted[i],
                    'segment_price_weighted': rep_segment_price_weighted[i],
                    'brand_price_weighted': rep_brand_price_weighted[i],
                    'dow_sum': rep_dow_sum[i],
                    'dom_sum': rep_dom_sum[i],
                    'count': rep_counts[i]
                }
                # Add per-weekday sums
                if USE_1HOT_WEEKDAYS:
                    for day in range(7):
                        rep_lookup[key_tuple][f'dow_{day}'] = dow_sums_per_day[day][i]
        else:
            rep_lookup = {}
        
        # Build feature names based on feature toggles
        feature_names = ['reporting_sales']
        if USE_PRICE:
            feature_names.extend(['model_price', 'segment_price', 'brand_price'])
        if has_date_dim:
            feature_names.extend(['day_of_week', 'day_of_month'])
        if USE_1HOT_WEEKDAYS:
            feature_names.extend(['dow_mon', 'dow_tue', 'dow_wed', 'dow_thu', 'dow_fri', 'dow_sat', 'dow_sun'])
        n_features = len(feature_names)
        
        # Merge: for each non-reporting bin, get corresponding reporting features
        n_bins = len(unique_nonrep_keys)
        if n_bins == 0:
            return PreparedData(
                X=np.zeros((0, n_features), dtype=np.float32),
                feature_names=feature_names,
                y=np.zeros(0, dtype=np.float32),
                keys=np.zeros((0, n_dims), dtype=np.int32),
                dim_names=dims,
                reporting_sum=0.0,
                nonreporting_sum=0.0
            )
        
        # Build feature matrix
        X = np.zeros((n_bins, n_features), dtype=np.float32)
        
        for i in range(n_bins):
            # Build feature key from non-reporting bin
            feature_key_parts = []
            for fd in valid_feature_dims:
                if fd in dims:
                    idx = dims.index(fd)
                    feature_key_parts.append(unique_nonrep_keys[i, idx])
                else:
                    feature_key_parts.append(0)
            feature_key = tuple(feature_key_parts)
            
            rep_data = rep_lookup.get(feature_key)
            if rep_data is not None:
                vin_sum = rep_data['vin_sum']
                feat_idx = 0
                X[i, feat_idx] = vin_sum  # reporting_sales
                feat_idx += 1
                
                # Weighted average prices (only if USE_PRICE is True)
                if USE_PRICE and vin_sum > 0:
                    X[i, feat_idx] = rep_data['model_price_weighted'] / vin_sum
                    X[i, feat_idx + 1] = rep_data['segment_price_weighted'] / vin_sum
                    X[i, feat_idx + 2] = rep_data['brand_price_weighted'] / vin_sum
                if USE_PRICE:
                    feat_idx += 3
                
                if has_date_dim and rep_data['count'] > 0:
                    X[i, feat_idx] = rep_data['dow_sum'] / rep_data['count']  # avg day_of_week
                    X[i, feat_idx + 1] = rep_data['dom_sum'] / rep_data['count']  # avg day_of_month
                if has_date_dim:
                    feat_idx += 2
                
                # One-hot weekday proportions (only if USE_1HOT_WEEKDAYS is True)
                if USE_1HOT_WEEKDAYS and vin_sum > 0:
                    for day in range(7):
                        X[i, feat_idx + day] = rep_data[f'dow_{day}'] / vin_sum
            else:
                log_fallback('no_reporting_features', 
                             'No reporting data for bin, using zero features')
        
        # Build group keys (non-temporal dimensions for Option 5)
        temporal_dims = {'date', 'week', 'ALL_TIME'}
        group_dim_indices = [i for i, d in enumerate(dims) if d not in temporal_dims]
        if group_dim_indices:
            group_keys = unique_nonrep_keys[:, group_dim_indices]
        else:
            group_keys = None
        
        return PreparedData(
            X=X,
            feature_names=feature_names,
            y=nonrep_sums.astype(np.float32),
            keys=unique_nonrep_keys,
            dim_names=dims,
            reporting_sum=float(sum(d['vin_sum'] for d in rep_lookup.values())),
            nonreporting_sum=float(nonrep_sums.sum()),
            group_keys=group_keys
        )
    
    train_data = prepare_single(train_table)
    test_data = prepare_single(test_table)
    
    return train_data, test_data


# =============================================================================
# (7) NOWCASTING MODELS
# =============================================================================

class BaseModel:
    """Base class for nowcasting models."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    def fit(self, train_data: PreparedData) -> 'BaseModel':
        raise NotImplementedError
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        raise NotImplementedError


class ZeroModel(BaseModel):
    """Always predicts zero. Baseline reference."""
    
    def __init__(self):
        super().__init__('zero')
    
    def fit(self, train_data: PreparedData) -> 'ZeroModel':
        self.is_fitted = True
        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        return np.zeros(len(test_data.y), dtype=np.float32)


class MeanModel(BaseModel):
    """Always predicts the training mean. Baseline reference."""

    def __init__(self):
        super().__init__('mean')
        self._mean = 0.0

    def fit(self, train_data: PreparedData) -> 'MeanModel':
        self._mean = float(train_data.y.mean()) if len(train_data.y) > 0 else 0.0
        self.is_fitted = True
        return self

    def predict(self, test_data: PreparedData) -> np.ndarray:
        return np.full(len(test_data.y), self._mean, dtype=np.float32)


class LinearExtrapolationModel(BaseModel):
    """
    Linear extrapolation: non_rep = (hist_nonrep / hist_rep) * current_rep
    Learns ratio per group and applies to current reporting sales.
    """
    
    def __init__(self):
        super().__init__('linear_extrapolation')
        self.group_ratios: Dict[tuple, float] = {}
        self.global_ratio = 0.0
    
    def fit(self, train_data: PreparedData) -> 'LinearExtrapolationModel':
        # Global ratio
        total_rep = train_data.X[:, 0].sum()
        total_nonrep = train_data.y.sum()
        self.global_ratio = total_nonrep / total_rep if total_rep > 0 else 0.0
        
        # Per-group ratios
        if train_data.group_keys is not None:
            for i in range(len(train_data.y)):
                group_key = tuple(train_data.group_keys[i])
                rep = train_data.X[i, 0]
                nonrep = train_data.y[i]
                
                if group_key not in self.group_ratios:
                    self.group_ratios[group_key] = {'rep': 0.0, 'nonrep': 0.0}
                self.group_ratios[group_key]['rep'] += rep
                self.group_ratios[group_key]['nonrep'] += nonrep
            
            # Convert to ratios
            for key in self.group_ratios:
                rep = self.group_ratios[key]['rep']
                nonrep = self.group_ratios[key]['nonrep']
                self.group_ratios[key] = nonrep / rep if rep > 0 else 0.0
        
        self.is_fitted = True
        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        predictions = np.zeros(len(test_data.y), dtype=np.float32)
        
        for i in range(len(test_data.y)):
            rep = test_data.X[i, 0]
            
            if test_data.group_keys is not None:
                group_key = tuple(test_data.group_keys[i])
                if group_key in self.group_ratios:
                    ratio = self.group_ratios[group_key]
                else:
                    ratio = self.global_ratio
                    log_fallback('linear_extrap_global_ratio', 
                                 'Used global ratio when group ratio unavailable')
            else:
                ratio = self.global_ratio
            
            predictions[i] = rep * ratio
        
        return predictions


class CrostonSBAModel(BaseModel):
    """Croston's method with Syntetos-Boylan Adjustment."""
    
    def __init__(self, alpha: float = 0.1):
        super().__init__('croston_sba')
        self.alpha = alpha
        self.group_models: Dict[tuple, dict] = {}
        self.global_model: dict = {'demand': 0, 'interval': 1}
    
    def fit(self, train_data: PreparedData) -> 'CrostonSBAModel':
        # Group by non-temporal dimensions
        if train_data.group_keys is not None:
            groups = defaultdict(list)
            for i in range(len(train_data.y)):
                group_key = tuple(train_data.group_keys[i])
                groups[group_key].append(train_data.y[i])
            
            for group_key, values in groups.items():
                self.group_models[group_key] = self._fit_series(np.array(values))
        
        # Global model
        self.global_model = self._fit_series(train_data.y)
        self.is_fitted = True
        return self
    
    def _fit_series(self, y: np.ndarray) -> dict:
        nonzero_idx = np.where(y > 0)[0]
        
        if len(nonzero_idx) < 3:
            log_fallback('croston_insufficient_data', 
                         'Used mean fallback due to <3 non-zero observations')
            return {'demand': y.mean() if len(y) > 0 else 0, 'interval': 1, 'fallback': True}
        
        demands = y[nonzero_idx]
        intervals = np.diff(nonzero_idx)
        intervals = np.concatenate([[nonzero_idx[0] + 1], intervals])
        
        demand_est = demands[0]
        interval_est = max(intervals[0], 1)
        
        for i in range(1, len(demands)):
            demand_est = self.alpha * demands[i] + (1 - self.alpha) * demand_est
            interval_est = self.alpha * intervals[i] + (1 - self.alpha) * interval_est
        
        return {'demand': demand_est, 'interval': interval_est, 'fallback': False}
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        predictions = np.zeros(len(test_data.y), dtype=np.float32)
        
        for i in range(len(test_data.y)):
            if test_data.group_keys is not None:
                group_key = tuple(test_data.group_keys[i])
                if group_key in self.group_models:
                    model = self.group_models[group_key]
                else:
                    model = self.global_model
                    log_fallback('croston_global_model', 
                                 'Used global model when group model unavailable')
            else:
                model = self.global_model
            
            predictions[i] = self._predict_from_model(model)
        
        return predictions
    
    def _predict_from_model(self, model: dict) -> float:
        if model.get('fallback', False):
            log_fallback('croston_fallback_predict', 
                         'Used simple mean prediction from fallback model')
            return model['demand']
        
        if model['interval'] > 0:
            sba_factor = 1 - self.alpha / 2
            return max(0, sba_factor * (model['demand'] / model['interval']))
        return model['demand']


class LASSOModel(BaseModel):
    """LASSO regression with cross-validation."""
    
    def __init__(self):
        super().__init__('lasso')
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, train_data: PreparedData) -> 'LASSOModel':
        X = train_data.X
        y = train_data.y
        
        if len(X) < 2:
            self.is_fitted = False
            return self
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            cv = min(5, len(X))
            self.model = LassoCV(alphas=np.logspace(-4, 1, 50), cv=cv, max_iter=5000, 
                                  positive=True, random_state=42)
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        except Exception as e:
            log_error('lasso_fit', f"LASSO fit failed: {e}", exc=e)
            self.is_fitted = False
        
        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        if not self.is_fitted:
            log_fallback('lasso_not_fitted', 'LASSO model not fitted, returning zeros')
            return np.zeros(len(test_data.y), dtype=np.float32)
        
        X_scaled = self.scaler.transform(test_data.X)
        return np.maximum(0, self.model.predict(X_scaled)).astype(np.float32)


class ElasticNetModel(BaseModel):
    """Elastic Net regression with cross-validation."""
    
    def __init__(self):
        super().__init__('elastic_net')
        self.model = None
        self.scaler = StandardScaler()
    
    def fit(self, train_data: PreparedData) -> 'ElasticNetModel':
        X = train_data.X
        y = train_data.y
        
        if len(X) < 2:
            self.is_fitted = False
            return self
        
        try:
            X_scaled = self.scaler.fit_transform(X)
            cv = min(5, len(X))
            self.model = ElasticNetCV(
                alphas=np.logspace(-4, 1, 50),
                l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95],
                cv=cv, max_iter=5000, positive=True, random_state=42
            )
            self.model.fit(X_scaled, y)
            self.is_fitted = True
        except Exception as e:
            log_error('enet_fit', f"ElasticNet fit failed: {e}", exc=e)
            self.is_fitted = False
        
        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        if not self.is_fitted:
            log_fallback('elastic_net_not_fitted', 'ElasticNet model not fitted, returning zeros')
            return np.zeros(len(test_data.y), dtype=np.float32)
        
        X_scaled = self.scaler.transform(test_data.X)
        return np.maximum(0, self.model.predict(X_scaled)).astype(np.float32)


class RandomForestModel(BaseModel):
    """Random Forest for nowcasting."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 8):
        super().__init__('random_forest')
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                           min_samples_leaf=5, n_jobs=-1, random_state=42)
    
    def fit(self, train_data: PreparedData) -> 'RandomForestModel':
        if len(train_data.X) < 2:
            self.is_fitted = False
            return self
        
        try:
            self.model.fit(train_data.X, train_data.y)
            self.is_fitted = True
        except Exception as e:
            log_error('rf_fit', f"RandomForest fit failed: {e}", exc=e)
            self.is_fitted = False
        
        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        if not self.is_fitted:
            log_fallback('random_forest_not_fitted', 'RandomForest model not fitted, returning zeros')
            return np.zeros(len(test_data.y), dtype=np.float32)
        return np.maximum(0, self.model.predict(test_data.X)).astype(np.float32)


class LightGBMModel(BaseModel):
    """LightGBM gradient boosting with Poisson objective for count data."""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 6):
        super().__init__('lightgbm')
        self.model = lgb.LGBMRegressor(
            n_estimators=n_estimators, 
            max_depth=max_depth,
            objective='poisson',
            learning_rate=0.05,
            min_child_samples=10,
            verbosity=-1, 
            random_state=42
        )
    
    def fit(self, train_data: PreparedData) -> 'LightGBMModel':
        if len(train_data.X) < 2:
            self.is_fitted = False
            return self
        
        try:
            self.model.fit(train_data.X, train_data.y)
            self.is_fitted = True
        except Exception as e:
            log_error('lgbm_fit', f"LightGBM fit failed: {e}", exc=e)
            self.is_fitted = False
        
        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        if not self.is_fitted:
            log_fallback('lightgbm_not_fitted', 'LightGBM model not fitted, returning zeros')
            return np.zeros(len(test_data.y), dtype=np.float32)
        return np.maximum(0, self.model.predict(test_data.X)).astype(np.float32)


class CatBoostModel(BaseModel):
    """CatBoost gradient boosting with Poisson loss for count data."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        super().__init__('catboost')
        self.model = cb.CatBoostRegressor(
            iterations=n_estimators, 
            depth=max_depth,
            loss_function='Poisson',
            verbose=False, 
            random_state=42
        )
    
    def fit(self, train_data: PreparedData) -> 'CatBoostModel':
        if len(train_data.X) < 2:
            self.is_fitted = False
            return self
        
        # Check for constant features (CatBoost fails on these)
        if train_data.X.shape[1] == 1 and np.std(train_data.X[:, 0]) < 1e-10:
            self.is_fitted = False
            return self
        
        try:
            self.model.fit(train_data.X, train_data.y)
            self.is_fitted = True
        except Exception as e:
            log_error('catboost_fit', f"CatBoost fit failed: {e}", exc=e)
            self.is_fitted = False
        
        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        if not self.is_fitted:
            log_fallback('catboost_not_fitted', 'CatBoost model not fitted, returning zeros')
            return np.zeros(len(test_data.y), dtype=np.float32)
        return np.maximum(0, self.model.predict(test_data.X)).astype(np.float32)


class XGBoostModel(BaseModel):
    """XGBoost gradient boosting with Poisson objective for count data."""
    
    def __init__(self, n_estimators: int = 100, max_depth: int = 6):
        super().__init__('xgboost')
        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            objective='count:poisson',
            learning_rate=0.1,
            min_child_weight=5,
            reg_alpha=0.1,
            reg_lambda=1.0,
            verbosity=0,
            random_state=42
        )
        self._y_scale = 1.0
    
    def fit(self, train_data: PreparedData) -> 'XGBoostModel':
        if len(train_data.X) < 2:
            self.is_fitted = False
            return self

        try:
            X = train_data.X.copy()
            y = train_data.y.copy()

            # Sanitize: replace inf/nan with 0
            X = np.where(np.isfinite(X), X, 0.0)
            y = np.where(np.isfinite(y), y, 0.0)

            # Poisson requires y > 0 on average (base_score = log(mean(y)))
            if y.sum() == 0:
                log_fallback('xgb_zero_target', 'All targets zero, skipping XGBoost fit')
                self.is_fitted = False
                return self

            # Scale y to ~1.0 range to prevent exp() overflow in Poisson gradient
            self._y_scale = float(y.mean())
            if self._y_scale > 1.0:
                y = y / self._y_scale

            self.model.fit(X, y)
            self.is_fitted = True
        except Exception as e:
            log_error('xgb_fit', f"XGBoost fit failed: {e}", exc=e)
            self.is_fitted = False

        return self
    
    def predict(self, test_data: PreparedData) -> np.ndarray:
        if not self.is_fitted:
            log_fallback('xgboost_not_fitted', 'XGBoost model not fitted, returning zeros')
            return np.zeros(len(test_data.y), dtype=np.float32)
        preds = np.maximum(0, self.model.predict(test_data.X)).astype(np.float32)
        return preds * self._y_scale


MODEL_REGISTRY = {
    'zero': ZeroModel,
    'mean': MeanModel,
    'linear_extrapolation': LinearExtrapolationModel,
    'croston_sba': CrostonSBAModel,
    'lasso': LASSOModel,
    'elastic_net': ElasticNetModel,
    'random_forest': RandomForestModel,
    'lightgbm': LightGBMModel,
    'catboost': CatBoostModel,
    'xgboost': XGBoostModel
}


# =============================================================================
# (8) NOWCAST RESULT
# =============================================================================

@dataclass
class NowcastResult:
    """Result of nowcasting for one level-model combination."""
    level: str
    model_name: str
    
    # Predictions and actuals as LevelData
    predictions: Optional[LevelData] = None
    
    # Training data for status tracking (keys and values)
    train_keys: Optional[np.ndarray] = None  # (N, D) bin keys from training
    train_values: Optional[np.ndarray] = None  # (N,) actual values from training
    train_dim_names: Optional[List[str]] = None
    
    # Metrics
    mae: float = 0.0
    calibrated_error: float = 0.0
    
    # Car counts
    train_reporting_cars: float = 0.0
    train_nonreporting_cars: float = 0.0
    infer_reporting_cars: float = 0.0
    infer_nonreporting_cars: float = 0.0
    actual_nonreporting_cars: float = 0.0
    
    # Bin counts
    train_bins: int = 0
    infer_bins: int = 0
    new_bins_count: int = 0
    new_bins_actual_cars: float = 0.0
    known_bins_count: int = 0
    
    # Timing
    train_time: float = 0.0
    predict_time: float = 0.0


def train_and_predict(train_data: PreparedData, test_data: PreparedData,
                      level_name: str, model_name: str) -> NowcastResult:
    """Train model and generate predictions."""
    result = NowcastResult(level=level_name, model_name=model_name)
    
    # Store training stats
    result.train_bins = len(train_data.y)
    result.train_reporting_cars = train_data.reporting_sum
    result.train_nonreporting_cars = train_data.nonreporting_sum
    result.infer_bins = len(test_data.y)
    result.infer_reporting_cars = test_data.reporting_sum
    result.actual_nonreporting_cars = test_data.y.sum()
    
    # Store training keys and values for status tracking
    result.train_keys = train_data.keys.copy() if train_data.keys is not None else None
    result.train_values = train_data.y.copy() if train_data.y is not None else None
    result.train_dim_names = train_data.dim_names
    
    if len(train_data.y) == 0:
        log_warning(f'no_train_{level_name}_{model_name}', f"No training data for {level_name}/{model_name}")
        return result
    
    # Identify new bins (groups not seen in training)
    if train_data.group_keys is not None and test_data.group_keys is not None:
        train_groups = set(tuple(k) for k in train_data.group_keys)
        new_bin_mask = np.array([tuple(k) not in train_groups for k in test_data.group_keys])
        result.new_bins_count = new_bin_mask.sum()
        result.new_bins_actual_cars = test_data.y[new_bin_mask].sum()
        result.known_bins_count = (~new_bin_mask).sum()
    else:
        new_bin_mask = np.zeros(len(test_data.y), dtype=bool)
        result.known_bins_count = len(test_data.y)
    
    # Get model
    if model_name not in MODEL_REGISTRY:
        log_error(f'unknown_model_{model_name}', f"Unknown model: {model_name}")
        return result
    
    model = MODEL_REGISTRY[model_name]()
    
    # Train
    t0 = time.time()
    try:
        model.fit(train_data)
    except Exception as e:
        log_error(f'{model_name}_fit', f"{model_name} fit failed: {e}", exc=e)
        return result
    result.train_time = time.time() - t0
    
    # Predict
    t0 = time.time()
    try:
        predictions = model.predict(test_data)
    except Exception as e:
        log_error(f'{model_name}_predict', f"{model_name} predict failed: {e}", exc=e)
        return result
    result.predict_time = time.time() - t0
    
    # Zero out new bins (Option 5)
    predictions = np.where(new_bin_mask, 0.0, predictions)
    
    # Store as LevelData
    result.predictions = LevelData(
        level=level_name,
        dim_names=test_data.dim_names,
        keys=test_data.keys,
        values=predictions.astype(np.float32),
        actuals=test_data.y
    )
    
    # Calculate metrics
    result.mae = np.mean(np.abs(predictions - test_data.y))
    result.infer_nonreporting_cars = predictions.sum()
    
    return result


# =============================================================================
# (9) RECONCILIATION ENGINE
# =============================================================================

@dataclass
class ReconciliationResult:
    """Result of reconciliation between two levels."""
    level_from: str
    level_to: str
    method: str
    
    # Original and reconciled data
    original_detail: Optional[LevelData] = None
    original_agg: Optional[LevelData] = None
    reconciled_detail: Optional[LevelData] = None
    reconciled_agg: Optional[LevelData] = None
    
    # Error metrics
    error_detail: float = 0.0  # MAE at detail level after reconciliation
    error_agg: float = 0.0  # MAE at agg level after reconciliation
    compute_time: float = 0.0


class ReconciliationEngine:
    """Handles reconciliation between hierarchy levels."""
    
    def __init__(self, dim_registry: DimensionRegistry, level_registry: LevelRegistry):
        self.dim_registry = dim_registry
        self.level_registry = level_registry
    
    def bottom_up(self, child_data: LevelData, parent_level: str) -> LevelData:
        """Aggregate child predictions to parent level."""
        parent_dims = self.level_registry.get_dims(parent_level)
        return child_data.aggregate_to(parent_level, parent_dims, self.dim_registry)
    
    def _align_actuals(self, target_data: LevelData, source_data: LevelData) -> LevelData:
        """
        Align actuals from source_data to target_data based on matching keys.
        Returns target_data with actuals filled in where keys match.
        """
        if source_data.actuals is None:
            return target_data
        
        # Build source key -> actual lookup
        source_data.build_index()
        
        # Create aligned actuals
        aligned_actuals = np.zeros(len(target_data.values), dtype=np.float32)
        
        for i, key in enumerate(target_data.keys):
            key_tuple = tuple(key)
            if key_tuple in source_data._key_index:
                src_idx = source_data._key_index[key_tuple]
                aligned_actuals[i] = source_data.actuals[src_idx]
        
        target_data.actuals = aligned_actuals
        return target_data
    
    def top_down(self, parent_data: LevelData, child_data: LevelData) -> LevelData:
        """Distribute parent predictions to child level using child as reference."""
        return child_data.disaggregate_from(parent_data, self.dim_registry)
    
    def mint(self, child_data: LevelData, parent_data: LevelData,
             child_level: str, parent_level: str) -> Tuple[LevelData, LevelData]:
        """
        MinT reconciliation: per-group blend of child aggregate and parent prediction.
        
        For each parent group:
        - child_sum = sum of child predictions that aggregate to this parent
        - parent_pred = parent's prediction for this group
        - weight_child = n_children (more children = more reliable aggregate)
        - weight_parent = 1 (single parent prediction)
        - reconciled = weighted average, then distribute back to children
        """
        parent_dims = self.level_registry.get_dims(parent_level)
        
        # Map child keys to parent keys
        n_child = len(child_data.keys)
        mapped_parent_keys = np.zeros((n_child, len(parent_dims)), dtype=np.int32)
        
        for i, parent_dim in enumerate(parent_dims):
            if parent_dim in child_data.dim_names:
                src_idx = child_data.dim_names.index(parent_dim)
                mapped_parent_keys[:, i] = child_data.keys[:, src_idx]
            else:
                for j, src_dim in enumerate(child_data.dim_names):
                    if self.dim_registry.has(src_dim):
                        dim_table = self.dim_registry.get(src_dim)
                        if parent_dim in dim_table.parents:
                            mapped_parent_keys[:, i] = dim_table.map_to_parent(parent_dim, child_data.keys[:, j])
                            break
                else:
                    mapped_parent_keys[:, i] = 0
        
        # Build parent index
        parent_data.build_index()
        
        # Group children by parent key
        unique_parent_keys, inverse = np.unique(mapped_parent_keys, axis=0, return_inverse=True)
        
        # Reconcile per group
        reconciled_child_values = child_data.values.copy()
        reconciled_parent_values = np.zeros(len(unique_parent_keys), dtype=np.float32)
        reconciled_parent_keys = unique_parent_keys
        
        for parent_idx, parent_key in enumerate(unique_parent_keys):
            child_mask = (inverse == parent_idx)
            child_vals = child_data.values[child_mask]
            child_sum = child_vals.sum()
            n_children = child_mask.sum()
            
            # Get parent prediction for this group
            parent_pred = parent_data.get_value(tuple(parent_key))
            
            if child_sum == 0 and parent_pred == 0:
                reconciled_parent_values[parent_idx] = 0
                continue
            
            # MinT weights: proportional to information (# of observations)
            # More children = more reliable child aggregate
            w_child = float(n_children)
            w_parent = 1.0  # Single parent prediction
            total_w = w_child + w_parent
            
            # Weighted average for reconciled total
            if total_w > 0:
                reconciled_total = (w_child * child_sum + w_parent * parent_pred) / total_w
            else:
                reconciled_total = (child_sum + parent_pred) / 2
            
            reconciled_parent_values[parent_idx] = reconciled_total
            
            # Distribute to children proportionally
            if child_sum > 0:
                scale = reconciled_total / child_sum
                reconciled_child_values[child_mask] = child_vals * scale
            elif n_children > 0:
                # Equal distribution if no child reference
                log_fallback('mint_equal_distribution', 
                             'MinT: child_sum=0, distributing equally')
                reconciled_child_values[child_mask] = reconciled_total / n_children
        
        # Build reconciled LevelData objects
        reconciled_child = LevelData(
            level=child_data.level,
            dim_names=child_data.dim_names,
            keys=child_data.keys.copy(),
            values=reconciled_child_values.astype(np.float32),
            actuals=child_data.actuals
        )
        
        reconciled_parent = LevelData(
            level=parent_level,
            dim_names=parent_dims,
            keys=reconciled_parent_keys,
            values=reconciled_parent_values.astype(np.float32)
        )
        reconciled_parent = self._align_actuals(reconciled_parent, parent_data)
        
        return reconciled_child, reconciled_parent
    
    def wls(self, child_data: LevelData, parent_data: LevelData,
            child_level: str, parent_level: str) -> Tuple[LevelData, LevelData]:
        """
        Weighted Least Squares reconciliation: per-group blend using variance-based weights.
        
        For each parent group:
        - child_sum = sum of child predictions
        - parent_pred = parent's prediction
        - weight_child = 1 / var(child) proxy = n_children / child_sum (higher density = lower var)
        - weight_parent = 1 / var(parent) proxy = 1 / parent_pred (single prediction)
        - reconciled = weighted average, then distribute back to children
        """
        parent_dims = self.level_registry.get_dims(parent_level)
        
        # Map child keys to parent keys
        n_child = len(child_data.keys)
        mapped_parent_keys = np.zeros((n_child, len(parent_dims)), dtype=np.int32)
        
        for i, parent_dim in enumerate(parent_dims):
            if parent_dim in child_data.dim_names:
                src_idx = child_data.dim_names.index(parent_dim)
                mapped_parent_keys[:, i] = child_data.keys[:, src_idx]
            else:
                for j, src_dim in enumerate(child_data.dim_names):
                    if self.dim_registry.has(src_dim):
                        dim_table = self.dim_registry.get(src_dim)
                        if parent_dim in dim_table.parents:
                            mapped_parent_keys[:, i] = dim_table.map_to_parent(parent_dim, child_data.keys[:, j])
                            break
                else:
                    mapped_parent_keys[:, i] = 0
        
        # Build parent index
        parent_data.build_index()
        
        # Group children by parent key
        unique_parent_keys, inverse = np.unique(mapped_parent_keys, axis=0, return_inverse=True)
        
        # Reconcile per group
        reconciled_child_values = child_data.values.copy()
        reconciled_parent_values = np.zeros(len(unique_parent_keys), dtype=np.float32)
        reconciled_parent_keys = unique_parent_keys
        
        for parent_idx, parent_key in enumerate(unique_parent_keys):
            child_mask = (inverse == parent_idx)
            child_vals = child_data.values[child_mask]
            child_sum = child_vals.sum()
            n_children = child_mask.sum()
            
            # Get parent prediction for this group
            parent_pred = parent_data.get_value(tuple(parent_key))
            
            if child_sum == 0 and parent_pred == 0:
                reconciled_parent_values[parent_idx] = 0
                continue
            
            # WLS weights: inverse variance proxy
            # For child aggregate: var ~ sum/n, so weight ~ n/sum
            # For parent: single point, weight ~ 1/value (or use n_children as proxy)
            
            # Use coefficient of variation as variance proxy
            # Higher mean with more observations = lower relative variance = higher weight
            child_mean = child_sum / n_children if n_children > 0 else 0
            
            # Weight = precision = 1/variance
            # Variance proxy: (value + 1) / (n + 1) to avoid div by zero and handle zeros
            w_child = (n_children + 1) / (child_sum + 1)
            w_parent = (1 + 1) / (parent_pred + 1)  # Single observation
            
            total_w = w_child + w_parent
            
            # Weighted average for reconciled total
            if total_w > 0:
                reconciled_total = (w_child * child_sum + w_parent * parent_pred) / total_w
            else:
                reconciled_total = (child_sum + parent_pred) / 2
            
            reconciled_parent_values[parent_idx] = reconciled_total
            
            # Distribute to children proportionally
            if child_sum > 0:
                scale = reconciled_total / child_sum
                reconciled_child_values[child_mask] = child_vals * scale
            elif n_children > 0:
                log_fallback('wls_equal_distribution', 
                             'WLS: child_sum=0, distributing equally')
                reconciled_child_values[child_mask] = reconciled_total / n_children
        
        # Build reconciled LevelData objects
        reconciled_child = LevelData(
            level=child_data.level,
            dim_names=child_data.dim_names,
            keys=child_data.keys.copy(),
            values=reconciled_child_values.astype(np.float32),
            actuals=child_data.actuals
        )
        
        reconciled_parent = LevelData(
            level=parent_level,
            dim_names=parent_dims,
            keys=reconciled_parent_keys,
            values=reconciled_parent_values.astype(np.float32)
        )
        reconciled_parent = self._align_actuals(reconciled_parent, parent_data)
        
        return reconciled_child, reconciled_parent
    
    def middle_out(self, child_data: LevelData, parent_data: LevelData,
                   child_level: str, parent_level: str) -> Tuple[LevelData, LevelData]:
        """
        Middle-out reconciliation: blend of bottom-up and top-down.
        Uses bottom-up for aggregation, preserving detail structure.
        """
        # Essentially same as bottom-up for two-level case
        return self.bottom_up_pair(child_data, parent_data, parent_level)
    
    def bottom_up_pair(self, child_data: LevelData, parent_data: LevelData,
                       parent_level: str) -> Tuple[LevelData, LevelData]:
        """Bottom-up returning both levels."""
        reconciled_parent = self.bottom_up(child_data, parent_level)
        reconciled_parent = self._align_actuals(reconciled_parent, parent_data)
        return child_data, reconciled_parent
    
    def reconcile(self, child_data: LevelData, parent_data: LevelData,
                  method: str) -> ReconciliationResult:
        """
        Reconcile predictions between two levels.
        """
        result = ReconciliationResult(
            level_from=child_data.level,
            level_to=parent_data.level,
            method=method
        )
        result.original_detail = child_data
        result.original_agg = parent_data
        
        t0 = time.time()
        
        parent_level = parent_data.level
        child_level = child_data.level
        
        if method == 'bottom_up':
            result.reconciled_detail = child_data
            result.reconciled_agg = self.bottom_up(child_data, parent_level)
            # Copy actuals from parent_data if keys align
            if parent_data.actuals is not None:
                result.reconciled_agg = self._align_actuals(result.reconciled_agg, parent_data)
            
        elif method == 'top_down':
            result.reconciled_agg = parent_data
            result.reconciled_detail = self.top_down(parent_data, child_data)
            # Copy actuals from child_data if keys align
            if child_data.actuals is not None:
                result.reconciled_detail = self._align_actuals(result.reconciled_detail, child_data)
            
        elif method == 'mint':
            result.reconciled_detail, result.reconciled_agg = self.mint(
                child_data, parent_data, child_level, parent_level)
            
        elif method == 'wls':
            result.reconciled_detail, result.reconciled_agg = self.wls(
                child_data, parent_data, child_level, parent_level)
        
        elif method == 'middle_out':
            result.reconciled_detail, result.reconciled_agg = self.middle_out(
                child_data, parent_data, child_level, parent_level)
        else:
            log_warning(f'unknown_recon_{method}', f"Unknown reconciliation method: {method}")
            result.reconciled_detail = child_data
            result.reconciled_agg = parent_data
        
        result.compute_time = time.time() - t0
        
        # Calculate errors
        if result.reconciled_detail.actuals is not None:
            result.error_detail = result.reconciled_detail.compute_mae()
        if result.reconciled_agg.actuals is not None:
            result.error_agg = result.reconciled_agg.compute_mae()
        
        return result


# =============================================================================
# (10) CONFIGURATION
# =============================================================================

# Two-cutoff approach:
# - cutoff_train (t): Training simulation time - learn from historically missing data
# - cutoff_test (T): Test simulation time - predict currently missing data
force_min_date = "2025-02-01"
force_max_date = "2025-12-01"
CUTOFF_TRAIN = pd.Timestamp('2025-06-01')  # t
CUTOFF_TEST = pd.Timestamp('2025-09-01')   # T
DATA_FILE = 'tmp_20260205.csv'

# Feature toggle: whether to use price features (model_price, segment_price, brand_price)
USE_PRICE = False

# Feature toggle: whether to use one-hot encoded weekday proportions (7 features)
USE_1HOT_WEEKDAYS = True

ALL_MODELS = ['mean', 'linear_extrapolation', 'croston_sba', 'lasso', 'elastic_net',
              'random_forest', 'lightgbm', 'catboost', 'xgboost']

RECONCILIATION_METHODS = ['bottom_up', 'top_down', 'mint', 'wls']

INCLUDE_DETAILED_LEVELS = True
# Define levels
LEVELS = [
    LevelSpec('micro', ['model_key_short', 'tract', 'date'], 'sparse'),
    LevelSpec('model_county_day', ['model_key_short', 'county', 'date'], 'sparse'), ] if INCLUDE_DETAILED_LEVELS else []
LEVELS = LEVELS + [
    LevelSpec('brand_county_day', ['brand','county', 'date'], 'dense'),
    LevelSpec('brand_county_week', ['brand','county', 'week'], 'dense'),
    LevelSpec('brand_state_week', ['brand', 'state', 'week'], 'dense'),
    LevelSpec('segment_state_day', ['segment', 'state', 'date'], 'moderate'),
    LevelSpec('brand_state_all', ['brand', 'state', 'ALL_TIME'], 'dense'),
    LevelSpec('state_all', ['ALL_BRANDS', 'state', 'ALL_TIME'], 'dense'),
    #LevelSpec('macro', ['ALL_BRANDS', 'ALL_STATES', 'ALL_TIME'], 'dense')
]

# Define reconciliation pairs (child -> parent)
RECON_PAIRS = [
    ('micro', 'model_county_day'),
    ('model_county_day', 'brand_county_day'), ] if INCLUDE_DETAILED_LEVELS else []
RECON_PAIRS = RECON_PAIRS + [
    ('brand_county_day', 'brand_county_week'),
    ('brand_county_week', 'brand_state_week'),
    ('brand_state_week', 'brand_state_all'),
    ('brand_state_all', 'state_all'),
    #('state_all', 'macro')
]


# =============================================================================
# (11) MAIN EXECUTION
# =============================================================================

def main():
    """Main entry point."""
    print("=" * 70)
    print("US NEW CAR MARKET NOWCASTING SYSTEM")
    print("Columnar Storage Architecture - Two-Cutoff Evaluation")
    print("=" * 70)
    
    # =========================================================================
    # INITIALIZE REGISTRIES
    # =========================================================================
    
    dim_registry = DimensionRegistry()
    level_registry = LevelRegistry()
    
    for level in LEVELS:
        level_registry.add(level)
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    # Load and encode
    fact_table, min_date, max_date = load_and_encode_data(DATA_FILE, dim_registry, force_min_date=force_min_date, force_max_date=force_max_date)

    if fact_table is None:
        print_error_summary()
        return
    
    # Store dim_registry in debug for inspection
    debug.dim_registry = dim_registry
    debug.level_registry = level_registry
    
    print_progress(f"Date range: {min_date} to {max_date}")
    print_progress(f"Training cutoff (t): {CUTOFF_TRAIN}")
    print_progress(f"Test cutoff (T): {CUTOFF_TEST}")
    
    # =========================================================================
    # SPLIT DATA (Two-Cutoff Approach)
    # =========================================================================
    
    train_table, test_table = split_data_two_cutoff(fact_table, CUTOFF_TRAIN, CUTOFF_TEST)
    
    # =========================================================================
    # NOWCASTING FOR ALL LEVELS
    # =========================================================================
    
    print_progress("\nStarting nowcasting for all levels...")
    # Build feature list message based on toggles
    features_msg = "Features: reporting_sales"
    if USE_PRICE:
        features_msg += ", model_price, segment_price, brand_price"
    features_msg += " (+ day_of_week, day_of_month for date levels)"
    if USE_1HOT_WEEKDAYS:
        features_msg += " (+ dow_mon..dow_sun one-hot proportions)"
    print_progress(features_msg)
    
    all_results: Dict[str, Dict[str, NowcastResult]] = {}  # level -> model -> result
    
    for level_spec in LEVELS:
        level_name = level_spec.name
        print_progress(f"\nLevel: {level_name} (sparsity: {level_spec.sparsity})")
        
        # Prepare data for this level
        train_data, test_data = prepare_level_data(train_table, test_table, level_spec, dim_registry)
        
        all_results[level_name] = {}
        
        # Always run zero model
        models_to_run = ['zero'] + ALL_MODELS
        
        for model_name in models_to_run:
            print_progress(f"  Training {model_name}...")
            result = train_and_predict(train_data, test_data, level_name, model_name)
            all_results[level_name][model_name] = result
    
    # =========================================================================
    # MODEL ASSESSMENT
    # =========================================================================
    
    print("\n" + "-" * 70)
    print("MODEL ASSESSMENT RESULTS (Calibrated Error % - lower is better)")
    print("-" * 70)
    
    best_models = {}
    
    for level_name in level_registry.level_order:
        level_results = all_results.get(level_name, {})
        
        # Get zero MAE for calibration
        zero_result = level_results.get('zero')
        zero_mae = zero_result.mae if zero_result else 1.0
        
        # Calculate calibrated errors
        for model_name, result in level_results.items():
            if zero_mae > 0:
                result.calibrated_error = (result.mae / zero_mae) * 100.0
        
        # Find best model (excluding zero)
        best_model = None
        best_error = float('inf')
        
        for model_name, result in level_results.items():
            if model_name != 'zero' and result.predictions is not None:
                if result.calibrated_error < best_error:
                    best_error = result.calibrated_error
                    best_model = model_name
        
        if best_model:
            best_models[level_name] = best_model
        
        # Print table
        new_bins_info = ""
        for result in level_results.values():
            if result.new_bins_count > 0:
                new_bins_info = f" | new_bins: {result.new_bins_count} ({result.new_bins_actual_cars:.0f} cars missed)"
                break
        
        print(f"\n{level_name}:{new_bins_info}")
        print(f"  {'Model':<20} {'Error %':>10} {'MAE':>10} {'Corr':>10}")
        print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
        
        sorted_results = sorted(level_results.items(), key=lambda x: x[1].calibrated_error)
        
        for model_name, result in sorted_results:
            if result.predictions is not None:
                corr = result.predictions.compute_correlation()
                corr_str = f"{corr:.3f}" if corr is not None else "N/A"
                print(f"  {model_name:<20} {result.calibrated_error:>10.3f} {result.mae:>10.4f} {corr_str:>10}")
            else:
                print(f"  {model_name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10}")
        
        if best_model:
            print(f"  --> Best: {best_model} ({best_error:.3f}%)")
    
    debug.best_models = best_models
    
    # =========================================================================
    # REGISTER MODEL RESULTS
    # =========================================================================
    
    for level_name in level_registry.level_order:
        level_results = all_results.get(level_name, {})
        
        # Register zero MAE first
        zero_result = level_results.get('zero')
        if zero_result and zero_result.predictions and zero_result.predictions.actuals is not None:
            zero_mae = float(np.mean(np.abs(zero_result.predictions.values - zero_result.predictions.actuals)))
            results_registry.register_zero_mae(level_name, zero_mae)
        
        # Register all model results
        for model_name, result in level_results.items():
            if result.predictions is not None:
                results_registry.register(
                    level=level_name,
                    origin=model_name,
                    predictions=result.predictions,
                    cutoff_train=CUTOFF_TRAIN,
                    cutoff_test=CUTOFF_TEST
                )
    
    # =========================================================================
    # HIERARCHICAL RECONCILIATION
    # =========================================================================
    
    print_progress("\nRunning hierarchical reconciliation...")
    
    recon_engine = ReconciliationEngine(dim_registry, level_registry)
    recon_results = []
    
    for child_level, parent_level in RECON_PAIRS:
        # Get best model predictions for each level
        child_best_model = best_models.get(child_level)
        parent_best_model = best_models.get(parent_level)
        
        if not child_best_model or not parent_best_model:
            log_warning(f'no_best_{child_level}_{parent_level}', 
                       f"No best model for {child_level} or {parent_level}")
            continue
        
        child_result = all_results[child_level].get(child_best_model)
        parent_result = all_results[parent_level].get(parent_best_model)
        
        if not child_result or not child_result.predictions:
            continue
        if not parent_result or not parent_result.predictions:
            continue
        
        child_data = child_result.predictions
        parent_data = parent_result.predictions
        
        for method in RECONCILIATION_METHODS:
            rec_result = recon_engine.reconcile(child_data, parent_data, method)
            recon_results.append(rec_result)
            
            # Register reconciliation results
            # Origin format: {method}_{det_model}_{agg_model}_{det|agg}
            # Skip bottom_up detail (unchanged from original)
            # Skip top_down agg (unchanged from original)
            
            if method != 'bottom_up' and rec_result.reconciled_detail is not None:
                origin_det = f"{method}_{child_best_model}_{parent_best_model}_det"
                results_registry.register(
                    level=child_level,
                    origin=origin_det,
                    predictions=rec_result.reconciled_detail,
                    cutoff_train=CUTOFF_TRAIN,
                    cutoff_test=CUTOFF_TEST
                )
            
            if method != 'top_down' and rec_result.reconciled_agg is not None:
                origin_agg = f"{method}_{child_best_model}_{parent_best_model}_agg"
                results_registry.register(
                    level=parent_level,
                    origin=origin_agg,
                    predictions=rec_result.reconciled_agg,
                    cutoff_train=CUTOFF_TRAIN,
                    cutoff_test=CUTOFF_TEST
                )
    
    debug.reconciliation_results = recon_results
    
    # =========================================================================
    # RECONCILIATION ASSESSMENT
    # =========================================================================
    
    print("\n" + "-" * 70)
    print("RECONCILIATION ASSESSMENT")
    print("-" * 70)
    
    pair_w = max(len("From -> To"), max(len(f"{r.level_from} -> {r.level_to}") for r in recon_results))
    meth_w = max(len("Method"), max(len(r.method) for r in recon_results))
    det_w = max(len("Error Detail"), 12)
    agg_w = max(len("Error Agg"), 12)
    print(f"\n{'From -> To':<{pair_w}} {'Method':<{meth_w}} {'Error Detail':>{det_w}} {'Error Agg':>{agg_w}}")
    print(f"{'-'*pair_w} {'-'*meth_w} {'-'*det_w} {'-'*agg_w}")

    for rec in recon_results:
        pair = f"{rec.level_from} -> {rec.level_to}"
        print(f"{pair:<{pair_w}} {rec.method:<{meth_w}} {rec.error_detail:>{det_w}.4f} {rec.error_agg:>{agg_w}.4f}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    print("\nBest nowcasting models by level:")
    for level, model in best_models.items():
        print(f"  {level}: {model}")
    
    print("\nPerformance (calibrated error %):")
    for level_name in level_registry.level_order:
        best_model = best_models.get(level_name)
        if best_model:
            result = all_results[level_name].get(best_model)
            if result:
                print(f"  {level_name}: {result.calibrated_error:.1f}% (MAE: {result.mae:.4f})")
    
    # Store results in debug
    debug.model_results = all_results
    
    print_error_summary()
    print("\nSystem ready. Use 'debug' object for inspection.")


# =============================================================================
# (11) INSPECTION UTILITIES
# =============================================================================

def inspect_model(level_name: str, model_name: str = None):
    """Inspect model results for a level."""
    if level_name not in debug.model_results:
        print(f"No results for level: {level_name}")
        return
    
    level_results = debug.model_results[level_name]
    
    if model_name is None:
        model_name = debug.best_models.get(level_name, 'linear_extrapolation')
    
    result = level_results.get(model_name)
    if not result:
        # Fallback: check results_registry (holds reconciled/aggregated origins)
        reg_entry = results_registry.get_level_entries(level_name).get(model_name)
        if not reg_entry:
            print(f"No results for {model_name} at {level_name}")
            return

        print(f"\n{'='*60}")
        print(f"MODEL INSPECTION (from registry): {model_name} at {level_name}")
        print(f"{'='*60}")
        print(f"\nCutoffs: {reg_entry.cutoff_train.strftime('%Y-%m-%d')} / {reg_entry.cutoff_test.strftime('%Y-%m-%d')}")
        print(f"\nMetrics:")
        print(f"  MAE: {reg_entry.mae:.4f}")
        print(f"  MAE % vs zero: {reg_entry.mae_pct_vs_zero:.2f}%")

        if reg_entry.predictions:
            corr = reg_entry.predictions.compute_correlation()
            if corr is not None:
                print(f"  Correlation: {corr:.4f}")

            print(f"\nPrediction statistics:")
            preds = reg_entry.predictions.values
            actuals = reg_entry.predictions.actuals
            print(f"  Predictions - Sum: {preds.sum():.0f}, Mean: {preds.mean():.2f}, Std: {preds.std():.2f}")
            if actuals is not None:
                print(f"  Actuals     - Sum: {actuals.sum():.0f}, Mean: {actuals.mean():.2f}, Std: {actuals.std():.2f}")
                diff = preds - actuals
                print(f"  Diff        - Sum: {diff.sum():.0f}, Mean: {diff.mean():.2f}, Std: {diff.std():.2f}")
        return

    print(f"\n{'='*60}")
    print(f"MODEL INSPECTION: {model_name} at {level_name}")
    print(f"{'='*60}")

    print(f"\nTraining:")
    print(f"  Bins: {result.train_bins}")
    print(f"  Reporting cars: {result.train_reporting_cars:,.0f}")
    print(f"  Non-reporting cars: {result.train_nonreporting_cars:,.0f}")

    print(f"\nInference:")
    print(f"  Bins: {result.infer_bins}")
    print(f"  Reporting cars (features): {result.infer_reporting_cars:,.0f}")
    print(f"  Actual non-reporting: {result.actual_nonreporting_cars:,.0f}")
    print(f"  Predicted non-reporting: {result.infer_nonreporting_cars:,.0f}")

    if result.new_bins_count > 0:
        print(f"\nNew bins (not in training):")
        print(f"  Count: {result.new_bins_count}")
        print(f"  Missed cars: {result.new_bins_actual_cars:,.0f}")
        print(f"  Known bins: {result.known_bins_count}")

    print(f"\nMetrics:")
    print(f"  MAE: {result.mae:.4f}")
    print(f"  Calibrated error: {result.calibrated_error:.1f}%")

    if result.predictions:
        corr = result.predictions.compute_correlation()
        if corr is not None:
            print(f"  Correlation: {corr:.4f}")

        print(f"\nPrediction statistics:")
        preds = result.predictions.values
        actuals = result.predictions.actuals
        print(f"  Predictions - Sum: {preds.sum():.0f}, Mean: {preds.mean():.2f}, Std: {preds.std():.2f}")
        if actuals is not None:
            print(f"  Actuals     - Sum: {actuals.sum():.0f}, Mean: {actuals.mean():.2f}, Std: {actuals.std():.2f}")


def inspect_reconciliation(child_level: str, parent_level: str, method: str):
    """Inspect reconciliation results."""
    for rec in debug.reconciliation_results:
        if rec.level_from == child_level and rec.level_to == parent_level and rec.method == method:
            print(f"\n{'='*60}")
            print(f"RECONCILIATION: {child_level} -> {parent_level} ({method})")
            print(f"{'='*60}")
            
            print(f"\nDetail level ({child_level}):")
            if rec.original_detail:
                print(f"  Original bins: {len(rec.original_detail)}")
                print(f"  Original sum: {rec.original_detail.values.sum():.0f}")
            if rec.reconciled_detail:
                print(f"  Reconciled sum: {rec.reconciled_detail.values.sum():.0f}")
                print(f"  Error (MAE): {rec.error_detail:.4f}")
            
            print(f"\nAggregated level ({parent_level}):")
            if rec.original_agg:
                print(f"  Original bins: {len(rec.original_agg)}")
                print(f"  Original sum: {rec.original_agg.values.sum():.0f}")
            if rec.reconciled_agg:
                print(f"  Reconciled bins: {len(rec.reconciled_agg)}")
                print(f"  Reconciled sum: {rec.reconciled_agg.values.sum():.0f}")
                print(f"  Error (MAE): {rec.error_agg:.4f}")
            
            print(f"\nCompute time: {rec.compute_time:.4f}s")
            return
    
    print(f"No reconciliation found: {child_level} -> {parent_level} ({method})")


def inspect_level_results(level: str):
    """
    Print one-liner information about each result in the registry for this level.
    Sorted by MAE % (best first).
    """
    entries = results_registry.get_level_entries(level)
    
    if not entries:
        print(f"No results in registry for level: {level}")
        return
    
    # Sort by MAE %
    sorted_entries = sorted(entries.values(), key=lambda e: e.mae_pct_vs_zero)
    
    # Calculate column widths
    origin_width = max(40, max(len(e.origin) for e in sorted_entries) + 2)
    
    print(f"\n{'='*80}")
    print(f"RESULTS REGISTRY: {level}")
    print(f"{'='*80}")
    print(f"  {'Origin':<{origin_width}} {'MAE%':>8} {'MAE':>10} {'Cutoffs':>25}")
    print(f"  {'-'*origin_width} {'-'*8} {'-'*10} {'-'*25}")
    
    max_rows = 20
    head_n = 10
    n = len(sorted_entries)
    for i, entry in enumerate(sorted_entries):
        if n > max_rows and head_n <= i < n - head_n:
            if i == head_n:
                print(f"  {'...':<{origin_width}} {'':>8} {'':>10} {'':>25}")
            continue
        cutoffs = f"{entry.cutoff_train.strftime('%Y-%m-%d')} / {entry.cutoff_test.strftime('%Y-%m-%d')}"
        print(f"  {entry.origin:<{origin_width}} {entry.mae_pct_vs_zero:>8.2f} {entry.mae:>10.4f} {cutoffs:>25}")
    
    print(f"  {'-'*origin_width} {'-'*8} {'-'*10} {'-'*25}")
    print(f"  Total entries: {len(sorted_entries)}")


def compute_mae_between(data1: LevelData, data2: LevelData) -> float:
    """
    Compute MAE between two LevelData objects with key alignment.
    Only compares values where keys match in both.
    """
    if data1 is None or data2 is None:
        return float('nan')
    
    # Build index for data2
    data2.build_index()
    
    matched_diffs = []
    for i, key in enumerate(data1.keys):
        key_tuple = tuple(key)
        if key_tuple in data2._key_index:
            idx2 = data2._key_index[key_tuple]
            matched_diffs.append(abs(data1.values[i] - data2.values[idx2]))
    
    if not matched_diffs:
        return float('nan')
    
    return float(np.mean(matched_diffs))


def check_consistency():
    """
    Check hierarchical consistency of reconciliations with detailed MAE breakdown.
    
    For reconciled_detail (3 columns):
    - vs orig_detail: direct comparison (same level)
    - vs recon_agg: convert recon_agg to detail via top_down, compare
    - vs orig_agg: convert orig_agg to detail via top_down, compare
    
    For reconciled_agg (3 columns):
    - vs orig_detail: convert orig_detail to agg via bottom_up, compare
    - vs recon_detail: convert recon_detail to agg via bottom_up, compare
    - vs orig_agg: direct comparison (same level)
    """
    print("\n" + "=" * 140)
    print("RECONCILIATION CONSISTENCY CHECK")
    print("=" * 140)
    
    # Create reconciliation engine for conversions
    recon_engine = ReconciliationEngine(debug.dim_registry, debug.level_registry)
    
    # Header - compute column widths from data
    col_w = 14
    pair_w = max(len("From -> To"), max(len(f"{r.level_from} -> {r.level_to}") for r in debug.reconciliation_results))
    meth_w = max(len("Method"), max(len(r.method) for r in debug.reconciliation_results))
    print(f"\n{'From -> To':<{pair_w}} {'Method':<{meth_w}} | "
          f"{'Detail MAE':^{3*col_w+2}} | {'Agg MAE':^{3*col_w+2}}")
    print(f"{'':<{pair_w}} {'':<{meth_w}} | "
          f"{'vs orig_d':>{col_w}} {'vs recon_a':>{col_w}} {'vs orig_a':>{col_w}} | "
          f"{'vs orig_d':>{col_w}} {'vs recon_d':>{col_w}} {'vs orig_a':>{col_w}}")
    print(f"{'-'*pair_w} {'-'*meth_w} | {'-'*col_w} {'-'*col_w} {'-'*col_w} | {'-'*col_w} {'-'*col_w} {'-'*col_w}")

    for rec in debug.reconciliation_results:
        pair = f"{rec.level_from} -> {rec.level_to}"

        if rec.reconciled_detail is None or rec.reconciled_agg is None:
            print(f"{pair:<{pair_w}} {rec.method:<{meth_w}} | {'N/A':>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}} | {'N/A':>{col_w}} {'N/A':>{col_w}} {'N/A':>{col_w}}")
            continue
        
        parent_level = rec.level_to
        
        # === Reconciled Detail MAE columns ===
        # 1. vs original_detail (same level, direct)
        mae_d_vs_orig_d = compute_mae_between(rec.reconciled_detail, rec.original_detail)
        
        # 2. vs reconciled_agg (convert agg to detail via top_down)
        recon_agg_as_detail = recon_engine.top_down(rec.reconciled_agg, rec.reconciled_detail)
        mae_d_vs_recon_a = compute_mae_between(rec.reconciled_detail, recon_agg_as_detail)
        
        # 3. vs original_agg (convert orig_agg to detail via top_down)
        orig_agg_as_detail = recon_engine.top_down(rec.original_agg, rec.original_detail)
        mae_d_vs_orig_a = compute_mae_between(rec.reconciled_detail, orig_agg_as_detail)
        
        # === Reconciled Agg MAE columns ===
        # 1. vs original_detail (convert orig_detail to agg via bottom_up)
        orig_detail_as_agg = recon_engine.bottom_up(rec.original_detail, parent_level)
        mae_a_vs_orig_d = compute_mae_between(rec.reconciled_agg, orig_detail_as_agg)
        
        # 2. vs reconciled_detail (convert recon_detail to agg via bottom_up)
        recon_detail_as_agg = recon_engine.bottom_up(rec.reconciled_detail, parent_level)
        mae_a_vs_recon_d = compute_mae_between(rec.reconciled_agg, recon_detail_as_agg)
        
        # 3. vs original_agg (same level, direct)
        mae_a_vs_orig_a = compute_mae_between(rec.reconciled_agg, rec.original_agg)
        
        # Format output
        def fmt(v):
            if np.isnan(v):
                return "N/A"
            return f"{v:.4f}"
        
        print(f"{pair:<{pair_w}} {rec.method:<{meth_w}} | "
              f"{fmt(mae_d_vs_orig_d):>{col_w}} {fmt(mae_d_vs_recon_a):>{col_w}} {fmt(mae_d_vs_orig_a):>{col_w}} | "
              f"{fmt(mae_a_vs_orig_d):>{col_w}} {fmt(mae_a_vs_recon_d):>{col_w}} {fmt(mae_a_vs_orig_a):>{col_w}}")
    
    print("\nLegend:")
    print("  Detail MAE: MAE of reconciled_detail vs {orig_detail, recon_agg->detail, orig_agg->detail}")
    print("  Agg MAE:    MAE of reconciled_agg vs {orig_detail->agg, recon_detail->agg, orig_agg}")
    print("  ->detail: top_down conversion  |  ->agg: bottom_up conversion")


def compute_reconciliation_matrix(detail_level: str, agg_level: str, method: str) -> Optional[Dict]:
    """
    Compute reconciliation matrix without printing.
    
    Returns:
        Dict with keys:
            'matrix': Dict[(det_model, agg_model)] -> (a, A, b, B)
            'detail_models': List of detail model names
            'agg_models': List of agg model names
            'min_A': (A_value, det_model, agg_model)
            'min_B': (B_value, det_model, agg_model)
        Or None if computation fails.
    """
    if detail_level not in debug.model_results:
        return None
    if agg_level not in debug.model_results:
        return None
    
    # Get all models at each level
    detail_models = list(debug.model_results[detail_level].keys())
    agg_models = list(debug.model_results[agg_level].keys())
    
    if not detail_models or not agg_models:
        return None
    
    # Get zero model MAEs for reference
    zero_det_result = debug.model_results[detail_level].get('zero')
    zero_agg_result = debug.model_results[agg_level].get('zero')
    
    zero_det_mae = float('nan')
    zero_agg_mae = float('nan')
    
    if zero_det_result and zero_det_result.predictions and zero_det_result.predictions.actuals is not None:
        zero_det_pred = zero_det_result.predictions
        zero_det_mae = float(np.mean(np.abs(zero_det_pred.values - zero_det_pred.actuals)))
    
    if zero_agg_result and zero_agg_result.predictions and zero_agg_result.predictions.actuals is not None:
        zero_agg_pred = zero_agg_result.predictions
        zero_agg_mae = float(np.mean(np.abs(zero_agg_pred.values - zero_agg_pred.actuals)))
    
    # Create reconciliation engine
    recon_engine = ReconciliationEngine(debug.dim_registry, debug.level_registry)
    
    # Build the matrix
    # (detail_model, agg_model) -> (det_ratio_vs_orig, det_ratio_vs_zero, agg_ratio_vs_orig, agg_ratio_vs_zero)
    matrix = {}
    
    for det_model in detail_models:
        det_result = debug.model_results[detail_level].get(det_model)
        if not det_result or not det_result.predictions:
            continue
        
        # Calculate original detail MAE vs actuals
        det_pred = det_result.predictions
        if det_pred.actuals is None:
            continue
        orig_det_mae = float(np.mean(np.abs(det_pred.values - det_pred.actuals)))
        
        for agg_model in agg_models:
            agg_result = debug.model_results[agg_level].get(agg_model)
            if not agg_result or not agg_result.predictions:
                continue
            
            # Calculate original agg MAE vs actuals
            agg_pred = agg_result.predictions
            if agg_pred.actuals is None:
                continue
            orig_agg_mae = float(np.mean(np.abs(agg_pred.values - agg_pred.actuals)))
            
            # Apply reconciliation
            try:
                if method == 'bottom_up':
                    recon_det, recon_agg = recon_engine.bottom_up_pair(det_pred, agg_pred, agg_level)
                elif method == 'top_down':
                    recon_det = recon_engine.top_down(agg_pred, det_pred)
                    recon_agg = agg_pred  # top_down preserves agg
                    # Need to create proper LevelData for recon_det with actuals
                    recon_det = LevelData(
                        level=det_pred.level,
                        dim_names=det_pred.dim_names,
                        keys=det_pred.keys.copy(),
                        values=recon_det.values,
                        actuals=det_pred.actuals
                    )
                elif method == 'mint':
                    recon_det, recon_agg = recon_engine.mint(det_pred, agg_pred, detail_level, agg_level)
                elif method == 'wls':
                    recon_det, recon_agg = recon_engine.wls(det_pred, agg_pred, detail_level, agg_level)
                else:
                    continue
                
                # Align actuals for reconciled data
                recon_det.actuals = det_pred.actuals
                recon_agg = recon_engine._align_actuals(recon_agg, agg_pred)
                
                # Calculate reconciled MAEs
                recon_det_mae = float(np.mean(np.abs(recon_det.values - recon_det.actuals)))
                if recon_agg.actuals is not None:
                    recon_agg_mae = float(np.mean(np.abs(recon_agg.values - recon_agg.actuals)))
                else:
                    recon_agg_mae = float('nan')
                
                # Calculate ratios vs original (as percentages)
                det_ratio_orig = (recon_det_mae / orig_det_mae * 100) if orig_det_mae > 0 else float('nan')
                agg_ratio_orig = (recon_agg_mae / orig_agg_mae * 100) if orig_agg_mae > 0 else float('nan')
                
                # Calculate ratios vs zero method (as percentages)
                det_ratio_zero = (recon_det_mae / zero_det_mae * 100) if zero_det_mae > 0 else float('nan')
                agg_ratio_zero = (recon_agg_mae / zero_agg_mae * 100) if zero_agg_mae > 0 else float('nan')
                
                matrix[(det_model, agg_model)] = (det_ratio_orig, det_ratio_zero, agg_ratio_orig, agg_ratio_zero)
                
                # Register reconciliation results to registry
                # Skip bottom_up detail (unchanged), skip top_down agg (unchanged)
                if method != 'bottom_up' and recon_det is not None:
                    origin_det = f"{method}_{det_model}_{agg_model}_det"
                    results_registry.register(
                        level=detail_level,
                        origin=origin_det,
                        predictions=recon_det,
                        cutoff_train=CUTOFF_TRAIN,
                        cutoff_test=CUTOFF_TEST
                    )
                
                if method != 'top_down' and recon_agg is not None:
                    origin_agg = f"{method}_{det_model}_{agg_model}_agg"
                    results_registry.register(
                        level=agg_level,
                        origin=origin_agg,
                        predictions=recon_agg,
                        cutoff_train=CUTOFF_TRAIN,
                        cutoff_test=CUTOFF_TEST
                    )
                
            except Exception as e:
                log_warning(f'recon_matrix_{det_model}_{agg_model}', 
                           f"Reconciliation failed: {e}")
                matrix[(det_model, agg_model)] = (float('nan'), float('nan'), float('nan'), float('nan'))
    
    # Find minimum A and B
    min_A = (float('inf'), None, None)
    min_B = (float('inf'), None, None)
    
    for (det_model, agg_model), (a, A, b, B) in matrix.items():
        if not np.isnan(A) and A < min_A[0]:
            min_A = (A, det_model, agg_model)
        if not np.isnan(B) and B < min_B[0]:
            min_B = (B, det_model, agg_model)
    
    return {
        'matrix': matrix,
        'detail_models': detail_models,
        'agg_models': agg_models,
        'min_A': min_A if min_A[1] is not None else (float('nan'), None, None),
        'min_B': min_B if min_B[1] is not None else (float('nan'), None, None),
    }


def print_reconciliation_matrix(detail_level: str, agg_level: str, method: str) -> Optional[Tuple]:
    """
    Print a matrix showing reconciliation effectiveness for all model combinations.
    
    Args:
        detail_level: Name of the detail (child) level
        agg_level: Name of the aggregate (parent) level
        method: Reconciliation method to use
    
    Matrix format:
        - Rows: Models used at detail level
        - Columns: Models used at agg level
        - Cell value: "a:A|b:B" where
            a = 100 * MAE(reconciled_detail vs actuals) / MAE(original_detail vs actuals)
            A = 100 * MAE(reconciled_detail vs actuals) / MAE(zero_detail vs actuals)
            b = 100 * MAE(reconciled_agg vs actuals) / MAE(original_agg vs actuals)
            B = 100 * MAE(reconciled_agg vs actuals) / MAE(zero_agg vs actuals)
        
        Values < 100 mean reconciliation improved accuracy
        Values > 100 mean reconciliation degraded accuracy
    
    Returns:
        Tuple of (min_A_info, min_B_info) where each is (value, det_model, agg_model)
        Or None if computation fails.
    """
    result = compute_reconciliation_matrix(detail_level, agg_level, method)
    
    if result is None:
        print(f"No results for levels: {detail_level} / {agg_level}")
        return None
    
    matrix = result['matrix']
    detail_models = result['detail_models']
    agg_models = result['agg_models']
    min_A = result['min_A']
    min_B = result['min_B']
    
    # Compute row minimums (for each detail model, min across all agg models)
    row_mins = {}  # det_model -> (min_a, min_A, min_b, min_B)
    for det_model in detail_models:
        min_a, min_A_row, min_b, min_B_row = float('inf'), float('inf'), float('inf'), float('inf')
        for agg_model in agg_models:
            if (det_model, agg_model) in matrix:
                a, A, b, B = matrix[(det_model, agg_model)]
                if not np.isnan(a) and a < min_a:
                    min_a = a
                if not np.isnan(A) and A < min_A_row:
                    min_A_row = A
                if not np.isnan(b) and b < min_b:
                    min_b = b
                if not np.isnan(B) and B < min_B_row:
                    min_B_row = B
        row_mins[det_model] = (min_a, min_A_row, min_b, min_B_row)
    
    # Compute column minimums (for each agg model, min across all detail models)
    col_mins = {}  # agg_model -> (min_a, min_A, min_b, min_B)
    for agg_model in agg_models:
        min_a, min_A_col, min_b, min_B_col = float('inf'), float('inf'), float('inf'), float('inf')
        for det_model in detail_models:
            if (det_model, agg_model) in matrix:
                a, A, b, B = matrix[(det_model, agg_model)]
                if not np.isnan(a) and a < min_a:
                    min_a = a
                if not np.isnan(A) and A < min_A_col:
                    min_A_col = A
                if not np.isnan(b) and b < min_b:
                    min_b = b
                if not np.isnan(B) and B < min_B_col:
                    min_B_col = B
        col_mins[agg_model] = (min_a, min_A_col, min_b, min_B_col)
    
    # Global minimums for the MIN row
    global_min_a, global_min_A, global_min_b, global_min_B = float('inf'), float('inf'), float('inf'), float('inf')
    for (det_model, agg_model), (a, A, b, B) in matrix.items():
        if not np.isnan(a) and a < global_min_a:
            global_min_a = a
        if not np.isnan(A) and A < global_min_A:
            global_min_A = A
        if not np.isnan(b) and b < global_min_b:
            global_min_b = b
        if not np.isnan(B) and B < global_min_B:
            global_min_B = B
    
    # Format cell helper
    def format_cell(a, A, b, B):
        if any(np.isnan(x) or np.isinf(x) for x in [a, A, b, B]):
            return "N/A"
        return f"{a:.0f}:{A:.0f}|{b:.0f}:{B:.0f}"
    
    # Calculate all cell strings to determine column widths
    cell_strings = {}
    for det_model in detail_models:
        for agg_model in agg_models:
            if (det_model, agg_model) in matrix:
                a, A, b, B = matrix[(det_model, agg_model)]
                cell_strings[(det_model, agg_model)] = format_cell(a, A, b, B)
            else:
                cell_strings[(det_model, agg_model)] = "N/A"
        # Row min cell
        min_vals = row_mins[det_model]
        cell_strings[(det_model, 'MIN')] = format_cell(*min_vals)
    
    # Column min cells
    for agg_model in agg_models:
        min_vals = col_mins[agg_model]
        cell_strings[('MIN', agg_model)] = format_cell(*min_vals)
    cell_strings[('MIN', 'MIN')] = format_cell(global_min_a, global_min_A, global_min_b, global_min_B)
    
    # Calculate dynamic column widths
    model_col_w = max(12, max(len(m) for m in detail_models), len('Detail : Agg'))
    
    col_widths = {}
    for agg_model in agg_models + ['MIN']:
        max_w = len(agg_model)
        for det_model in detail_models + ['MIN']:
            cell = cell_strings.get((det_model, agg_model), 'N/A')
            max_w = max(max_w, len(cell))
        col_widths[agg_model] = max_w + 1  # +1 for spacing
    
    # Print the matrix
    total_width = model_col_w + sum(col_widths.values()) + len(col_widths)
    print("\n" + "=" * total_width)
    print(f"RECONCILIATION MATRIX: {detail_level} -> {agg_level} (method: {method})")
    print("=" * total_width)
    print("Cell format: a:A|b:B  (a=% vs orig det, A=% vs zero det, b=% vs orig agg, B=% vs zero agg)")
    
    # Header row (agg models + MIN)
    header = f"{'Detail : Agg':<{model_col_w}}"
    for agg_model in agg_models + ['MIN']:
        header += f" {agg_model:^{col_widths[agg_model]}}"
    print(header)
    print("-" * len(header))
    
    # Data rows (detail models)
    for det_model in detail_models:
        row = f"{det_model:<{model_col_w}}"
        for agg_model in agg_models + ['MIN']:
            cell = cell_strings.get((det_model, agg_model), 'N/A')
            row += f" {cell:^{col_widths[agg_model]}}"
        print(row)
    
    # MIN row
    print("-" * len(header))
    row = f"{'MIN':<{model_col_w}}"
    for agg_model in agg_models + ['MIN']:
        cell = cell_strings.get(('MIN', agg_model), 'N/A')
        row += f" {cell:^{col_widths[agg_model]}}"
    print(row)
    print("=" * total_width)
    
    # Print and return minimum A and B
    print(f"Min A (detail vs zero): {min_A[0]:.1f}% with det_model={min_A[1]}, agg_model={min_A[2]}")
    print(f"Min B (agg vs zero):    {min_B[0]:.1f}% with det_model={min_B[1]}, agg_model={min_B[2]}")
    
    return (min_A, min_B)


def print_level_bins(level_name: str, model_name: str, source=None):
    """
    Print all bins for a level and model with predicted, actual values, and status.
    
    Shows union of training and test bins with status:
    - NEW_GROUP: Group (veh-geo) not in training, t<sale<=T, reg>T, pred=0
    - NEW_SALE: Group (veh-geo) known, t<sale<=T, reg>T, pred>0
    - KNOWN_IN_TEST: sale<=t, t<reg<=T, uninteresting as known
    - STILL_UNREG: sale<=t, reg>T, still unregistered, pred>0
    
    Decision tree:
    - reg > t:
        - sale <= t: IN training (sale <= t AND reg > t)
            - reg <= T: not in test -> KNOWN_IN_TEST (resolved between t and T)
            - reg > T: IN test -> STILL_UNREG (same bin in train & test, long delay)
        - sale > t: NOT in training
            - sale <= T: (sold between t and T)
                - reg > T: IN test (sold between t and T, still unregistered)
                    - group (veh-geo) in training: -> NEW_SALE (pred > 0)
                    - group (veh-geo) not in training: -> NEW_GROUP (pred = 0)
    
    Args:
        level_name: Name of the evaluation level (e.g., 'brand_state_all')
        model_name: Name of the model (e.g., 'linear_extrapolation')
        dim_registry: Optional DimensionRegistry for decoding keys to labels
    """
    # Determine source type and resolve result + dim_registry
    from_registry = False
    dim_registry = None
    result = None

    if isinstance(source, ResultsRegistry):
        reg_entry = source.get_level_entries(level_name).get(model_name)
        if not reg_entry or not reg_entry.predictions:
            print(f"No predictions for {model_name} at {level_name} in registry")
            return
        level_data = reg_entry.predictions
        from_registry = True
        dim_registry = debug.dim_registry  # use global for key formatting
    else:
        # source is a DimensionRegistry (or None) -- original behavior
        dim_registry = source
        if level_name not in debug.model_results:
            print(f"No results for level: {level_name}")
            return
        level_results = debug.model_results[level_name]
        result = level_results.get(model_name)
        if not result or not result.predictions:
            print(f"No predictions for {model_name} at {level_name}")
            return
        level_data = result.predictions

    dim_names = level_data.dim_names
    
    # Identify temporal vs group dimensions (matching train_and_predict logic)
    temporal_dims = {'date', 'week', 'ALL_TIME'}
    group_dim_indices = [i for i, d in enumerate(dim_names) if d not in temporal_dims]
    
    def get_group_key(full_key):
        """Extract group key (veh-geo, non-temporal dimensions) from full key."""
        if group_dim_indices:
            return tuple(full_key[i] for i in group_dim_indices)
        return ()
    
    # Build test lookup dicts (predictions and actuals)
    test_pred_dict = {}
    test_actual_dict = {}
    test_keys_set = set()
    
    for i, key in enumerate(level_data.keys):
        key_tuple = tuple(key)
        test_keys_set.add(key_tuple)
        test_pred_dict[key_tuple] = level_data.values[i]
        if level_data.actuals is not None:
            test_actual_dict[key_tuple] = level_data.actuals[i]
    
    # Build train lookup dicts - both full keys and group keys
    train_actual_dict = {}
    train_keys_set = set()
    train_groups_set = set()
    
    if result is not None and result.train_keys is not None and result.train_values is not None:
        for i, key in enumerate(result.train_keys):
            key_tuple = tuple(key)
            train_keys_set.add(key_tuple)
            train_groups_set.add(get_group_key(key_tuple))
            train_actual_dict[key_tuple] = result.train_values[i]
    
    # Get all unique keys (union of train and test)
    all_keys = train_keys_set | test_keys_set
    sorted_keys = sorted(all_keys)
    
    def format_key(key_tuple):
        if dim_registry:
            parts = []
            for i, dim_name in enumerate(dim_names):
                if dim_registry.has(dim_name):
                    dim = dim_registry.get(dim_name)
                    if key_tuple[i] < len(dim.values):
                        parts.append(str(dim.values[key_tuple[i]]))
                    else:
                        parts.append(str(key_tuple[i]))
                else:
                    parts.append(dim_name)
            return ' | '.join(parts)
        else:
            return ' | '.join(str(k) for k in key_tuple)
    
    def get_status(key_tuple):
        in_train = key_tuple in train_keys_set
        in_test = key_tuple in test_keys_set
        group_key = get_group_key(key_tuple)
        group_in_train = group_key in train_groups_set
        
        if in_train and in_test:
            return "STILL_UNREG"   # sale<=t, reg>T, still unregistered
        elif in_train and not in_test:
            return "KNOWN_IN_TEST" # sale<=t, t<reg<=T, resolved/known
        elif not in_train and in_test:
            if group_in_train:
                return "NEW_SALE"  # t<sale<=T, reg>T, group known, pred>0
            else:
                return "NEW_GROUP" # t<sale<=T, reg>T, group unseen, pred=0
        else:
            return "UNKNOWN"
    
    # Column widths
    key_width = max(40, max(len(format_key(k)) for k in sorted_keys) if sorted_keys else 40)
    pred_width = 12
    actual_width = 12
    diff_width = 12
    status_width = 14
    total_width = key_width + pred_width + actual_width + diff_width + status_width + 13
    
    # Print header
    print("\n" + "=" * total_width)
    source_tag = " (from registry)" if from_registry else ""
    print(f"LEVEL: {level_name} | MODEL: {model_name}{source_tag}")
    print(f"Dimensions: {dim_names}")
    print(f"Group dims (veh-geo): {[dim_names[i] for i in group_dim_indices]}")
    print("=" * total_width)
    print(f"{'Bin Key':<{key_width}} {'Predicted':>{pred_width}} {'Actual':>{actual_width}} {'Diff':>{diff_width}} {'Status':>{status_width}}")
    print(f"{'-' * key_width} {'-' * pred_width} {'-' * actual_width} {'-' * diff_width} {'-' * status_width}")
    
    # Counters for each status
    status_counts = {'NEW_GROUP': 0, 'NEW_SALE': 0, 'KNOWN_IN_TEST': 0, 'STILL_UNREG': 0}
    status_pred_sums = {'NEW_GROUP': 0.0, 'NEW_SALE': 0.0, 'KNOWN_IN_TEST': 0.0, 'STILL_UNREG': 0.0}
    status_actual_sums = {'NEW_GROUP': 0.0, 'NEW_SALE': 0.0, 'KNOWN_IN_TEST': 0.0, 'STILL_UNREG': 0.0}
    status_train_sums = {'NEW_GROUP': 0.0, 'NEW_SALE': 0.0, 'KNOWN_IN_TEST': 0.0, 'STILL_UNREG': 0.0}
    total_pred = 0.0
    total_actual = 0.0
    
    for key_tuple in sorted_keys:
        key_str = format_key(key_tuple)
        status = get_status(key_tuple)
        status_counts[status] += 1
        
        pred_val = test_pred_dict.get(key_tuple)
        actual_val = test_actual_dict.get(key_tuple)
        train_val = train_actual_dict.get(key_tuple)
        
        if pred_val is not None:
            pred_str = f"{pred_val:.4f}"
            total_pred += pred_val
            status_pred_sums[status] += pred_val
        else:
            pred_str = "N/A"
        
        if status == "KNOWN_IN_TEST":
            if train_val is not None:
                actual_str = f"{train_val:.4f}"
                status_train_sums[status] += train_val
            else:
                actual_str = "N/A"
            diff_str = "N/A"
        else:
            if actual_val is not None:
                actual_str = f"{actual_val:.4f}"
                total_actual += actual_val
                status_actual_sums[status] += actual_val
            else:
                actual_str = "N/A"
            if pred_val is not None and actual_val is not None:
                diff_str = f"{pred_val - actual_val:+.4f}"
            else:
                diff_str = "N/A"
        
        print(f"{key_str:<{key_width}} {pred_str:>{pred_width}} {actual_str:>{actual_width}} {diff_str:>{diff_width}} {status:>{status_width}}")
    
    # Totals
    print(f"{'-' * key_width} {'-' * pred_width} {'-' * actual_width} {'-' * diff_width} {'-' * status_width}")
    diff_total = total_pred - total_actual
    print(f"{'TOTAL (test)':}{' '*(key_width-12)} {total_pred:>{pred_width}.4f} {total_actual:>{actual_width}.4f} {diff_total:>+{diff_width}.4f}")
    
    # Summary by status
    print(f"\n{'='*95}")
    print("SUMMARY BY STATUS")
    print(f"{'='*95}")
    print(f"{'Status':<14} {'Count':>8} {'Pred Sum':>12} {'Actual Sum':>12} {'Note':<45}")
    print(f"{'-'*14} {'-'*8} {'-'*12} {'-'*12} {'-'*45}")
    print(f"{'NEW_GROUP':<14} {status_counts['NEW_GROUP']:>8} {status_pred_sums['NEW_GROUP']:>12.2f} {status_actual_sums['NEW_GROUP']:>12.2f} {'t<sale<=T, reg>T, group (veh-geo) unseen, pred=0':<45}")
    print(f"{'NEW_SALE':<14} {status_counts['NEW_SALE']:>8} {status_pred_sums['NEW_SALE']:>12.2f} {status_actual_sums['NEW_SALE']:>12.2f} {'t<sale<=T, reg>T, group (veh-geo) known, pred>0':<45}")
    print(f"{'KNOWN_IN_TEST':<14} {status_counts['KNOWN_IN_TEST']:>8} {'N/A':>12} {status_train_sums['KNOWN_IN_TEST']:>12.2f} {'sale<=t, t<reg<=T, uninteresting as known':<45}")
    print(f"{'STILL_UNREG':<14} {status_counts['STILL_UNREG']:>8} {status_pred_sums['STILL_UNREG']:>12.2f} {status_actual_sums['STILL_UNREG']:>12.2f} {'sale<=t, reg>T, still unregistered, pred>0':<45}")
    print(f"{'-'*14} {'-'*8} {'-'*12} {'-'*12}")
    total_bins = sum(status_counts.values())
    total_train_actual = status_train_sums['KNOWN_IN_TEST']
    print(f"{'TOTAL':<14} {total_bins:>8} {total_pred:>12.2f} {total_actual + total_train_actual:>12.2f}")
    
    print(f"\nTest target breakdown:")
    print(f"  NEW_GROUP (pred=0):   {status_counts['NEW_GROUP']:>4} bins, {status_actual_sums['NEW_GROUP']:>6.0f} cars missed")
    print(f"  NEW_SALE (pred>0):    {status_counts['NEW_SALE']:>4} bins, {status_actual_sums['NEW_SALE']:>6.0f} cars")
    print(f"  STILL_UNREG (pred>0): {status_counts['STILL_UNREG']:>4} bins, {status_actual_sums['STILL_UNREG']:>6.0f} cars")
    print(f"  ---")
    print(f"  Total test:           {status_counts['NEW_GROUP'] + status_counts['NEW_SALE'] + status_counts['STILL_UNREG']:>4} bins, {total_actual:>6.0f} cars")
    print(f"\nTrain resolved by T (KNOWN_IN_TEST): {status_counts['KNOWN_IN_TEST']} bins, {total_train_actual:.0f} cars")
    print("=" * total_width)


def optimum_search_analysis(recon_pairs: List[Tuple[str, str]], 
                            recon_methods: List[str] = None):
    """
    Search for optimal reconciliation parameters across all pairs and methods.
    
    For each reconciliation pair, iterates over all methods and finds:
    - Lowest A (detail level % vs zero) with parameters
    - Lowest B (agg level % vs zero) with parameters
    
    Args:
        recon_pairs: List of (child_level, parent_level) tuples
        recon_methods: List of reconciliation methods to try (default: all)
    
    Prints one line per recon pair with best A and B across all methods.
    """
    if recon_methods is None:
        recon_methods = ['bottom_up', 'top_down', 'mint', 'wls']
    
    # Calculate column widths
    pair_col_w = max(35, max(len(f"{p[0]} -> {p[1]}") for p in recon_pairs) + 2)
    method_w = max(len(m) for m in recon_methods) + 2
    val_w = 8
    model_w = 22
    
    # Print header
    total_width = pair_col_w + 2 * (method_w + val_w + 2 * model_w) + 10
    print("\n" + "=" * total_width)
    print("OPTIMUM SEARCH ANALYSIS: Best reconciliation parameters per level pair")
    print("=" * total_width)
    print(f"{'Recon Pair':<{pair_col_w}} | "
          f"{'Method':^{method_w}} {'A%':>{val_w}} {'det_model':<{model_w}} {'agg_model':<{model_w}} | "
          f"{'Method':^{method_w}} {'B%':>{val_w}} {'det_model':<{model_w}} {'agg_model':<{model_w}}")
    print("-" * total_width)
    
    results = []
    
    for child_level, parent_level in recon_pairs:
        pair_str = f"{child_level} -> {parent_level}"
        
        # Track best A and B across all methods
        best_A = (float('inf'), None, None, None)  # (value, det_model, agg_model, method)
        best_B = (float('inf'), None, None, None)  # (value, det_model, agg_model, method)
        
        for method in recon_methods:
            result = compute_reconciliation_matrix(child_level, parent_level, method)
            
            if result is None:
                continue
            
            min_A = result['min_A']
            min_B = result['min_B']
            
            # Check if this method gives better A
            if min_A[0] < best_A[0]:
                best_A = (min_A[0], min_A[1], min_A[2], method)
            
            # Check if this method gives better B
            if min_B[0] < best_B[0]:
                best_B = (min_B[0], min_B[1], min_B[2], method)
        
        # Format output
        if best_A[1] is not None:
            A_method = best_A[3]
            A_val = f"{best_A[0]:.1f}"
            A_det = best_A[1][:model_w-1] if len(best_A[1]) >= model_w else best_A[1]
            A_agg = best_A[2][:model_w-1] if len(best_A[2]) >= model_w else best_A[2]
        else:
            A_method, A_val, A_det, A_agg = "N/A", "N/A", "N/A", "N/A"
        
        if best_B[1] is not None:
            B_method = best_B[3]
            B_val = f"{best_B[0]:.1f}"
            B_det = best_B[1][:model_w-1] if len(best_B[1]) >= model_w else best_B[1]
            B_agg = best_B[2][:model_w-1] if len(best_B[2]) >= model_w else best_B[2]
        else:
            B_method, B_val, B_det, B_agg = "N/A", "N/A", "N/A", "N/A"
        
        print(f"{pair_str:<{pair_col_w}} | "
              f"{A_method:^{method_w}} {A_val:>{val_w}} {A_det:<{model_w}} {A_agg:<{model_w}} | "
              f"{B_method:^{method_w}} {B_val:>{val_w}} {B_det:<{model_w}} {B_agg:<{model_w}}")
        
        results.append({
            'pair': (child_level, parent_level),
            'best_A': best_A,
            'best_B': best_B
        })
    
    print("=" * total_width)
    print("A = % MAE(reconciled_detail vs actuals) / MAE(zero_detail vs actuals)")
    print("B = % MAE(reconciled_agg vs actuals) / MAE(zero_agg vs actuals)")
    
    # =========================================================================
    # BEST 3 RESULTS PER LEVEL (from results_registry)
    # =========================================================================
    
    all_levels = results_registry.get_all_levels()
    if all_levels:
        # Order levels as defined in LEVELS, then any remaining
        level_order = [lvl.name for lvl in LEVELS if lvl.name in all_levels]
        level_order += [lvl for lvl in all_levels if lvl not in level_order]
        
        # Calculate column widths
        level_w = max(25, max(len(l) for l in all_levels) + 2)
        origin_w = 45
        mae_w = 10
        
        print("\n" + "=" * (level_w + 3 * (origin_w + mae_w + 3) + 5))
        print("BEST 3 RESULTS PER LEVEL (by MAE % vs zero)")
        print("=" * (level_w + 3 * (origin_w + mae_w + 3) + 5))
        print(f"{'Level':<{level_w}} | "
              f"{'#1 Origin':<{origin_w}} {'MAE%':>{mae_w}} | "
              f"{'#2 Origin':<{origin_w}} {'MAE%':>{mae_w}} | "
              f"{'#3 Origin':<{origin_w}} {'MAE%':>{mae_w}}")
        print("-" * (level_w + 3 * (origin_w + mae_w + 3) + 5))
        
        for level in level_order:
            best3 = results_registry.get_best_n(level, 3)
            
            row = f"{level:<{level_w}} | "
            for i in range(3):
                if i < len(best3):
                    entry = best3[i]
                    origin_trunc = entry.origin[:origin_w-1] if len(entry.origin) >= origin_w else entry.origin
                    row += f"{origin_trunc:<{origin_w}} {entry.mae_pct_vs_zero:>{mae_w}.2f} | "
                else:
                    row += f"{'':<{origin_w}} {'N/A':>{mae_w}} | "
            
            print(row.rstrip(" |"))
        
        print("=" * (level_w + 3 * (origin_w + mae_w + 3) + 5))
    
    return results


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
    
    # Example inspections
    print("\n" + "=" * 70)
    print("EXAMPLE INSPECTIONS")
    print("=" * 70)
    
    # Inspect best model at a level
    if debug.best_models:
        first_level = list(debug.best_models.keys())[0]
        inspect_model(first_level)
    
    # Check consistency
    if debug.reconciliation_results:
        check_consistency()
    
    # Print reconciliation matrix
    print_reconciliation_matrix('brand_county_day', 'brand_state_week', 'top_down')
    
    # Run optimum search analysis (uses global RECON_PAIRS)
    optimum_search_analysis(RECON_PAIRS)
    
    # Inspect results registry for brand_county_day
    inspect_level_results('brand_county_day')

    # Inspect brand_state_all with wls_croston_sba_lightgbm_agg
    #inspect_level_results('brand_state_all')
    #inspect_model('brand_state_all', 'wls_croston_sba_lightgbm_agg')
    print_level_bins('brand_state_all', 'mint_croston_sba_linear_extrapolation_det', results_registry)
    print_level_bins('brand_state_week', 'mint_lasso_zero_agg', results_registry)
    #print_level_bins('brand_county_day', 'mint_elastic_net_zero_det', results_registry)

    # breakpoint()
    print()
