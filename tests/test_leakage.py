import pandas as pd
import numpy as np
import pytest
from datetime import date, timedelta
import sys
from pathlib import Path

# Add project root to sys.path to allow imports
# This assumes 'tests' is a directory at the project root, and 'jumpmodels', 'examples' are also at the root.
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Specific path for examples.nasdaq if feature.py relies on utils_dir from its own path
examples_nasdaq_path = project_root / "examples" / "nasdaq"
if str(examples_nasdaq_path) not in sys.path:
    sys.path.insert(0, str(examples_nasdaq_path))

# Now, attempt to import from the project structure
from jumpmodels.utils.index import align_index, filter_date_range
from jumpmodels.preprocess import StandardScalerPD
from examples.nasdaq.feature import feature_engineer


# Sample Data for testing
@pytest.fixture
def sample_return_series():
    dates = pd.to_datetime([date(2023, 1, 1) + timedelta(days=i) for i in range(100)])
    returns_data = np.random.randn(100) * 0.01 + 0.0001
    returns = pd.Series(returns_data, index=dates, name="returns")
    returns.iloc[0] = np.nan
    if len(returns.dropna()) > 5: # Ensure some negative returns for DD calculation if possible
        idx_points = returns.dropna().index
        if len(idx_points) > 1: returns.loc[idx_points[1]] = -0.01
        if len(idx_points) > 3: returns.loc[idx_points[3]] = -0.005
    return returns.dropna()

@pytest.fixture
def sample_feature_dataframe(sample_return_series):
    if sample_return_series.empty:
      return pd.DataFrame() # Return empty if input is empty
    X = pd.DataFrame({
        'feature1': np.random.rand(len(sample_return_series)),
        'feature2': np.random.rand(len(sample_return_series))
    }, index=sample_return_series.index)
    return X

def test_feature_engineering_direct_causality(sample_return_series):
    if sample_return_series.empty:
        pytest.skip("Sample return series is empty after dropna, skipping test.")

    features = feature_engineer(sample_return_series, ver="v0")

    pd.testing.assert_index_equal(features.index, sample_return_series.index)
    assert not features.isnull().values.any(), "Features should not contain NaNs"

    ret_series_copy = sample_return_series.copy()
    if len(ret_series_copy) < 15:
        pytest.skip("Sample return series too short for modification tests.")

    idx_t = ret_series_copy.index[10]
    # Ensure idx_t is valid after potential dropna
    if idx_t not in feature_engineer(ret_series_copy, ver="v0").index:
         idx_t = feature_engineer(ret_series_copy, ver="v0").index[0] # fallback if index changes

    features_orig_t = feature_engineer(ret_series_copy, ver="v0").loc[idx_t].copy()

    idx_t_plus_1 = ret_series_copy.index[11]
    val_orig_t_plus_1 = ret_series_copy.loc[idx_t_plus_1]
    ret_series_copy.loc[idx_t_plus_1] = 100.0
    features_mod_future_t = feature_engineer(ret_series_copy, ver="v0").loc[idx_t]
    pd.testing.assert_series_equal(features_orig_t, features_mod_future_t, check_dtype=False, rtol=1e-5)
    ret_series_copy.loc[idx_t_plus_1] = val_orig_t_plus_1

    val_orig_t = ret_series_copy.loc[idx_t]
    ret_series_copy.loc[idx_t] = 100.0
    features_mod_present_t = feature_engineer(ret_series_copy, ver="v0").loc[idx_t]
    assert not features_orig_t.equals(features_mod_present_t), "Features at t should change if ret_ser[t] changes."
    ret_series_copy.loc[idx_t] = val_orig_t

def test_align_index_behavior(sample_return_series):
    if len(sample_return_series) < 20:
        pytest.skip("Sample series too short for align index test.")
    full_series = sample_return_series.copy()
    subset_index = sample_return_series.index[10:20]
    dummy_y = pd.Series(1, index=subset_index)

    aligned_series = align_index(full_series, dummy_y)

    pd.testing.assert_index_equal(aligned_series.index, subset_index)
    assert len(aligned_series) == len(subset_index)
    pd.testing.assert_series_equal(aligned_series, full_series.loc[subset_index], check_dtype=False)

def test_train_test_split_chronological(sample_feature_dataframe):
    df = sample_feature_dataframe
    if df.empty or len(df) < 2: # Need at least 2 for a split
        pytest.skip("Sample dataframe too small or empty for this split test")

    train_end_date_loc = df.index[len(df) // 2]
    train_df = filter_date_range(df, end_date=train_end_date_loc)

    if train_df.empty:
        pytest.skip("Train df is empty after filtering.")
    actual_train_end_date = train_df.index.max()

    if actual_train_end_date == df.index.max():
        # This means train_df includes the last date, so test_df would be empty or non-existent
        test_df = pd.DataFrame()
    else:
        next_day_after_train_end_loc_idx = df.index.get_loc(actual_train_end_date) + 1
        next_day_after_train_end_loc = df.index[next_day_after_train_end_loc_idx]
        test_df = filter_date_range(df, start_date=next_day_after_train_end_loc)

    if not train_df.empty and not test_df.empty:
        assert train_df.index.max() < test_df.index.min()
    # Other cases (one or both empty) mean no overlap violation.

def test_scaler_fitting_on_train_only(sample_feature_dataframe):
    df = sample_feature_dataframe
    if df.empty or len(df) < 2: # Need at least 2 for a split
        pytest.skip("Sample dataframe too small or empty for this split test")

    train_end_date_loc = df.index[len(df) // 2]
    X_train = filter_date_range(df, end_date=train_end_date_loc)

    if X_train.empty: pytest.skip("X_train is empty after filtering.")
    actual_train_end_date = X_train.index.max()

    if actual_train_end_date == df.index.max():
        X_test = pd.DataFrame() # No data left for test set
    else:
        next_day_idx = df.index.get_loc(actual_train_end_date) + 1
        next_day_after_train_end_loc = df.index[next_day_idx]
        X_test = filter_date_range(df, start_date=next_day_after_train_end_loc)

    if X_test.empty : # If X_test becomes empty, cannot proceed with parts of test
        pytest.skip("X_test is empty for scaler test, cannot compare scaler params.")

    scaler = StandardScalerPD()
    # Ensure X_train has non-zero std for all columns if possible, or scaler might behave unexpectedly
    X_train = X_train.loc[:, X_train.std() > 1e-9] # Keep columns with some variance
    if X_train.empty: pytest.skip("X_train has no columns with variance.")

    X_train_processed = scaler.fit_transform(X_train)

    # Check transformed training data
    assert np.allclose(X_train_processed.mean(axis=0), 0, atol=1e-7)
    assert np.allclose(X_train_processed.std(axis=0, ddof=0), 1, atol=1e-7) # Use ddof=0 to match scaler's calculation

    # Apply to test data (if X_test has compatible columns)
    X_test_compatible = X_test[X_train.columns] # Ensure same columns
    if not X_test_compatible.empty:
      X_test_processed = scaler.transform(X_test_compatible)
      manual_X_test_processed = pd.DataFrame((X_test_compatible.values - scaler.scaler.mean_) / scaler.scaler.scale_, columns=X_test_compatible.columns, index=X_test_compatible.index)
      pd.testing.assert_frame_equal(X_test_processed, manual_X_test_processed, check_dtype=False, rtol=1e-5)

      # Compare with a scaler fit only on X_test
      scaler_for_test = StandardScalerPD()
      X_test_compatible_variant = X_test_compatible.loc[:, X_test_compatible.std() > 1e-9]
      if not X_test_compatible_variant.empty:
          scaler_for_test.fit(X_test_compatible_variant)
          mean_diff = np.abs(scaler.scaler.mean_ - scaler_for_test.scaler.mean_)
          scale_diff = np.abs(scaler.scaler.scale_ - scaler_for_test.scaler.scale_)
          if not np.all(mean_diff < 1e-3) and not X_train.equals(X_test_compatible_variant):
              assert not np.allclose(scaler.scaler.mean_, scaler_for_test.scaler.mean_, atol=1e-3)
          if not np.all(scale_diff < 1e-3)and not X_train.equals(X_test_compatible_variant):
              assert not np.allclose(scaler.scaler.scale_, scaler_for_test.scaler.scale_, atol=1e-3)
