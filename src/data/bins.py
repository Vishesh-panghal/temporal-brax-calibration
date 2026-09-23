"""Temporal binning module for dividing locked test cohort into chronological deployment windows."""

from typing import Tuple
import pandas as pd


def recover_calendar_dates(df: pd.DataFrame) -> pd.Series:
    """Recovers true calendar StudyDate from string or integer representation.
    
    In BRAX raw metadata, StudyDate is represented as YYYYMMDD (e.g. 20080325).
    """
    raw = df['StudyDate'].astype(str)
    # Extract 8 consecutive digits at the end if nanosecond formatting was applied
    extracted = raw.str.extract(r'(\d{8})$')[0]
    return pd.to_datetime(extracted, format='%Y%m%d', errors='coerce')


def discretize_temporal_test_bins(
    manifest_path: str = "data/processed/brax_temporal_manifest_updated.csv",
    output_path: str = "data/processed/brax_test_bins_manifest.csv",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions the locked test cohort (approx 20%) into ordered future temporal bins.
    
    Returns:
        full_df: The complete manifest with recovered 'calendar_date' and 'temporal_bin' columns.
        test_df: The subset corresponding to the locked test split with bin assignments.
    """
    df = pd.read_csv(manifest_path)
    df['calendar_date'] = recover_calendar_dates(df)
    df['year'] = df['calendar_date'].dt.year

    # Initialize bin column
    df['temporal_bin'] = 'anchor_train_val'

    # Filter test split
    test_mask = df['split'] == 'test'
    test_df = df[test_mask].copy()

    # Assign chronological bins:
    # Bin T1: 2015 (Late 2015 arrival)
    # Bin T2: 2016 (Mid future horizon)
    # Bin T3: 2017 (Distant future horizon)
    def assign_bin(row):
        yr = row['year']
        if yr <= 2015:
            return 'T1_2015'
        elif yr == 2016:
            return 'T2_2016'
        else:
            return 'T3_2017'

    df.loc[test_mask, 'temporal_bin'] = test_df.apply(assign_bin, axis=1)
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved temporal bins manifest to {output_path}")

    return df, df[test_mask]


if __name__ == '__main__':
    full_manifest, test_manifest = discretize_temporal_test_bins()
    print("\n--- Test Set Temporal Bin Distribution ---")
    print(test_manifest['temporal_bin'].value_counts().sort_index())
