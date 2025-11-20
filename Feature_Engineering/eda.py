from pathlib import Path

import numpy as np
import pandas as pd

from data_prep import (
    preprocess_df,
    BASE_FEATURE_COLS,
    TARGET_VALUE,
    TARGET_OVERALL,
    TARGET_POSITION,
)


def main():
    df = preprocess_df()
    print(len(df))
    print("=== BASIC INFO ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumns (first 30):", df.columns.tolist()[:30], "...\n")

    # ------------------------------------------------------------
    # TARGET SUMMARIES
    # ------------------------------------------------------------
    print("=== TARGET DISTRIBUTIONS ===")
    print("\nValue (EUR):")
    print(df[TARGET_VALUE].describe())

    print("\nLog Value (EUR):")
    print(df["log_value_eur"].describe())

    print("\nOverall rating:")
    print(df[TARGET_OVERALL].describe())

    print("\nPosition_10 distribution:")
    print(df[TARGET_POSITION].value_counts())

    # ------------------------------------------------------------
    # CORRELATIONS
    # ------------------------------------------------------------
    # We'll look at correlations between features and:
    #   - log_value_eur
    #   - value_eur (raw)
    #   - overall
    corr_cols = BASE_FEATURE_COLS + [TARGET_VALUE, "log_value_eur", TARGET_OVERALL]
    corr_df = df[corr_cols].corr()

    # Save full correlation matrix to CSV for later inspection
    out_path = Path("./correlation_matrix.csv")
    corr_df.to_csv(out_path, index=True)
    print(f"\n[Saved] Full correlation matrix → {out_path.resolve()}")

    # Helper to pretty-print top correlations
    def print_top_corr(target_col: str, top_n: int = 15):
        print(f"\n=== Top {top_n} correlations with {target_col} ===")
        # Drop self-correlation and sort
        series = corr_df[target_col].drop(labels=[target_col])
        series = series.sort_values(ascending=False)
        print(series.head(top_n))

    print_top_corr("log_value_eur", top_n=15)
    print_top_corr(TARGET_VALUE, top_n=15)
    print_top_corr(TARGET_OVERALL, top_n=15)

    # Optional: also show correlation between value and overall
    print("\nCorrelation between value_eur and overall:",
          float(corr_df.loc[TARGET_VALUE, TARGET_OVERALL]))


if __name__ == "__main__":
    main()
