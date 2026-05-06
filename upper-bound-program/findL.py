import pandas as pd

df = pd.read_csv(
    "bf-5000-all.csv",
    dtype={
        "config": "int8",
        "pred1": "int8",
        "pred2": "int8",
        "d": "float64",
        "w": "float64",
        "distance": "float64",
    },
)

K = 5000

expected_rows = 2 * 2 * 2 * (K + 1) * (K + 2) // 2
print(f"rows: actual = {len(df):,}, expected = {expected_rows:,}")
assert len(df) == expected_rows, "row count mismatch"
assert df[["config", "d", "w", "pred1", "pred2"]].duplicated().sum() == 0, "duplicates found"

# --- Max diff over pairs differing in one OR both preds (fixed config, d, w) ---
group_cols = ["config", "d", "w"]
g_all = df.groupby(group_cols)["distance"]
max_both_change = float((g_all.max() - g_all.min()).max())

# --- Max diff over pairs differing in EXACTLY one pred ---
def spread(keys):
    s = df.groupby(keys)["distance"].agg(["max", "min"])
    return float((s["max"] - s["min"]).max())

max_one_change = max(
    spread(group_cols + ["pred1"]),
    spread(group_cols + ["pred2"]),
)

print("max diff changing only one pred:", max_one_change)
print("max diff changing one or both preds:", max_both_change)
