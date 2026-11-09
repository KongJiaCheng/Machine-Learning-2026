from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "bank-full.csv"

if not DATA_PATH.is_file():
    raise FileNotFoundError(
        "\nDataset file not found.\n"
        f"Expected location:\n{DATA_PATH}\n\n"
        "Download the UCI Bank Marketing dataset from:\n"
        "https://archive.ics.uci.edu/dataset/222/bank+marketing\n\n"
        "Extract bank-full.csv and put it beside this Python script."
    )

df = pd.read_csv(DATA_PATH, sep=";")
if len(df.columns) == 1:
    raise ValueError(
        "The CSV appears to have been read as one column. "
        "Check that bank-full.csv is the original semicolon-separated file."
    )

df.columns = [str(column).strip() for column in df.columns]

if "y" not in df.columns:
    raise ValueError(
        "Target column 'y' was not found. "
        f"Available columns: {df.columns.tolist()}"
    )

print("\nDataset loaded successfully.")
print("Dataset dimensions:", df.shape)
print("Column names:", df.columns.tolist())


# Check duplicate records in the entire dataset
duplicate_count = df.duplicated().sum()

print("Total number of records:", len(df))
print("Number of duplicate records:", duplicate_count)

if duplicate_count == 0:
    print("\nNo duplicate records identified.\n")
else:
    print("\nDuplicate records identified.")
    duplicate_rows = df[df.duplicated(keep=False)].copy()
    duplicate_rows.insert(
        0,
        "CSVFileRow",
        duplicate_rows.index + 2
    )
    print("\nDuplicate records:")
    print(duplicate_rows)


numeric_columns = [
    "age",
    "balance",
    "day",
    "campaign"
]

outlier_results = []

for column in numeric_columns:

    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outlier_mask = (
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    )

    outlier_count = outlier_mask.sum()

    outlier_percentage = (
        outlier_count / len(df)
    ) * 100

    outlier_results.append({
        "Variable": column,
        "  Q1  ": Q1,
        "  Q3  ": Q3,
        "  IQR  ": IQR,
        "Lower Bound": lower_bound,
        "Upper Bound": upper_bound,
        "Potential Outliers": outlier_count,
        "Percentage (%)": outlier_percentage
    })

outlier_df = pd.DataFrame(outlier_results)

print("\nOutlier Analysis Using IQR Method:")
print(outlier_df.round(2).to_string(index=False))