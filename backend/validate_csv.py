import pandas as pd, sys

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/first_aid_sample.csv"
    df = pd.read_csv(path)
    print(df.head())
    required = ["Scenario","Question","Primary Action","Emergency Contact","VisualGuideURL"]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"
    assert len(df) > 0, "CSV has no rows"
    print("CSV validation passed:", len(df), "rows")

if __name__ == "__main__":
    main()
