# app/utils.py
import orjson, pandas as pd, os

def csv_to_docs(csv_path: str, out_jsonl: str):
    df = pd.read_csv(csv_path)
    docs = []
    for _, r in df.iterrows():
        # Build a single text chunk with all fields to keep retrieval robust
        text = (
            f"Scenario: {r['Scenario']}\n"
            f"Question: {r['Question']}\n"
            f"Primary Action: {r['Primary Action']}\n"
            f"Emergency Contact: {r['Emergency Contact']}\n"
            f"VisualGuideURL: {r['VisualGuideURL']}\n"
        )
        meta = {"title": r["Scenario"], "contacts": r["Emergency Contact"], "visual": r["VisualGuideURL"]}
        docs.append({"text": text, "metadata": meta})
    with open(out_jsonl, "wb") as f:
        for d in docs:
            f.write(orjson.dumps(d) + b"\n")
    return docs
