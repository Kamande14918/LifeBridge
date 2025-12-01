import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
p = ROOT / 'data' / 'first_aid_sample.csv'
print('CSV path:', p)
with open(p, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for i, r in enumerate(reader):
        print('ROW', i, r)
        if i > 5:
            break
