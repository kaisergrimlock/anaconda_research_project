from pathlib import Path
import csv

HERE = Path(__file__).resolve().parent
csv_files = HERE.glob("*.csv")

unique_queries = set()
passages = 0
label_0 = 0
label_1 = 0
label_2 = 0
label_3 = 0

for fp in csv_files:
    with fp.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            passages +=1
            unique_queries.add(row["query"])

            rel = row.get("relevance", "").strip()
            match rel:
                case "0":
                    label_0 += 1
                case "1":
                    label_1 += 1
                case "2":
                    label_2 += 1
                case "3":
                    label_3 += 1


print("Unique queries:", len(unique_queries))
print("Label 0 count:", label_0)
print("Label 1 count:", label_1)
print("Label 2 count:", label_2)
print("Label 3 count:", label_3)
print("Total passages:", passages)