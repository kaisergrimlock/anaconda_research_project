import os
import csv

# Folder containing your files
input_folder = "outputs/trec_dl_llm_label/translated/viet"
output_file = os.path.join(input_folder, "scatter_mixtral.csv")

# Dictionary to store: {docid: [rel_1, rel_2, ...]}
doc_rels = {}

# List and sort all CSV files that match the model name
files = sorted([f for f in os.listdir(input_folder)
                if "mistral" in f.lower() and "mixtral" in f.lower() and f.endswith(".csv")])

for idx, filename in enumerate(files, start=1):
    filepath = os.path.join(input_folder, filename)
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            docid = row["docid"]
            rel = row["relevance"]
            if docid not in doc_rels:
                doc_rels[docid] = [""] * len(files)  # prefill empty slots
            doc_rels[docid][idx - 1] = rel

# Write the combined output
with open(output_file, "w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out)
    header = ["docid"] + [f"rel_{i}" for i in range(1, len(files) + 1)]
    writer.writerow(header)
    for docid, rels in doc_rels.items():
        writer.writerow([docid] + rels)

print(f"Combined file written to: {output_file}")
