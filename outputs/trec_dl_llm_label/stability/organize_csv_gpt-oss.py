import os
import csv
import glob


# Folder containing your files
input_folder = "outputs/trec_dl_llm_label"

# Match just the GPT-OSS runs shown in your screenshot
pattern = os.path.join(input_folder, "*llm_labels_openai.gpt-oss-20b-1_0_top2*.csv")
files = sorted(glob.glob(pattern))  # your filenames start with timestamps, so this sorts chronologically

if not files:
    raise SystemExit(f"No matching files found for pattern: {pattern}")

# {docid: [rel_1, rel_2, ...]}
doc_rels = {}

for idx, filepath in enumerate(files, start=1):
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Be tolerant of column naming across your variants
        candidates = ["rel", "relevance", "rel_translated", "label"]
        try:
            rel_col = next(c for c in candidates if c in (reader.fieldnames or []))
        except StopIteration:
            raise ValueError(
                f"No relevance/label column found in {os.path.basename(filepath)}. "
                f"Have columns: {reader.fieldnames}"
            )

        for row in reader:
            docid = str(row.get("docid", "")).strip()
            if not docid:
                continue
            if docid not in doc_rels:
                doc_rels[docid] = [""] * len(files)  # prefill empty slots
            doc_rels[docid][idx - 1] = str(row.get(rel_col, "")).strip()

# Write the combined output
output_file = os.path.join(input_folder, "scatter_gpt-oss.csv")
with open(output_file, "w", encoding="utf-8", newline="") as out:
    writer = csv.writer(out)
    header = ["docid"] + [f"rel_{i}" for i in range(1, len(files) + 1)]
    writer.writerow(header)
    for docid in sorted(doc_rels):  # stable order
        writer.writerow([docid] + doc_rels[docid])

print(f"Wrote {len(doc_rels)} rows from {len(files)} files to: {output_file}")
