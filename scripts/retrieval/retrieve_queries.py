import argparse
import csv
from pathlib import Path


def normalize_text(value):
    if value is None:
        return ""
    return " ".join(str(value).split())


def sort_qid(qid):
    qid = str(qid)
    return (0, int(qid)) if qid.isdigit() else (1, qid)


def collect_queries_from_csv(path):
    """
    Collect unique qid/query pairs from one CSV.
    Expected useful columns:
      qid, query
    """
    queries = {}

    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)

            if not reader.fieldnames:
                return queries

            fieldnames = [f.strip() for f in reader.fieldnames]
            field_lookup = {f.lower(): f for f in fieldnames}

            qid_col = (
                field_lookup.get("qid")
                or field_lookup.get("query_id")
                or field_lookup.get("topic")
                or field_lookup.get("topic_id")
            )

            query_col = (
                field_lookup.get("query")
                or field_lookup.get("title")
                or field_lookup.get("question")
                or field_lookup.get("text")
            )

            if not qid_col or not query_col:
                return queries

            for row in reader:
                qid = normalize_text(row.get(qid_col, ""))
                query = normalize_text(row.get(query_col, ""))

                if qid and query and qid not in queries:
                    queries[qid] = query

    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)

            if not reader.fieldnames:
                return queries

            fieldnames = [f.strip() for f in reader.fieldnames]
            field_lookup = {f.lower(): f for f in fieldnames}

            qid_col = (
                field_lookup.get("qid")
                or field_lookup.get("query_id")
                or field_lookup.get("topic")
                or field_lookup.get("topic_id")
            )

            query_col = (
                field_lookup.get("query")
                or field_lookup.get("title")
                or field_lookup.get("question")
                or field_lookup.get("text")
            )

            if not qid_col or not query_col:
                return queries

            for row in reader:
                qid = normalize_text(row.get(qid_col, ""))
                query = normalize_text(row.get(query_col, ""))

                if qid and query and qid not in queries:
                    queries[qid] = query

    return queries


def discover_input_dirs(project_root, year):
    """
    Try common locations in order.

    Your earlier error happened because retrieved/raw did not exist.
    This version supports both:

      retrieved/raw/trec_dl_2022/...
      retrieved/trec_dl_2022/judged/...
      retrieved/trec_dl_2022/raw/...
      retrieved/trec_dl_2022/...
    """
    candidates = [
        project_root / "retrieved" / "raw" / f"trec_dl_{year}",
        project_root / "retrieved" / "raw" / f"trec_dl{year}",
        project_root / "retrieved" / f"trec_dl_{year}" / "judged",
        project_root / "retrieved" / f"trec_dl_{year}" / "raw",
        project_root / "retrieved" / f"trec_dl_{year}",
        project_root / "retrieved" / f"trec_dl{year}" / "judged",
        project_root / "retrieved" / f"trec_dl{year}" / "raw",
        project_root / "retrieved" / f"trec_dl{year}",
    ]

    return [p for p in candidates if p.exists() and p.is_dir()]


def collect_queries_for_year(project_root, year):
    queries = {}
    input_dirs = discover_input_dirs(project_root, year)

    if not input_dirs:
        print(f"[WARN] No input folder found for year={year}")
        print("[TRIED]")
        for p in [
            project_root / "retrieved" / "raw" / f"trec_dl_{year}",
            project_root / "retrieved" / "raw" / f"trec_dl{year}",
            project_root / "retrieved" / f"trec_dl_{year}" / "judged",
            project_root / "retrieved" / f"trec_dl_{year}" / "raw",
            project_root / "retrieved" / f"trec_dl_{year}",
        ]:
            print(f"  {p}")
        return queries

    print(f"[INFO] Year {year}: reading from:")
    for d in input_dirs:
        print(f"  {d}")

    for input_dir in input_dirs:
        for csv_path in sorted(input_dir.rglob("*.csv")):
            # Avoid reading our output folder again if script is rerun.
            if "retrieved\\queries" in str(csv_path) or "retrieved/queries" in str(csv_path):
                continue

            parsed = collect_queries_from_csv(csv_path)

            for qid, query in parsed.items():
                if qid not in queries:
                    queries[qid] = query

    return queries


def write_queries_csv(queries, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["qid", "query"])

        for qid, query in sorted(queries.items(), key=lambda item: sort_qid(item[0])):
            writer.writerow([qid, query])


def infer_years_from_retrieved(project_root):
    years = set()
    retrieved = project_root / "retrieved"

    if not retrieved.exists():
        return []

    for path in retrieved.iterdir():
        if not path.is_dir():
            continue

        name = path.name

        if name.startswith("trec_dl_"):
            years.add(name.replace("trec_dl_", ""))
        elif name.startswith("trec_dl"):
            years.add(name.replace("trec_dl", ""))

    raw_root = retrieved / "raw"
    if raw_root.exists():
        for path in raw_root.iterdir():
            if not path.is_dir():
                continue

            name = path.name

            if name.startswith("trec_dl_"):
                years.add(name.replace("trec_dl_", ""))
            elif name.startswith("trec_dl"):
                years.add(name.replace("trec_dl", ""))

    return sorted(years)


def main():
    project_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(
        description="Extract unique qid/query pairs from retrieved TREC-DL CSV files."
    )

    parser.add_argument(
        "--years",
        nargs="*",
        default=None,
        help="Years to process, e.g. --years 2021 2022. If omitted, years are inferred from retrieved folders.",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=project_root / "retrieved" / "queries",
        help="Output folder. Default: retrieved/queries",
    )

    args = parser.parse_args()

    years = args.years or infer_years_from_retrieved(project_root)

    if not years:
        raise FileNotFoundError(
            "No TREC-DL year folders found under retrieved/. "
            "Expected folders like retrieved/trec_dl_2022/judged or retrieved/raw/trec_dl_2022."
        )

    for year in years:
        queries = collect_queries_for_year(project_root, year)

        out_file = args.output_root / f"trec_dl{year}.csv"

        if queries:
            write_queries_csv(queries, out_file)
            print(f"[DONE] Wrote {len(queries)} unique queries to {out_file}")
        else:
            print(f"[WARN] No queries found for year={year}")


if __name__ == "__main__":
    main()
