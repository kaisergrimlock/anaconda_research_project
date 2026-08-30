# pandas 3.0.5 API lookup

Verified locally with `inspect.signature` against the uv environment.

- `pandas.read_csv(..., dtype=str, keep_default_na=False, encoding="utf-8-sig")`
  returns a `DataFrame` while preserving CSV values as strings and empty fields as
  empty strings.
- `DataFrame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")`
  writes rows without adding the DataFrame index.
- `DataFrame.apply(func, axis=1)` applies a row transform.
- `Series.map(mapping_or_callable)` maps values using a mapping or callable.
