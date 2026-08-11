# Tools

Small, standalone utility scripts that support the pipeline but are not part
of it.

| File | Purpose |
|---|---|
| `inspect_ninapro_mat_files.py` | Prints the variable names, shapes, and dtypes found in a raw NinaPro `.mat` file (or every `.mat` file in a directory). Useful for sanity-checking a fresh NinaPro download before running the pipeline. Takes an optional path argument: `python inspect_ninapro_mat_files.py /path/to/ninapro/db2`. |
