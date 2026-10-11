# Third Party Notices

TypeType Signals is licensed under the GNU Affero General Public License, version 3
(`LICENSE`). It depends on the Python packages listed in `pyproject.toml` and `uv.lock`,
each under its own license:

| Package | License |
|---|---|
| `fastapi` | MIT License |
| `uvicorn` | BSD-3-Clause |
| `pydantic` | MIT |
| `numpy` | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| `onnxruntime` | MIT License |
| `tokenizers` | Apache Software License |
| `huggingface-hub` | Apache Software License |
| `psycopg` | LGPL-3.0-only |
| `pgvector` | MIT |
| `redis` | MIT |
| `msgspec` | BSD-3-Clause |

`psycopg` is used as a library, unmodified.

## The model

Titles are read with `xenova/paraphrase-multilingual-mpnet-base-v2`, the ONNX build of
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, which is published under the
Apache License 2.0. The model is not part of this repository or of its image: the service
downloads it on first use. Another model can be set with `RECOMMENDER_EMBEDDING_MODEL`.
