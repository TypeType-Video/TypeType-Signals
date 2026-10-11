# Third Party Notices

TypeType Signals is licensed under the GNU Affero General Public License, version 3
(`LICENSE`). It depends on the Python packages listed in `pyproject.toml` and `uv.lock`,
each under its own license:

| Package | License |
|---|---|
| [`fastapi`](https://github.com/fastapi/fastapi) | MIT License |
| [`uvicorn`](https://github.com/Kludex/uvicorn) | BSD-3-Clause |
| [`pydantic`](https://github.com/pydantic/pydantic) | MIT |
| [`numpy`](https://github.com/numpy/numpy) | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| [`onnxruntime`](https://github.com/microsoft/onnxruntime) | MIT License |
| [`tokenizers`](https://github.com/huggingface/tokenizers) | Apache Software License |
| [`huggingface-hub`](https://github.com/huggingface/huggingface_hub) | Apache Software License |
| [`psycopg`](https://github.com/psycopg/psycopg) | LGPL-3.0-only |
| [`pgvector`](https://github.com/pgvector/pgvector-python) | MIT |
| [`redis`](https://github.com/redis/redis-py) | MIT |
| [`msgspec`](https://github.com/msgspec/msgspec) | BSD-3-Clause |

`psycopg` is used as a library, unmodified.

## The model

Titles are read with
[`xenova/paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/Xenova/paraphrase-multilingual-mpnet-base-v2),
the ONNX build of
[`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2),
which is published under the Apache License 2.0. The model is not part of this repository or of its image: the service
downloads it on first use. Another model can be set with `RECOMMENDER_EMBEDDING_MODEL`.
