FROM python:3.12.15-slim-trixie AS build

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project --no-cache


FROM python:3.12.15-slim-trixie

RUN groupadd --system --gid 10001 typetype \
    && useradd --system --uid 10001 --gid typetype --home-dir /app --no-create-home typetype \
    && mkdir /models \
    && chown typetype:typetype /models
WORKDIR /app
COPY --from=build /app/.venv ./.venv
COPY recommender ./recommender
COPY LICENSE THIRD_PARTY_NOTICES.md ./

ENV PATH="/app/.venv/bin:$PATH" RECOMMENDER_MODEL_CACHE=/models
ENV MALLOC_ARENA_MAX=1 MALLOC_TRIM_THRESHOLD_=65536
ARG REVISION=unknown
ARG BUILD_TIME=unknown
ENV RECOMMENDER_REVISION=$REVISION RECOMMENDER_BUILD_TIME=$BUILD_TIME
USER typetype
EXPOSE 8190
CMD ["python", "-m", "recommender"]
