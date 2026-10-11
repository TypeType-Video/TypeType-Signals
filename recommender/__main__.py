"""Starts the service: `python -m recommender`. Address and port come from the environment."""

import uvicorn

from .settings import text, whole

if __name__ == "__main__":
    uvicorn.run(
        "recommender.main:app",
        host=text("RECOMMENDER_HOST", "0.0.0.0"),
        port=whole("RECOMMENDER_PORT", 8190),
    )
