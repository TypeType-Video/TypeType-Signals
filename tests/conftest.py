import os

# Tests never download a model: titles are compared by their words.
os.environ["RECOMMENDER_EMBEDDER"] = "hashing"
