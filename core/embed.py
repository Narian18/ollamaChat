from typing import Sequence

import numpy as np
from ollama import embed


EmbedVector = np.typing.NDArray[np.float64]


def embed_text(text: str) -> Sequence[float]:
    response = embed(model="embeddinggemma", input=text)

    return response.embeddings[0]


def cosine_similarity(embedding1: EmbedVector, embedding2: EmbedVector) -> float:
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )


def text_difference(text1: str, text2: str):
    embed1 = np.array(embed_text(text1))
    embed2 = np.array(embed_text(text2))

    return cosine_similarity(embed1, embed2)
