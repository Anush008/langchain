import numpy as np
import pytest

from langchain_qdrant import FastEmbedSparse


@pytest.mark.parametrize("model_name", ["Qdrant/bm25", "Qdrant/bm42-all-minilm-l6-v2-attentions"])
def test_attention_embeddings(model_name):
    model = FastEmbedSparse(model_name=model_name)

    output = model.embed_query(
        "I must not fear. Fear is the mind-killer.",
    )

    assert len(output.indices) == len(output.values)
    assert np.allclose(output.values, np.ones(len(output.values)))

    quotes = [
        "I must not fear. Fear is the mind-killer.",
        "All animals are equal, but some animals are more equal than others.",
        "It was a pleasure to burn.",
        "The sky above the port was the color of television, tuned to a dead channel.",
        "In the beginning, the universe was created."
        "War is peace. Freedom is slavery. Ignorance is strength.",
        "We're not in Infinity; we're in the suburbs.",
        "I was a thousand times more evil than thou!",
    ]

    output = model.embed_documents(quotes)

    assert len(output) == len(quotes)

    for result in output:
        assert len(result.indices) == len(result.values)
        assert len(result.indices) > 0

    # Test support for unknown languages
    output = model.embed_query(
        [
            "привет мир!",
        ]
    )

    assert len(output.indices) == len(output.values)
    assert len(output.indices) == 2
