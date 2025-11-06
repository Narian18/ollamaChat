from opensearchpy import OpenSearch

from config import config
from core.embed import embed_text


# Note: locally not using SSL so there's no `ca_certs_path`
client: OpenSearch = OpenSearch(
    hosts=[{"host": config.opensearch_host, "port": config.opensearch_port}],
    http_compress=True,  # enables gzip on request bodies
    http_auth=(config.opensearch_username, config.opensearch_password),
    use_ssl=False,
    verify_certs=False,
)


def create_embeddings_index(name: str) -> dict[str, str | bool]:
    """
    Creates a index (table) for storing chunks of text with their corresponding embedding, with a knn_vector column for the embeddings
    Remember: "index" is analogous to table here
    """
    index_body = {
        "settings": {"index.knn": True},
        "mappings": {
            "properties": {
                # These are effectively the columns / keys
                "embedding": {
                    "type": "knn_vector",
                    "dimension": config.embedding_length,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                    },
                },
                "chunk": {"type": "text"},
            },
        },
    }
    return client.indices.create(index=name, body=index_body)


def save_document(index_name: str, document: dict, refresh: bool = True):
    """
    Saves a document (json object) into the given index (table)

    "index" [noun] = { The top level "container" of "units" } ~= Table
    "document" = json object ~= SQL Row
    "index" [verb] = add to the index [noun]

    :param index_name: Name of the index to save the new document into
    :param document: The json object to be saved
    """
    return client.index(
        index=index_name,
        body=document,
        refresh=refresh,
    )


def save_embedding(index_name: str, text: str):
    embedding = embed_text(text)

    return save_document(index_name, {"embedding": embedding, "chunk": text})


def semantic_search(index_name: str, original_text: str, num_hits: int = 1):
    embedding = embed_text(original_text)

    query = {
        "size": num_hits,
        "query": {"knn": {"embedding": {"vector": embedding, "k": num_hits}}},
    }

    return client.search(index=index_name, body=query)
