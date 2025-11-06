import os

from opensearchpy import RequestError

from core.chunk import chunk_markdown_by_headings
from config import config
from core.vectordb import create_embeddings_index, save_embedding


def setup():
    try:
        # NOTE: Embedding the rust book took 2m15s
        # Create index (table)
        create_embeddings_index(config.index_name)
        # Save the knowledge-base (folder "kb") in the vector db
        for filename in os.listdir("kb"):
            if filename.endswith(".md"):
                with open(f"kb/{filename}") as f:
                    chunks = chunk_markdown_by_headings(f.read())

                for chunk in chunks:
                    save_embedding(config.index_name, chunk)
    except RequestError as e:
        if e.error != "resource_already_exists_exception":
            raise


if __name__ == "__main__":
    setup()
