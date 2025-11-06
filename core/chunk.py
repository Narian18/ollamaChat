from config import config


SENTENCE_ENDERS = [
    ". ",
    "? ",
    "! ",
    ".\n",
    "?\n",
    "!\n",
    ".\n ",
    "?\n ",
    "!\n ",
    ". \n",
    "? \n",
    "! \n",
]


def _is_end_of_sentence(word: str) -> bool:
    """This is naive. It checks for the punctuation mark and a space after (to avoid "3.1")"""
    for ender in SENTENCE_ENDERS:
        if word.endswith(ender):
            return True

    return False


def _check_for_table(chunk: str, remaining: list[str]) -> int | None:
    """Returns None if no table, otherwise returns the index of the end of the table"""
    if "|\n" not in chunk:
        return
    try:
        return remaining.index("|\n\n")
    except ValueError:
        return


# FIXME: This will break if the table exceeds 2048 tokens. A proper strategy has to actually break up tables if you can't split them
#  i.e.:
#   1. Attempt to complete the table in the chunk
#   2. If you can't complete the table in the chunk, end the chunk before the table and add the table separately
#   3. If the table itself is longer than the token limit, split the table by a newline once you get past the TOKEN_TARGET
#  You should be able to do 2 and 3 in the same function, something like `chunk_table(words: list[str], chunks: ChunksList, starting_markdown: str | None = None)`
def chunk_text_preserving_sentences(text: str) -> list[str]:
    """
    Splits the text into chunks of `TOKEN_TARGET` tokens, making sure not to break the middle of a sentence, keep tables intact
    """
    # ToDo: Overlap a bit
    chunks: list[str] = []

    current_chunk = ""
    word_count = 0
    # hack to force splits on "\n" as well as whitespace, without losing the \n in the chunk
    text = text.replace("\n", "\n ")
    split_text = text.split(" ")
    while split_text:
        # need to add the space back because we split on it
        word = split_text.pop(0) + " "
        word_count += 1
        current_chunk += word

        if (word_count > config.token_target) and _is_end_of_sentence(word):
            # hit a big enough chunk
            table_end_idx = _check_for_table(current_chunk, split_text)
            if table_end_idx:
                # Continue until the table ends
                current_chunk += " ".join(split_text[:table_end_idx])
                del split_text[:table_end_idx]

            chunks.append(current_chunk)
            current_chunk = ""  # reset chunk
            word_count = 0
        elif word_count >= config.max_tokens:
            # failsafe - ensure we never exceed the max
            chunks.append(current_chunk)
            current_chunk = ""  # reset chunk
            word_count = 0
    else:
        # last iteration
        chunks.append(current_chunk)

    return chunks


# FIXME: tables
def chunk_markdown_by_headings(document_text: str, heading_level: int = 1) -> list[str]:
    """
    Intention here is to chunk the markdown into blocks under H1s. Embeddinggemma can handle up to 2048 tokens
    Attempt #1 will chunk on headings, and use a recursive strategy:
    1. Split into H1s
    2. If that chunk is too large, split into H2s
    3. Apply recursively until the chunk size is less than TOKEN_TARGET
    This function maps the embedding vector directly to its text contents. Obviously this is stupid, but it sets
     you up to store it somewhere, e.g. in a db
    """
    chunks: list[str] = []

    delimiter = "\n" + "#" * heading_level + " "
    split_by_headings = document_text.split(delimiter)
    for block in split_by_headings:
        if len(block) == 0:
            return []
        elif block.count(" ") > config.token_target:
            # If the chunk at this heading level is too large, drop down to the next heading level, and keep going until the chunk is smaller
            #  Note that markdown bottoms-out at h6, so if it's still too large, break up the paragraph
            if heading_level > 6:
                return chunks + chunk_text_preserving_sentences(block)
            else:
                return chunks + chunk_markdown_by_headings(
                    block, heading_level=heading_level + 1
                )

        if not block.startswith(delimiter):
            # we have to add the delimiter back on because `split` drops it
            block = delimiter + block

        chunks.append(block)

    return chunks
