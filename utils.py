import hashlib
import encodings
import tiktoken
import logging
from config import settings

logger = logging.getLogger(__name__)

def gen_hash(user_input: str) -> str:
    h256 = hashlib.sha256()
    h256.update(user_input.encode(encodings.utf_8.getregentry().name))
    return h256.hexdigest()

def chunk_data(data: str, max_tokens: int = None) -> list[str]:
    if max_tokens is None or max_tokens > settings.EMBEDDING_LM_SEQ_LEN or max_tokens < 1:
        max_tokens = settings.EMBEDDING_LM_SEQ_LEN
        logger.debug(f"Resetting the sequence length to {max_tokens}")

    encoding = tiktoken.get_encoding(settings.TIKTOKEN_ENCODING)
    tokens = encoding.encode(data)
    chunks = []
    for counter in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[counter:counter + max_tokens]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def build_chat_messages(contextual_data: str, query: str) -> str:
    logger.debug(f"Forming a chat prompt using context:{contextual_data} and query:{query}")
    messages = []
    messages.append({"role": "system", "content": f"You're a Smart Backlog Assistant. Users expect you to reword, summarize, and format their queries into a nice JIRA-style ticket with fields such as 1. 'Description', 2. 'Acceptance Criteria', 3. 'Issue Priority', and 4. 'Issue Category' based on this Context - '{contextual_data}'"})
    messages.append({"role": "user", "content": f"{query}"})
    return messages
