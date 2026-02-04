import hashlib
import encodings
import tiktoken
import logging
from config import settings
import json

logger = logging.getLogger(__name__)

try:
    with open("samples/samples.json") as file:
        sample_io_pairs = json.load(file)
        logger.info(f"Loaded {len(sample_io_pairs)} sample input-output pairs from samples/samples.json")
except Exception as e:
    logger.error(f"Failed to load sample_io_pairs from samples/samples.json: {e}")
    sample_io_pairs = []

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
    messages.append({"role": "system", "content": f"You're a Smart Backlog Assistant. Users expect you to elaborate, summarize, and format their Query into a JIRA-style ticket which must contain the following sections: 1. 'Description', 2. 'Acceptance Criteria', 3. 'Issue Priority', and 4. 'Issue Category' using the Context provided."})
    if sample_io_pairs:
        messages.append({"role": "user", "content": f"{sample_io_pairs[1]['input']}"})
        messages.append({"role": "assistant", "content": f"{sample_io_pairs[1]['output']}"})
        messages.append({"role": "user", "content": f"{sample_io_pairs[2]['input']}"})
        messages.append({"role": "assistant", "content": f"{sample_io_pairs[2]['output']}"})
    messages.append({"role": "user", "content": f"Context: {contextual_data}. Query: {query}"})
    return messages
