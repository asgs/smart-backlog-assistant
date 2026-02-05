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

def build_message(role: str, content: str) -> dict:
    return {"role": role, "content": content}

def build_system_message(content: str) -> dict:
    return build_message("system", content)

def build_user_message(content: str) -> dict:
    return build_message("user", content)

def build_assistant_message(content: str) -> dict:
    return build_message("assistant", content)

def build_chat_messages(contextual_data: str, query: str) -> str:
    logger.debug(f"Forming a chat prompt using context:{contextual_data} and query:{query}")
    messages = []
    messages.append(build_system_message(settings.CHAT_SYSTEM_PROMPT))
    # TODO - Restrict to only a few samples to avoid context explosion and rot.
    for sample_io_pair in sample_io_pairs:
        messages.append(build_user_message(sample_io_pair[settings.CHAT_INPUT]))
        messages.append(build_assistant_message(sample_io_pair[settings.CHAT_OUTPUT]))
    messages.append(build_user_message(f"{settings.CHAT_CONTEXT_PREFIX}: {contextual_data}. {settings.CHAT_QUERY_PREFIX}: {query}"))
    messages.append(build_assistant_message(""))
    return messages
