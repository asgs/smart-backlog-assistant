import os

class Settings:
    def __init__(self):
        self.MAX_RECORD_COUNT = int(os.getenv("MAX_RECORD_COUNT", 100))
        self.MAX_THREAD_COUNT = int(os.getenv("MAX_THREAD_COUNT", 1))
        self.SRC_DATA_LOC = os.getenv("SRC_DATA_LOC", "source-data/issues.csv")

        # Model Config
        self.CAUSAL_LM_NAME = os.getenv("CAUSAL_LM_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
        self.CAUSAL_LM_REVISION = os.getenv("CAUSAL_LM_REVISION", "7ae557604adf67be50417f59c2c2f167def9a775")
        # Kept to experiment on other lightweight and/or accurate models.
        #self.CAUSAL_LM_NAME = os.getenv("CAUSAL_LM_NAME", "google/gemma-3-1b-it")
        #self.CAUSAL_LM_REVISION = os.getenv("CAUSAL_LM_REVISION", "dcc83ea841ab6100d6b47a070329e1ba4cf78752")
        self.EMBEDDING_LM_NAME = os.getenv("EMBEDDING_LM_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.EMBEDDING_LM_SEQ_LEN = int(os.getenv("EMBEDDING_LM_SEQ_LEN", 256))
        self.RERANKER_LM_NAME = os.getenv("RERANKER_LM_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.TIKTOKEN_ENCODING = os.getenv("TIKTOKEN_ENCODING", "gpt2")
        self.ASSISTANT_PREFIX = os.getenv("ASSISTANT_PREFIX", "assistant")

        # Summary and Description are good enough to store now and query later.
        self.JIRA_CSV_COLUMNS = os.getenv("JIRA_CSV_COLUMNS", "summary,description").split(",")

        # Update status every N records for brevity purposes.
        self.STATUS_UPDATE_STEP = int(os.getenv("STATUS_UPDATE_STEP", 10))
        # Prompt names
        self.DOCUMENT_PROMPT_NAME = os.getenv("DOCUMENT_PROMPT_NAME", "document")
        self.QUERY_PROMPT_NAME = os.getenv("QUERY_PROMPT_NAME", "query")

settings = Settings()
