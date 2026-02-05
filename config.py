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
        #self.CAUSAL_LM_NAME = os.getenv("CAUSAL_LM_NAME", "Qwen/Qwen3-0.6B")
        #self.CAUSAL_LM_REVISION = os.getenv("CAUSAL_LM_REVISION", "c1899de289a04d12100db370d81485cdf75e47ca")
        #self.CAUSAL_LM_NAME = os.getenv("CAUSAL_LM_NAME", "google/gemma-3-1b-it")
        #self.CAUSAL_LM_REVISION = os.getenv("CAUSAL_LM_REVISION", "dcc83ea841ab6100d6b47a070329e1ba4cf78752")
        self.EMBEDDING_LM_NAME = os.getenv("EMBEDDING_LM_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.EMBEDDING_LM_SEQ_LEN = int(os.getenv("EMBEDDING_LM_SEQ_LEN", 256))
        self.RERANKER_LM_NAME = os.getenv("RERANKER_LM_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

        self.TIKTOKEN_ENCODING = os.getenv("TIKTOKEN_ENCODING", "o200k_base")
        self.ASSISTANT_PREFIX = os.getenv("ASSISTANT_PREFIX", "assistant")

        # Summary and Description are good enough to store now and query later.
        self.JIRA_CSV_COLUMNS = os.getenv("JIRA_CSV_COLUMNS", "summary,description").split(",")

        # Update status every N records for brevity purposes.
        self.STATUS_UPDATE_STEP = int(os.getenv("STATUS_UPDATE_STEP", 10))

        # chat template constants
        self.CHAT_CONTEXT_PREFIX = os.getenv("CHAT_CONTEXT_PREFIX", "Context:")
        self.CHAT_QUERY_PREFIX = os.getenv("CHAT_QUERY_PREFIX", "Query:")
        self.CHAT_INPUT = os.getenv("CHAT_INPUT", "input")
        self.CHAT_OUTPUT = os.getenv("CHAT_OUTPUT", "output")
        self.CHAT_SYSTEM_PROMPT = os.getenv("CHAT_SYSTEM_PROMPT", "You're a Smart Backlog Assistant. Users expect you to elaborate, summarize, and format their Query into a JIRA-style ticket which must contain the following sections: 1. 'Description', 2. 'Acceptance Criteria', 3. 'Issue Priority', and 4. 'Issue Category' and not contain any tildes or code snippets using the Context provided.")

settings = Settings()
