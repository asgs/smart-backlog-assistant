import time
import numpy as np
import logging
import sys
from config import settings
from utils import build_chat_messages
from core.model_manager import model_manager
from core.vector_db import vector_db
from request_models import SummarizeRequest

sys.tracebacklimit = 2 # Makes the traces less verbose.
logger = logging.getLogger(__name__)
show_encoding_progress = False
if logger.isEnabledFor(logging.DEBUG):
    show_encoding_progress = True

class SummarizerService:

    def query_vector_db(self, user_input: str, top_k: int) -> list[str]:
        query_embedding = model_manager.transformer.encode(user_input, prompt_name="document", show_progress_bar=show_encoding_progress)
        chunked_search_results = vector_db.query_chunks(query_embeddings=[query_embedding], n_results=top_k)
        logger.debug(f"chunked_search_results are {chunked_search_results}")
        ids = chunked_search_results['ids'][0]
        logger.debug(f"ids from chunked_docs search are {ids}")

        parent_ids = list(dict.fromkeys(id.split("_")[0] for id in ids))
        logger.debug(f"ids from chunked_docs search after deduplication are {parent_ids}")

        full_search_results = vector_db.get_full_docs(ids=parent_ids)
        logger.debug(f"full search results are {full_search_results}")
        docs = full_search_results['documents']
        logger.debug(f"Docs are {docs}")

        return docs
    
    def rerank_docs(self, user_input: str, docs: list[str]) -> list[str]:
        scores = model_manager.reranker.predict([(user_input, doc) for doc in docs], show_progress_bar=show_encoding_progress)
        best_doc_idx = np.argmax(scores)
        doc = docs[best_doc_idx]
        logger.debug(f"Reranked doc is {doc}")

        return doc

    def query_lm(self, user_input: str, doc: str, token_count: int, top_p: float, temperature: float) -> str:
        messages = build_chat_messages(doc, user_input)
        prompt = model_manager.tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False, continue_final_message=True)
        in_tokens = model_manager.tokenizer(prompt, return_tensors="pt")

        out_tokens = model_manager.causal_model.generate(
            **in_tokens,
            max_new_tokens=token_count,
            top_p=top_p,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4
        )
        out_text = model_manager.tokenizer.decode(out_tokens[0], skip_special_tokens=True)
        logger.info(f"LLM reranked Summary is {out_text}")

        return out_text

    def extract_summary(self, response: str) -> str:
        summary_start = response.rfind(settings.ASSISTANT_PREFIX)
        if summary_start != -1:
            summary = response[summary_start + len(settings.ASSISTANT_PREFIX):].strip()
        else:
            summary = response
        return summary

    def build_error_summary_dict(self, st: float) -> dict:
        return self.build_summary_dict(None, "Unable to summarize the given query.", st)

    def build_summary_dict(self, nearest_doc: str, summary: str, st: float) -> dict:
        return {
            "result": {
                "nearest_doc": nearest_doc,
                "summary": summary
            },
            "metadata": {
                "time_taken_seconds": f"{time.perf_counter() - st:.3f}"
            }
        }

    async def summarize(self, request: SummarizeRequest) -> dict:
        user_input = request.user_input
        top_k = request.top_k
        logger.info(f"user_input is '{user_input}'")
        st = time.perf_counter()

        try:
            docs = self.query_vector_db(user_input, top_k)

            if not docs:
                logger.error("No docs found in the vector DB corresponding to the user input.")
                return self.build_error_summary_dict(st)

            doc = self.rerank_docs(user_input, docs)

            response = self.query_lm(user_input, doc, request.token_count, request.top_p, request.temperature)

            summary = self.extract_summary(response)

            return self.build_summary_dict(doc, summary, st)
        except Exception as e:
            logger.error(f"Failed to summarize. ", exc_info=True)
            return self.build_error_summary_dict(st)

summarizer_service = SummarizerService()
