from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import chromadb
import pandas
from pyarrow import csv
import tiktoken
import logging
import time
import concurrent.futures
import hashlib
import encodings
import numpy as np

MAX_RECORD_COUNT = 1000
MAX_THREAD_COUNT = 4
SRC_DATA_LOC = "source-data/issues.csv"
CAUSAL_LM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"#"Qwen/Qwen2.5-1.5B-Instruct"
#CAUSAL_LM_NAME = "ibm-granite/granite-4.0-h-350m"#"HuggingFaceTB/SmolLM2-135M-Instruct"
EMBEDDING_LM_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_LM_SEQ_LEN = 256
#EMBEDDING_LM_NAME = "jinaai/jina-embeddings-v2-small-en"
#EMBEDDING_LM_SEQ_LEN = 256
SUMMARY_PREFIX = "Summary:"
SUMMARY_PREFIX_LENGTH = len(SUMMARY_PREFIX)
app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s [%(levelname)s] [%(threadName)s/%(name)s] - %(message)s', level=logging.INFO)

logger.info(f"About to read source data from the location '{SRC_DATA_LOC}'")
dataset = csv.read_csv(SRC_DATA_LOC,
	parse_options=csv.ParseOptions(newlines_in_values=True),
	read_options=csv.ReadOptions(block_size=99999999))
logger.info("source data read successfully")

transformer = SentenceTransformer(EMBEDDING_LM_NAME)
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
tokenizer = AutoTokenizer.from_pretrained(CAUSAL_LM_NAME)
auto_causal_model = AutoModelForCausalLM.from_pretrained(CAUSAL_LM_NAME,
	device_map="auto")
#auto_causal_model.max_seq_length = 2000
chroma_client = chromadb.Client()

chunked_doc_collxn = chroma_client.create_collection("data-chunked")
doc_collxn = chroma_client.create_collection("data-full")

futures = []

def chunk_data(data, max_tokens=None):
	if max_tokens is None or max_tokens > EMBEDDING_LM_SEQ_LEN:
		max_tokens = EMBEDDING_LM_SEQ_LEN
	logger.info(f"Setting sequence length to {max_tokens}")
	encoding = tiktoken.get_encoding("gpt2")
	tokens = encoding.encode(data)
	chunks = []
	for counter in range(0, len(tokens), max_tokens):
		chunk_tokens = tokens[counter:counter + max_tokens]
		chunk_text = encoding.decode(chunk_tokens)
		chunks.append(chunk_text)
	return chunks

def gen_hash(user_input: str) -> str:
	h256 = hashlib.sha256()
	h256.update(user_input.encode(encodings.utf_8.getregentry().name))
	return h256.hexdigest()

def index_doc_into_collxn(data, index=None):
	embedding = transformer.encode(data)
	if index != None:
		logger.info(f"Indexing data#{index}")
		doc_collxn.add(embeddings=[embedding], ids=[str(index)], documents=[data])
	else:
		logger.info(f"Indexing new data's")
		doc_collxn.add(embeddings=[embedding], ids=[gen_hash(data)], documents=[data])

def index_chunked_doc_into_collxn(index, row_data):
	chunks = chunk_data(row_data)
	chunk_count = len(chunks)
	logger.info(f"Indexing row#{index} with {chunk_count} chunk(s)")
	source_embeddings = transformer.encode(chunks)
	if index == None:
		logger.info("Autogenerating id for this data")
		index = gen_hash(row_data)
	ids = []
	for counter in range(0, chunk_count):
		ids.append(f"{index}_{counter}")
	chunked_doc_collxn.add(embeddings=source_embeddings, documents=chunks, ids=ids)

def wait_for_indexing():
	for future in futures:
		logger.debug(f"Future {future.result()} completed")
	logger.info(f"Indexing completed in {time.perf_counter() - start_time:.3f} seconds")

# Entry point for now but to be moved out as a separate process.
start_time = time.perf_counter()

with concurrent.futures.ThreadPoolExecutor(MAX_THREAD_COUNT) as tp_executor:
	for index, row in dataset.to_pandas().iterrows():
		summary = row['summary']
		description = row['description']
		logger.info(f"Reading row#{index}")
		# Summary, and Description capture the essence of data involved.
		row_data = f"{summary}. {description}"
		futures.append(tp_executor.submit(index_doc_into_collxn, row_data, index))
		futures.append(tp_executor.submit(index_chunked_doc_into_collxn, index, row_data))
		logger.debug(f"Submitted the task to create embeddings for the row#{index}")
		if index == (MAX_RECORD_COUNT - 1):
			logger.info(f"Limiting the ingestion to {MAX_RECORD_COUNT} records")
			break
wait_for_indexing()

logger.info("Deleting the pyarrow table")
del dataset
logger.info("Ready to serve user queries now!")

@app.post("/summarize")
async def summarize(user_input: str, token_count: int = 500, top_p: float = 0.5, temperature: float = 0.7):
	logger.info(f"user_input is '{user_input}'")
	st = time.perf_counter()
	query_embedding = transformer.encode(user_input)
	chunked_search_results = chunked_doc_collxn.query(query_embeddings=[query_embedding], n_results=5)
	ids = chunked_search_results['ids'][0]
	logger.debug(f"ids from chunked_docs search are {ids}")
	ids = list(dict.fromkeys(id.split("_")[0] for id in ids))
	logger.debug(f"ids from chunked_docs search after deduplication are {ids}")
	full_search_results = doc_collxn.get(ids=ids)
	logger.debug(f"full search results are {full_search_results}")
	docs = full_search_results['documents']
	logger.debug(f"Docs are {docs}")
	scores = reranker.predict([(user_input, doc) for doc in docs])
	doc = docs[np.argmax(scores)]
	logger.info(f"Reranked doc is {doc}")
	in_tokens = tokenizer(build_model_prompt(doc, user_input), return_tensors="pt").to(auto_causal_model.device)
	out_tokens = auto_causal_model.generate(**in_tokens, max_new_tokens=token_count, top_p=top_p, temperature=temperature)
	out_text = tokenizer.decode(out_tokens[0], skip_special_tokens=True)
	logger.info(f"LLM reranked Summary is {out_text}")
	return {
		"result": {
			"nearest_doc": doc,
			"summary": out_text[out_text.find(SUMMARY_PREFIX) + SUMMARY_PREFIX_LENGTH:]
		},
		"metadata": {
			"time_taken_seconds": f"{time.perf_counter() - st:.3f}"
		}
	}

def build_model_prompt(contextual_data: str, query: str):
	logger.debug(f"Forming an LLM prompt using context:{contextual_data} and query:{query}")
	return f"""
		As a helpful AI assistant, your job is to 1. format and summarize the query with detailed requirements to the point using the Context provided. 2. not hallucinate and provide any detail outside this Context. 3. ensure your response is NOT truncated midway.

		Context: {contextual_data}

		Query: {query}

		Summary:
		"""

@app.post("/ingest")
async def ingest(user_input: str):
	index = gen_hash(user_input)
	index_doc_into_collxn(data=user_input, index=index)
	index_chunked_doc_into_collxn(index=index, row_data=user_input)
	return {"status": "data ingested successfully."}
