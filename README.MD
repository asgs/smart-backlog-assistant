# Overview
This project is an attempt to build an AI ~~Agent~~ **Chatbot** that will read Jira ticket descriptions for new tickets and summarize the contents in a neat and concise way based on data ingested in the past

# Problem Definition
Users discuss product and customer requirements during meetings, brainstorming discussions, and other virtual means. These discussions are not formally documented and may be lacking a few finer details which may be required to build a product. These details are to be captured in Jira tickets with all the required information for the Development, and QA teams to implement and verify the expected behavior of the features. Plus, every organization has Jira or a similar system which has tickets created for every feature request, bug report, etc. in the past which are available for the Development and QA teams to refer to.

The Goal of this project is to utilize AI technologies to read through existing Jira tickets data, understand the context, and patterns, and the format of the data to and let users key in new requirements or bug descriptions and summarize the contents in a neat and concise way. It will help reduce the effort and time required by the Users to gather the necessary details and build something that could be reviewed before finalizing the ticket

# Solution Design

## Tech Stack
- Python3
- FastAPI (with uvicorn to expose REST APIs)
- sentence-transformers (with MiniLM-L6-v2 for embedding purposes)
- transformers (with Qwen2.5-0.5B-Instruct for Instruct purposes)
- pandas (for CSV data manipulation)
- pyarrow (for efficient CSV reading)
- ChromaDB (for vector storage and retrieval)
- tiktoken (for token en/decoding)
- ReactJS (for frontend)
- Babel (for JSX)

## Overall Architecture / Data Flow Visualization
Thanks to [Mermaid.ai](https://mermaid.ai/d/7c698629-39e6-446f-bb39-2725ee697de4), you can visualize the overall Architecture and Component Interaction Diagram
![Architecture and Component Interaction Diagram](arch-diagram-with-interactions.png "Architecture Diagram")

## Key Components
- REST API
    - Exposes APIs to ingest data
    - Exposes API to summarize content
- Ingestion Service
    - Reads data from CSV files
    - Generates embeddings for the data
    - Stores embeddings in ChromaDB
- Summarization Service
    - Generates embeddings for the query
    - Retrieves similar embeddings from ChromaDB
    - Generates summary using Instruct LM
- Model Manager
    - Manages loading and unloading of models
    - Provides embeddings for the data
- Vector DB
    - Manages collections in ChromaDB
    - Persists into and loads embeddings from ChromaDB
- Config
    - Centralized static configuration to tweak various settings including but not limited to model names, data locations, etc

## Design Considerations
- Use a smaller LM (like MiniLM) for embedding purposes while a larger LM (like Qwen2.5) for instruct/inference purposes
- Host the small-to-mid-sized LMs locally to reduce cost while trading-off speed/latency
- Switch/swap the LMs and other components easily by updating the Config without impacting the overall system
- Build a mini-RAG system using a Vector Store (like ChromaDB) to build the knowledge-base
- Divide data into smaller chunks to avoid overflowing the context window
- Maintain two collections in the Vector Store - one for full document contents and one for chunked document contents
- Include sample inputs and expected outputs within the prompt (few-shot prompting) to guide the LM as against fine-tuning the model which may take more time and resources in favour of accuracy and correctness
- Use a reranker model to rerank the retrieved documents to improve the quality of the summary
- Make the parameters like temperature, and top-p configurable while choosing reasonable defaults to balance creativity and coherence

# Implementation
## App Layer
1. Exposes REST APIs for ingestion and summarization
2. Exposes a GUI for user interaction

## Service Layer
### Ingestion Service
1. Reads data from CSV files / User inputs
2. Generates embeddings for the data
3. Stores embeddings in ChromaDB along with data
### Summarization Service
1. Generates embeddings for the query
2. Retrieves similar embeddings from ChromaDB
3. Retrieves full documents corresponding to the embeddings
4. Reranks the retrieved documents
5. Builds the Chat Template message using the Reranked documents as the Context, along with example input and output pairs to guide the Instruct LM
6. Sends the Chat Template message to the Instruct LM to generate the summary

## Model Layer
Manages the models below

1. Embedding Model
2. Reranker Model
3. Instruct Model

# Data Setup
This agent depends on external data sources to build the knowledge base required to augment the responses from the LM. Currently, two publicly accessible data sources are tried from Kaggle

- [Apache JIRA Issues](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues?select=issues.csv)
- [Jira Dataset](https://www.kaggle.com/datasets/cesaranasco/jira-dataset)

whichever dataset is used, the expectation here is that it's [a CSV file](config.py#L7) with the respective CSV [column names](config.py#L25) to be used for ingestion.

# How to Run?
1. Install python3
2. Clone this git repo
3. Navigate to the root of the repo and setup a virtual environment
    1. Run `python3 -m venv .jts` to create a virtual env
    2. Run `source .jts/bin/activate` to activate/switch to this virtual env
4. Run `pip install -r requirements.txt` to setup the required dependencies. You may have to upgrade pip in case you encounter an `AssertionError`
5. Download the data sources mentioned in the [previous section](#data-setup) and unzip to the source-data directory
6. Run `uvicorn main:app` to ingest sample data and spin up the FastAPI ASGI server

Once the ingestion completes, the app is ready with a [GUI](http://localhost:8000/ui/index.html) providing a screen like below. ![ingestion and summarization features](ui-home-sample.png "UI Home Page")

# Testing

1. To Ingest a new document, run the [ingest](test-scripts/ingest.py) script
2. To Summarize a query, run the [summarize](test-scripts/summarize.py) script

# Usage of AI tools

well, isn't fascinating to use AI tools to build AI-powered applications? I used a variety of AI tools to build this application right from understanding the problem to ideation, design, implementation, and finally Architecture diagramming

- ChatGPT 5.2 Go for many basic and intermediate level Q&A
- GPT4All for local LLM inference
- Llama, Google Gemini 3 Pro for advanced Q&A and code generation
- Google Antigravity Agent for code generation, review, bug fixes, and debugging, and diagramming

In addition, I read and learnt from a lot of blogs, articles, and documentation to understand the problem and build this application including but not limited to IBM Watsonx AI, Huggingface, FastAPI, VentureBeat, and ChromaDB documentation

# TODOs
1. [Done] - ~~Build a basic GUI for user to submit their queries and view the responses returned~~
2. [Done] - ~~Provide an option to user to retrieve top-k results. The default is 3 now~~
3. [**WIP**] - Provide an endpoint to ingest one/more documents or reingest one/more datasources on demand
4. [Done] - ~~Move the local LM name, dataset location, CSV column names to a configuration file~~
5. [**WIP**] - Model Fine/Prompt-tuning
6. [Done] - ~~Configure ChromaDB to use hnsw indexing~~

# Future Enhancements
1. Provide options to switch to a cloud-hosted LM accepting necessary configs like URL, Auth Token/API Key, etc
2. Introduce admin, and user roles and implement Authn and Authz to beef up security
3. Provide an endpoint to save/load pre-trained models on demand
4. Quantize the Models to achieve a balance between speed and accuracy and reduce memory footprint
5. Introduce an "Agent Routing" layer to route processing to appropriate tool/agent based on the request
6. Build Data Integration layer to read and ingest data from external sources with different data formats