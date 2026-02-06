from diagrams import Diagram, Cluster, Edge
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.onprem.database import Clickhouse # Using as placeholder for VectorDB/Chroma if specific icon unavailable
from diagrams.programming.framework import Fastapi, React
from diagrams.programming.language import Python
from diagrams.generic.storage import Storage
from diagrams.generic.place import Datacenter

# Adjust graph_attr to control layout
graph_attr = {
    "fontsize": "24",
    "bgcolor": "white",
    "splines": "spline", # "ortho" can sometimes cause bent lines, "spline" is smoother, "polyline" is straight segments
    "rankdir": "LR", # Left to Right flow is often cleaner for data pipelines
    "pad": "0.5"
}

with Diagram("Smart Backlog Assistant", show=False, filename="architecture_diagram", outformat="png", graph_attr=graph_attr):
    
    user = User("User")
    
    with Cluster("Data Sources"):
        csv_source = Storage("CSV Data")
    
    ui = React("Frontend (React)")
        
    with Cluster("Backend Services"):
        api = Fastapi("REST API Layer")
        
        with Cluster("Service Layer"):
            ingestion_svc = Python("Ingestion\nService")
            summarizer_svc = Python("Summarizer\nService")
            
        with Cluster("Core Components"):
            model_manager = Python("Model\nManager")
            
            with Cluster("Models"):
                embedding_model = Server("Embedding\nModel")
                reranker_model = Server("Reranker")
                llm = Server("LLM (Causal)")
                
            vector_db = Clickhouse("Vector DB\n(Chroma)")

    # Data Flow
    
    # Ingestion Path
    csv_source >> Edge(label="Raw Data") >> ingestion_svc
    ingestion_svc >> Edge(label="2. Generate\nEmbeddings") >> model_manager
    model_manager >> Edge(label="2. Compute\nEmbedding") >> embedding_model
    ingestion_svc >> Edge(label="3. Store\nVectors") >> vector_db
    
    # User Query Path
    user >> Edge(label="Interacts") >> ui
    ui >> Edge(label="API Calls") >> api
    
    api >> Edge(label="1. Trigger\n/ingest") >> ingestion_svc
    api >> Edge(label="1. Request\n/summarize") >> summarizer_svc
    
    summarizer_svc >> Edge(label="2. Embed\nQuery") >> model_manager
    summarizer_svc >> Edge(label="3. Query") >> vector_db
    vector_db >> Edge(label="4. Results") >> summarizer_svc
    
    summarizer_svc >> Edge(label="5. Rerank") >> model_manager
    model_manager >> Edge(label="5. Score\nCandidates") >> reranker_model
    
    summarizer_svc >> Edge(label="6. Generate\nSummary") >> model_manager
    model_manager >> Edge(label="6. Generate\nText") >> llm
    
    summarizer_svc >> Edge(label="7. Response") >> api
