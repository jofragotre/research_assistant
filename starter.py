import time
import logging
import sys
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, SummaryIndex
# from llama_index.embeddings.ollama import OllamaEmbedding
# from llama_index.llms.ollama import Ollama
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.tools import QueryEngineTool
# from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
# from llama_index.core.selectors import LLMSingleSelector


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

documents = SimpleDirectoryReader("data").load_data()
documents = list(map(lambda x: x.to_langchain_format(), documents))
print(documents[0])

quit()

# nomic embedding model
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# ollama
Settings.llm = Ollama(model="llama3", request_timeout=1000)

splitter = SentenceSplitter(chunk_size=512, chunk_overlap=64)
nodes = splitter.get_nodes_from_documents(documents)

print(f"Created nodes {len(nodes)}")

t0 = time.perf_counter()
summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)
t1 = time.perf_counter()
print(f"Created index {t1-t0}")

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to MetaGPT"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the MetaGPT paper."
    ),
)


query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[
        summary_tool,
        vector_tool,
    ],
    verbose=True
)

print("Waiting for query...")
response = query_engine.query("Can you summarize the improvements from the most recent paper in relation with the first TALL paper?")
print(response)