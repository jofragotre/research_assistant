from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from multi_query import MULTI_QUERY_PROMPT, get_individual_queries
from hyde import HYDE_PROMPT
from query_decomposition import DECOMPOSITION_PROMPT


def test_hyde_query():
    print(f"Hyde example {HYDE_PROMPT}")
    llm = ChatOllama(model="llama3", temperature=0)
    chain = HYDE_PROMPT | llm | StrOutputParser()

    new_query = chain.invoke({"question": "What is the name of the person that is in charge of the united kingdom?"})
    print(new_query)
    assert new_query


def test_get_multi_query():
    print(f"Multi query example {MULTI_QUERY_PROMPT}")
    llm = ChatOllama(model="llama3", temperature=0)
    chain = MULTI_QUERY_PROMPT | llm | StrOutputParser() | RunnableLambda(get_individual_queries)

    new_queries = chain.invoke({"number_of_queries": "four",
                                "question": "What is the name of the person that is in charge of the united kingdom?"})
    assert len(new_queries) == 4


def test_query_decomposition():
    print(f"Query decomposition example {DECOMPOSITION_PROMPT}")
    llm = ChatOllama(model="llama3", temperature=0)
    chain = DECOMPOSITION_PROMPT | llm | StrOutputParser() | RunnableLambda(get_individual_queries)

    new_queries = chain.invoke({"number_of_queries": "three",
                                "question": "What are the main components of an LLM-powered autonomous agent system?"})
    print(new_queries)
    assert len(new_queries) == 3
