from langchain.prompts import ChatPromptTemplate

# Multi Query: Different Perspectives
template = """You are an expert question translator. Your task is to generate {number_of_queries} different versions of a given user question.
By generating multiple perspectives of the user question you will help the user understand the context of the question.
There are guidelines to follow:
-Do not return any other text apart from the reformatted questions.
-Provide JUST the alternative questions separated by newlines.
-Start your answer immediately with the first reformatted question.
Original question: {question}
Output ({number_of_queries} queries):"""

MULTI_QUERY_PROMPT = ChatPromptTemplate.from_template(template)


def get_individual_queries(response):
    individual_queries = response.split("\n")
    individual_queries = list(filter(None, individual_queries))
    return individual_queries
