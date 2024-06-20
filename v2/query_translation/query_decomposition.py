from langchain.prompts import ChatPromptTemplate

# Decomposition
template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n
The goal is to break down the input into a set of sub-problems / sub-questions that can be answers in isolation. \n
Generate multiple search queries related the question provided.
There are guidelines to follow:
-Do not return any other text apart from the reformatted questions.
-Provide JUST the alternative questions separated by newlines.
-Start your answer immediately with the first reformatted question.
Original question: {question}
Output ({number_of_queries} queries):"""
DECOMPOSITION_PROMPT = ChatPromptTemplate.from_template(template)


def get_individual_queries(response):
    individual_queries = response.split("\n")
    individual_queries = list(filter(None, individual_queries))
    return individual_queries
