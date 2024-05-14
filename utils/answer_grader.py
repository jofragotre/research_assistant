from prompts import answer_grader_prompt
from langchain_core.output_parsers import JsonOutputParser


def create_answer_grader(llm_model):
    answer_grader = answer_grader_prompt | llm_model | JsonOutputParser()
    return answer_grader
