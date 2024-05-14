from prompts import hallucination_grader_prompt
from langchain_core.output_parsers import JsonOutputParser


def create_hallucination_grader(llm_model):
    hallucination_grader = hallucination_grader_prompt | llm_model | JsonOutputParser()
    return hallucination_grader