from prompts import quality_grader_prompt
from langchain_core.output_parsers import JsonOutputParser


def create_quality_grader(llm_model):
    quality_grader = quality_grader_prompt | llm_model | JsonOutputParser()
    return quality_grader
