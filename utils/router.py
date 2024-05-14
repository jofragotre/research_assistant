from prompts import router_prompt
from langchain_core.output_parsers import JsonOutputParser


def create_router(llm_model):
    question_router = router_prompt | llm_model | JsonOutputParser()
    return question_router
