from langchain.prompts import ChatPromptTemplate

# HyDE document genration
template = """Please write a scientific paper passage to answer the question. Don't include references.
Question: {question}
Passage:"""
HYDE_PROMPT = ChatPromptTemplate.from_template(template)