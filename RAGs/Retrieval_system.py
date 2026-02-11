from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

class Retrieval:
    def __init__(self, device, index, Embeddings, vector_store):
        self.index = index
        self.device = device
        self.Embeddings = Embeddings
        self.vector_store = vector_store


    def get_retrieval(self):

        template = """<s>[INST] <<SYS>>
        You are an AI Intelligent Tutoring System. Analyze the error in <thinking> tags, then provide a hint.
        <</SYS>>

        Q: {Context}
        My Answer: {answer} [/INST]
        <thinking>"""
        prompt = PromptTemplate.from_template(template)
        retriever = self.vector_store.as_retriever()
        return retriever, prompt
 