from Retrieval_system import Retrieval
from langchain_ollama.llms import OllamaLLM
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import torch
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from pincone import Pincone_vectorStore
import warnings
from operator import itemgetter
from langchain_core.output_parsers import StrOutputParser


warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


Device = "cuda" if torch.cuda.is_available else "cpu"
index= Pincone_vectorStore()
Embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2", model_kwargs={"device": Device})
vector_store = PineconeVectorStore(embedding=Embeddings,index=index)
retrieval_obj = Retrieval(device=Device, index=index,Embeddings=Embeddings,vector_store=vector_store)
retrieval, prompt = retrieval_obj.get_retrieval()





# model = OllamaLLM(model="llama2:7b",
#                   temperature=0.7)
# chain = (
#     {
#         "context": itemgetter("question") | retrieval,
#         "question": itemgetter("question"),
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )

# result=chain.invoke({"question": "What is Temperature"})
# print(result)



docs = retrieval.get_relevant_documents("Principle of Relativity")
for i, doc in enumerate(docs):
   print(f'\nResult{i}:\n{doc.page_content}')

# USER = "Tell me the principle of relativity"
# Prompt = f"""[SYSTEM] You are an AI Tutor. Identify the student's misconception and provide a helpful hint.
# Q: {USER}"""

