from pinecone import pinecone
#import pinecone
import os
from langchain_pinecone import PineconeSparseVectorStore
from pinecone import Pinecone
from  dotenv import load_dotenv
load_dotenv()


def Pincone_vectorStore(index_name ="nairs-research"):
    vector_db = os.getenv("Pinecone_api_key")
    pc = Pinecone(api_key=vector_db)

    # if index_name not in pc.list_indexes().names():
    #     pc.create_index(
    #         name= index_name,
    #         metric= "cosine",
    #         dimension = 768,
    #         spec= pinecone.ServerlessSpec(
    #             cloud = "aws",
    #             region = "us-east-1"
    #         )
    #     )

    index = pc.Index(index_name)
    print(f'Successfully connect to {index}')
    return index

index = Pincone_vectorStore()
print(index.describe_index_stats())



