from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from uuid import uuid4
from tqdm.auto import tqdm
from pincone import Pincone_vectorStore
import torch
from transformers.utils import is_flash_attn_2_available

index = Pincone_vectorStore()


df = load_dataset("json", data_files="./Data/formatted_test.jsonl", split="train")


Device = "cuda" if torch.cuda.is_available() else "cpu"
attn_implementation = "flash_attention_2" if (is_flash_attn_2_available()) and (torch.cuda.get_device_capability(0)[0] >= 8.0) else "sdpa"
embeddings = SentenceTransformer("all-mpnet-base-v2", device=Device)



class DataIngestion:
    def __init__(self, index, data, text_splitter=None, 
                 tokenizer = tiktoken.get_encoding("cl100k_base"),
                 embedding = SentenceTransformer("all-mpnet-base-v2"),
                batch_limit=100):
        self.index = index
        self.embedding = embedding
        self.tokenizer = tokenizer
        self.data = data 

        self.text_splitter = (text_splitter or RecursiveCharacterTextSplitter(chunk_size=400,
                                                                      chunk_overlap=20,
                                                                      length_function=self.token_length,
                                                                      separators=["\n\n", "\n", " ", ""]
                                                                      ))
        self.batch_limit = batch_limit

    def token_length(self, text):
        tokens = self.tokenizer.encode(text, disallowed_special=())
        return len(tokens)
        

    def split_text_metadata(self, page):
        contents = self.text_splitter.split_text(page["formatted"])
        metadatas = [
        {"chunks": j, "text": content}
        for j, content in enumerate(contents)
        ]

        return contents, metadatas
        
    def upload_batch(self, texts, metadatas):
        ids = [str(uuid4()) for _ in range (len(texts))]
        embeddings = self.embedding.encode(texts)
        self.index.upsert(vectors=zip(ids, embeddings, metadatas))

    def batch_upload(self):
            batch_texts = []
            batch_metadatas = []
            for page in tqdm(self.data):
                contents, metadatas = self.split_text_metadata(page)
                batch_texts.extend(contents)
                batch_metadatas.extend(metadatas)
                if len(batch_texts) >= self.batch_limit:
                    self.upload_batch(batch_texts, batch_metadatas)
                    batch_texts = []
                    batch_metadatas = []
    
            if len(batch_texts) > 0:
                self.upload_batch(batch_texts, batch_metadatas)

data_ingestion = DataIngestion(index, data=df, embedding=embeddings)
data_ingestion.batch_upload()
