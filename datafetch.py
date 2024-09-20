import os
from typing import List, Optional
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

# Constants
DOCS_DIR = "./docs/"
DB_DIR = "./vector_db/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

class DocumentProcessor:
    def __init__(self, docs_dir: str, db_dir: str):
        self.docs_dir = docs_dir
        self.db_dir = db_dir
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    def load_documents(self) -> List[str]:
        if not os.path.exists(self.docs_dir):
            raise FileNotFoundError(f"Directory '{self.docs_dir}' does not exist.")
        
        files = [f for f in os.listdir(self.docs_dir) if f.endswith('.txt')]
        if not files:
            raise ValueError(f"No .txt files found in '{self.docs_dir}'.")
        
        loader = DirectoryLoader(self.docs_dir, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        return [doc.page_content for doc in documents]

    def split_documents(self, documents: List[str]) -> List[str]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        return text_splitter.split_text("\n\n".join(documents))

    def create_vector_store(self, splits: List[str]) -> Optional[Chroma]:
        if not splits:
            print("Error: No text splits to create vector store.")
            return None
        
        vector_store = Chroma.from_texts(
            texts=splits,
            embedding=self.embeddings,
            persist_directory=self.db_dir
        )
        return vector_store

    def process(self):
        try:
            print("Loading documents...")
            documents = self.load_documents()
            print(f"Loaded {len(documents)} documents.")

            print("Splitting documents...")
            splits = self.split_documents(documents)
            print(f"Created {len(splits)} splits.")

            print("Creating vector store...")
            vector_store = self.create_vector_store(splits)
            if vector_store:
                print(f"Vector store created with {vector_store._collection.count()} embeddings.")
                print("Vector store persisted to:", self.db_dir)
            else:
                print("Failed to create vector store.")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

def main():
    processor = DocumentProcessor(DOCS_DIR, DB_DIR)
    processor.process()

if __name__ == "__main__":
    main()