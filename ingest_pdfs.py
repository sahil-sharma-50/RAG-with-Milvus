"""
This script processes PDFs by extracting text, splitting the text into chunks,
and storing them in the milvus vector store.
"""

import hashlib
import os
import shutil
import sys
from typing import List

from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    Index,
    connections,
    utility,
)

DATA_PATH = "New_PDFs"
DESTINATION_PATH = "All_PDFs"
os.makedirs(DESTINATION_PATH, exist_ok=True)

os.environ["OPENAI_API_KEY"] = "sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# Define connection parameters
MILVUS_URI = "./milvus_db.db"
MILVUS_HOST = "127.0.0.1"
MILVUS_PORT = "19530"
COLLECTION_NAME = "rag_model"


def connect_to_milvus():
    """Connect to Milvus server"""
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)


schema = CollectionSchema(
    fields=[
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=False,
        ),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024),
    ],
    description="Schema for storing PDF vectors and metadata.",
)


def start_ingest():
    """Process PDFs and add their content (text and images) to Milvus vector store."""

    pdf_files = [
        filename for filename in os.listdir(DATA_PATH) if filename.endswith(".pdf")
    ]

    if not pdf_files:
        print("📁 No PDF files found in the directory.")
        return

    for idx, filename in enumerate(pdf_files, start=1):
        # Extract metadata from the filename

        pdf_path = os.path.join(DATA_PATH, filename)
        print(f"Processing PDF '{filename[:50]}': {idx}/{len(pdf_files)}")

        # Load and split documents
        documents = load_documents(pdf_path)
        chunks = split_documents(documents)

        # Initialize Milvus vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vectorstore = Milvus(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            connection_args={"uri": MILVUS_URI},
        )

        # Get the content-based hash for the PDF
        pdf_content = get_pdf_content(pdf_path)
        pdf_hash = generate_pdf_hash(pdf_content)

        chunks_with_ids = calculate_chunk_ids(chunks, pdf_hash)

        # Retrieve existing document IDs from the vector store
        existing_ids = []
        results = collection.query(
            expr=f"pdf_hash == '{pdf_hash}'", output_fields=["pdf_hash"]
        )
        existing_ids = [result["id"] for result in results] if results else []

        # Identify new chunks to be added
        new_chunks = [
            chunk
            for chunk in chunks_with_ids
            if chunk.metadata["id"] not in existing_ids
        ]

        if new_chunks:
            has_new_pdf = True

            # Add new content (text and images) to Milvus
            add_to_milvus(vectorstore, new_chunks, pdf_hash)

            destination_path = os.path.join(DESTINATION_PATH, filename)
            shutil.move(pdf_path, destination_path)
        else:
            os.remove(pdf_path)

    if not has_new_pdf:
        print("📁 No new PDF to add !!")

    else:
        print("✅ Documents added Successfully !!")


def load_documents(pdf_path: str) -> List[Document]:
    """
    Loads a PDF file and returns its content as a list of Document objects.

    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        list[Document]: A list of Document objects containing the content of the PDF.
    """
    document_loader = PyPDFLoader(pdf_path)
    documents = document_loader.load()

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Splits a list of Document objects into smaller chunks
    using a recursive character splitter.

    Args:
        documents (list[Document]): A list of Document objects.
    Returns:
        list[Document]: A list of Document objects split into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)


def get_pdf_content(pdf_path: str) -> bytes:
    """
    Reads the entire PDF file and returns its content as binary data.

    Args:
        pdf_path (str): Path to the PDF file.
    Returns:
        bytes: The binary content of the PDF.
    """
    with open(pdf_path, "rb") as file:
        return file.read()


def generate_pdf_hash(content: bytes) -> str:
    """
    Generate a unique hash for the entire PDF content.

    Args:
        content (bytes): The binary content of the PDF.
    Returns:
        str: The SHA-256 hash of the PDF content.
    """
    return hashlib.sha256(content).hexdigest()


def generate_chunk_hash(chunk_content: str, page_number: int) -> str:
    """
    Generate a unique hash for a specific chunk of content.

    Args:
        chunk_content (str): The content of the chunk.
        page_number (int): The page number of the chunk.
    Returns:
        str: The SHA-256 hash of the chunk content and page number.
    """
    combined_content = f"{chunk_content}:{page_number}"
    return hashlib.sha256(combined_content.encode()).hexdigest()


def calculate_chunk_ids(chunks: List[Document], pdf_hash: str) -> List[Document]:
    """
    Generates unique IDs for each chunk of a document
    based on its content and page number.

    Args:
        chunks (list[Document]): A list of Document chunks.
        pdf_hash (str): The hash of the entire PDF content.
    Returns:
        list[Document]: The list of Document chunks with
                        unique IDs added to their metadata.
    """

    for chunk in chunks:
        page = chunk.metadata.get("page")
        chunk_content = chunk.page_content
        chunk_hash = generate_chunk_hash(chunk_content, page)

        chunk_id = f"{pdf_hash}:{chunk_hash}"
        chunk.metadata["id"] = chunk_id

    return chunks


def add_to_milvus(
    vectorstore,
    new_chunks: List[Document],
    pdf_hash: str,
) -> None:
    print(f"👉 Adding new documents: {len(new_chunks)}")

    # Add text chunks to the vector store
    new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]

    # Add PDF hash to the metadata for each chunk
    for chunk in new_chunks:
        chunk.metadata["pdf_hash"] = pdf_hash

    vectorstore.add_documents(new_chunks, ids=new_chunk_ids)


if __name__ == "__main__":
    if not os.path.exists(DATA_PATH):
        print("No 'new_pdfs' folder found in the current directory.")
        sys.exit(1)
    connect_to_milvus()
    if utility.has_collection(COLLECTION_NAME):
        collection = Collection(name=COLLECTION_NAME)
    else:
        collection = Collection(name=COLLECTION_NAME, schema=schema)

    index_params = {
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128},
        "metric_type": "L2",
    }

    if not collection.has_index():
        index = Index(collection, "vector", index_params)
        collection.create_index("vector", index_params)

    collection.load()
    start_ingest()
