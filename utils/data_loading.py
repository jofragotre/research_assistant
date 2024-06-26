from llama_index.core import SimpleDirectoryReader

from langchain.schema.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from get_models import get_embedding_function

default_embedding_function = get_embedding_function()

# TODO: Add url data loader
# TODO: Add chunk splitter customization
# TODO: Add document type adaptive chunk splitting


def load_data(data_dir: str = "data"):
    # Load the documents
    documents = SimpleDirectoryReader(data_dir).load_data()
    # Convert the documents to langchain format
    documents = list(map(lambda x: x.to_langchain_format(), documents))
    return documents


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def split_chunks_into_batches(lst, batch_size=150):
    batches = []
    for i in range(0, len(lst), batch_size):
        batches.append(lst[i:i + batch_size])
    return batches


def add_to_chroma(chunks: list[Document], chroma_path: str = "../chroma", embedding_function=default_embedding_function):

    # Load the existing database.
    db = Chroma(
        persist_directory=chroma_path, embedding_function=embedding_function
    )
    batched_chunks = split_chunks_into_batches(chunks)

    for i, chunks in enumerate(batched_chunks):
        print(f"Processing batch {i} of chunks of size {len(chunks)}")

        # Calculate Page IDs.
        chunks_with_ids = calculate_chunk_ids(chunks)

        # Add or Update the documents.
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")

        # Only add documents that don't exist in the DB.
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)

        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            db.persist()
        else:
            print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks
