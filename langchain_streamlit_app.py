import os
import time
import asyncio
from langchain_openai import OpenAIEmbeddings

from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant

from langchain.chains import VectorDBQA
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain_community.vectorstores import Qdrant
from langchain.schema import retriever

import streamlit as st
from dotenv import load_dotenv

import time

# Read from .env file 
qdrant_host = os.getenv('QDRANT_HOST')

# Connect to the qdrant client
client = QdrantClient(url=qdrant_host)

# Function for querying vector points and retrieving their meta data
def extract_meta_data(sources):
    retrieved_docs = []
    list_of_metadata = []
    metadata = {}

    for source in sources:
        doc_id = source.metadata['_id']
        collection_name = source.metadata['_collection_name']

        # Query every doc and append to list
        document = client.retrieve(
            collection_name=collection_name,
            ids=[doc_id],
            with_payload=True
        )

        retrieved_docs.append(document)
    
    for each_doc in retrieved_docs:
        record = each_doc[0]
        
        # Destructuring the data type
        retrieved_payload = record.payload

        source_doc = retrieved_payload['source']
        page_content = retrieved_payload['page_content']
        page_number = retrieved_payload['page']

        metadata = {
            "Source Document" :source_doc,
            "Page Content" :page_content,
            "Page Number" : page_number
        }

        list_of_metadata.append(metadata)

    return list_of_metadata



def main():
    # Load the environment variables
    load_dotenv()

    # Configure the embedding model
    embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=768)
    st.set_page_config(page_title='Crypto compliance')

    st.header('Ask Crypto Regulatory Compliance Question!')
    st.write("Langchain pipeline")
    
    # Grab the user question
    user_question = st.text_input("Ask me anything crypto regulatory question!")


    # Initialize the document store
    doc_store = Qdrant(
        client=client,
        collection_name='test 768-dim', # Can change the collection here 
        embeddings = embeddings_model
    )

    llm = OpenAI()

    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever= doc_store.as_retriever(),
        return_source_documents=True
    )

    if user_question:
        st.markdown(f':green[Question:] {user_question}')

        # Hook up the user question
        response = qa.invoke(user_question)

        st.markdown(':green[Response:]')
        st.write(response['result'])

        time.sleep(1)

        # Print sources below
        st.header(':red[Sources used:]')

        sources = response['source_documents']

        # for source in sources:
        #     st.markdown(f':green[{source.page_content}]')

        sources = response['source_documents']
        

        # call extract metadata here
        list_of_metadata = extract_meta_data(sources)

        # print(len(list_of_metadata))
        
        for payload in list_of_metadata:
            source_doc = payload['Source Document']
            page_content = payload['Page Content']
            page_number = payload['Page Number']

            st.markdown(f':green[Source Document:] {source_doc}')
            st.markdown(f':green[Page Number] {page_number}')
            st.markdown(f':green[Page Content:] {page_content}')
            st.divider()
        



if __name__ == '__main__':
    main()