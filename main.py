import dotenv
from langchain_deepseek import ChatDeepSeek
import os
import getpass

from typing import List
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.schema import StrOutputParser
from langchain_community.document_loaders import (
    PyMuPDFLoader,TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager, index
from langchain_classic.schema import Document
from langchain_classic.schema.runnable import Runnable, RunnablePassthrough, RunnableConfig
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

import chainlit as cl


chunk_size = 1024
chunk_overlap = 50

DEVICE = "mps"
MODEL_PATH= "./hf/bge-small-zh-v1.5"
TXT_STORAGE_PATH = "./data"
def process_txt(txt_storage_path: str):
    txt_directory = Path(txt_storage_path)
    docs = []  # type: List[Document]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    for txt_path in txt_directory.glob("*.txt"):
        loader = TextLoader(str(txt_path))
        documents = loader.load()
        docs += text_splitter.split_documents(documents)

    model_name = MODEL_PATH
    model_kwargs = {"device": DEVICE}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

    doc_search = Chroma.from_documents(docs, hf)

    namespace = "chromadb/my_documents"
    record_manager = SQLRecordManager(
        namespace, db_url="sqlite:///record_manager_cache.sql"
    )
    record_manager.create_schema()

    index_result = index(
        docs,
        record_manager,
        doc_search,
        cleanup="incremental",
        source_id_key="source",
    )

    print(f"Indexing stats: {index_result}")

    return doc_search

doc_search = process_txt(TXT_STORAGE_PATH)
model = ChatDeepSeek(
        model="deepseek-chat",  # 或使用 deepseek-coder 等模型名
        temperature=0.7,
        streaming=True,
    )

@cl.on_chat_start
async def on_chat_start():
    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    retriever = doc_search.as_retriever()

    runnable = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    cl.user_session.set("runnable", runnable)

@cl.on_message
async def on_message(message: cl.Message):
    runnable: Runnable = cl.user_session.get("runnable")  # type: ignore
    msg = cl.Message(content="")

    class PostMessageHandler(BaseCallbackHandler):
        """
        Callback handler for handling the retriever and LLM processes.
        Used to post the sources of the retrieved documents as a Chainlit element.
        """

        def __init__(self, msg: cl.Message):
            BaseCallbackHandler.__init__(self)
            self.msg = msg
            self.sources = set()  # To store unique pairs

        def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
            for d in documents:
                source_page_pair = (d.metadata["source"], d.metadata["page"])
                self.sources.add(source_page_pair)  # Add unique pairs to the set

        def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
            if len(self.sources):
                sources_text = "\n".join(
                    [f"{source}#page={page}" for source, page in self.sources]
                )
                self.msg.elements.append(
                    cl.Element(name="Sources", content=sources_text, display="inline")
                )

    async for chunk in runnable.astream(
        message.content,
        config=RunnableConfig(
            callbacks=[cl.LangchainCallbackHandler(), PostMessageHandler(msg)]
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()

def init_env():
    dotenv.load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")
    
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
    os.environ["DEEPSEEK_API_BASE"] = os.getenv("DEEPSEEK_API_BASE", "")    

def main():
    pass
init_env()
if __name__ == "__main__":
    init_env()
    main()
