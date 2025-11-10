from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_deepseek import ChatDeepSeek
from langchain.agents import create_agent
from langchain_classic.tools.retriever import create_retriever_tool

import dotenv
import os
import getpass
import platform
import torch

DEVICE = "cpu"
MODEL_PATH= "./hf/bge-small-zh-v1.5"
def load():
    # 1. 加载小说
    loader = TextLoader("./data/zhetian.txt", encoding="utf-8")
    docs = loader.load()
    return docs
def text_split( docs):
    # 2. 分块
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    return chunks
def vec(chunks):
    # 3. 向量化
    model_name = MODEL_PATH
    model_kwargs = {"device": DEVICE}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    vectorstore = InMemoryVectorStore.from_documents(chunks, hf)
    return vectorstore

def agent():
    docs = load()
    chunks = text_split(docs)
    vectorstore = vec(chunks)
    llm = ChatDeepSeek(
        model="deepseek-chat",  # 或使用 deepseek-coder 等模型名
        temperature=0.7,
    )
    # 4. 构建问答链
    retriever_tool = create_retriever_tool(
        retriever=vectorstore.as_retriever(),
        name="search_documents",
        description="搜索文档内容，用于回答关于文档的问题"
    )    
 
    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt="你是一个有帮助的助手，擅长从提供的小说内容中回答问题。",
    )
    return agent

def get_device():
    """自动检测最佳 device"""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        if torch.backends.mps.is_available():
            print("检测到 macOS + Apple Silicon → 使用 MPS 加速")
            return "mps"
        else:
            print("macOS 但 MPS 不可用 → 回退 CPU")
            return "cpu"
    
    elif system in ["Windows", "Linux"]:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"检测到 {system} + NVIDIA GPU ({gpu_name}) → 使用 CUDA")
            return "cuda"
        else:
            print(f"{system} 无 GPU → 使用 CPU")
            return "cpu"
    
    else:
        print(f"未知系统 {system} → 使用 CPU")
        return "cpu"
def init_env():
    dotenv.load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")
    
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
    os.environ["DEEPSEEK_API_BASE"] = os.getenv("DEEPSEEK_API_BASE", "")
    DEVICE = get_device()
    if DEVICE == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # 防止部分算子报错


def main():
    docs = load()
    chunks = text_split(docs)
    vectorstore = vec(chunks)
    llm = ChatDeepSeek(
        model="deepseek-chat",  # 或使用 deepseek-coder 等模型名
        temperature=0.7,
    )
    # 4. 构建问答链
    retriever_tool = create_retriever_tool(
        retriever=vectorstore.as_retriever(),
        name="search_documents",
        description="搜索文档内容，用于回答关于文档的问题"
    )    
 
    agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        system_prompt="你是一个有帮助的助手，擅长从提供的小说内容中回答问题。",
    )


    # 5. 提问
    response = agent.invoke({
        "messages": [{"role": "user", "content": "请讲一下叶凡是一个怎么样的人，给出小说里的依据"}]
    })
    print(response)

if __name__ == "__main__":
    init_env()
    main()
