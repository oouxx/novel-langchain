
from fastapi import FastAPI
from langserve import add_routes
from novel import agent as create_novel_rag_agent
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage,AIMessage

app = FastAPI(
    title="小说 RAG 问答系统",
    version="1.0",
    description="基于《遮天》等小说的智能问答，支持角色分析、情节检索。"
)

# 创建 Agent 并注册到 LangServe
rag_agent = create_novel_rag_agent()


add_routes(
    app,
    rag_agent,
    path="/novel-qa",
    playground_type="default"  # 启用聊天式 Playground
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("serve:app", host="127.0.0.1", port=8000, reload=True)