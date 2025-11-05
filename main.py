import dotenv
from langchain_deepseek import ChatDeepSeek
import os
import getpass
def init_env():
    dotenv.load_dotenv()
    if not os.getenv("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = getpass.getpass("Enter your DeepSeek API key: ")
    
    os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY", "")
    os.environ["DEEPSEEK_API_BASE"] = os.getenv("DEEPSEEK_API_BASE", "")    
def main():

    llm = ChatDeepSeek(
        model="deepseek-chat",  # 或使用 deepseek-coder 等模型名
        temperature=0.7,
    )
    messages = [
        (
            "system",
            "请说出最终幻想最受欢迎的三个角色，并简要说明理由。",
        )
    ]
    response = llm.invoke(messages)
    print(response)



if __name__ == "__main__":
    init_env()
    main()
