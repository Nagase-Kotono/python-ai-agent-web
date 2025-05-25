import asyncio
import operator
from dotenv import load_dotenv
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_teddynote import logging
from langchain_teddynote.graphs import visualize_graph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
from langchain_experimental.tools import PythonREPLTool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables from .env file
load_dotenv()

# 추적할 프로젝트 이름을 입력합니다.
logging.langsmith("websearch_agent")

# ✅ 상태 정의
class State(TypedDict):
    messages: Annotated[list, add_messages]

memory = MemorySaver()

async def create_client():
    # 1. 클라이언트 설정
    return MultiServerMCPClient(
        {
            "searxng": {
                "command": "npx",
                "args": [
                    "-y",
                    "mcp-searxng"
                ],
                "env": {
                    "SEARXNG_URL": "http://localhost:8000"
                }
            }
        }
    )
    
    
# ✅ MCP Graph 생성 함수
def mcp_graph(client):
    tools = client.get_tools()
    print("🔧 MCP Tools:", tools)

    # LLM 설정
    llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0, max_tokens=20000)
    llm_with_tools = llm.bind_tools(tools)

    # 챗봇 노드 정의
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}

    # 상태 그래프 정의
    graph_builder = StateGraph(State)

    # 노드 구성
    graph_builder.add_node("chatbot", chatbot)

    tool_node = ToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    # 시작과 종료 정의
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    # 그래프 컴파일
    return graph_builder.compile(checkpointer=memory)



    
# ✅ 메인 함수
async def main():
    config = RunnableConfig(
        recursion_limit=10,
        configurable={"thread_id": "1"},
        tags=["my-tag"]
    )

    async with await create_client() as client:
        agent = mcp_graph(client)
        response = await agent.ainvoke(
            {"messages": "이재명에 대해서 검색해줘"},  # 메시지를 MCP 도구에 맞게 조정
            config=config
        )
        print("📨 Agent Response:", response)


# ✅ 실행
asyncio.run(main())