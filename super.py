import asyncio
import operator
from dotenv import load_dotenv
from typing import Sequence, Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_teddynote import logging
from langchain_teddynote.graphs import visualize_graph
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# .env 파일에서 환경 변수 로드
load_dotenv()

# 프로젝트 로깅 설정
logging.langsmith("multi_agent_supervisor")

# ✅ 상태 정의 - 간단한 구조로 유지
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    next_agent: str
    iteration: int

# 메모리 세이버 초기화
memory = MemorySaver()

# ✅ 클라이언트 생성 함수
async def create_client():
    # 기존에 작동하던 설정 사용
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

# ✅ 검색어 추출기 함수
def extract_search_query(messages, iteration=1):
    """
    사용자 메시지에서 검색어를 추출합니다.
    
    iteration: 검색 반복 횟수. 2회차 이상은 다른 결과를 얻기 위해 검색어를 약간 변형합니다.
    """
    if not messages:
        return "최신 뉴스"
        
    # 사용자의 원래 질문 추출
    user_message = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
    
    # 기본 검색어는 사용자 메시지 그대로 사용
    search_query = user_message
    
    # 특정 의도 파악 및 검색어 조정
    if "날씨" in user_message:
        # 지역명 추출 시도
        if "서울" in user_message:
            search_query = "서울 날씨"
        elif "부산" in user_message:
            search_query = "부산 날씨"
        elif "대구" in user_message:
            search_query = "대구 날씨"
        elif "인천" in user_message:
            search_query = "인천 날씨"
        else:
            search_query = "오늘 날씨"
    elif "뉴스" in user_message or "소식" in user_message:
        search_query = "최신 뉴스"
    
    # 인물 검색인 경우
    elif any(keyword in user_message for keyword in ["누구", "인물", "사람"]):
        # 인물명 추출
        words = user_message.split()
        for word in words:
            if len(word) >= 2 and word not in ["누구", "인물", "사람", "검색", "알려줘", "대해", "대하여", "정보"]:
                search_query = f"{word} 인물 정보"
                break
    
    # 특정 인물명이 있는 경우
    for name in ["이재명", "윤석열", "문재인"]:
        if name in user_message:
            # 기본적으로 인물명으로 검색
            search_query = name
            
            # 특정 측면을 묻는 경우 검색어 조정
            if "경력" in user_message or "프로필" in user_message:
                search_query = f"{name} 경력 프로필"
            elif "정책" in user_message or "공약" in user_message:
                search_query = f"{name} 정책 공약"
            elif "논란" in user_message or "비판" in user_message:
                search_query = f"{name} 논란 이슈"
                
            break
    
    # 2회차 이상인 경우 검색 변형
    if iteration > 1:
        if iteration == 2:
            search_query += " 최근 소식"
        elif iteration == 3:
            search_query += " 주요 이슈"
    
    return search_query

# ✅ 다단계 검색 에이전트
def create_search_node(client):
    async def search_node(state: AgentState):
        # 현재 반복 횟수
        iteration = state.get("iteration", 1)
        
        # 사용자 질문에서 검색어 추출
        search_query = extract_search_query(state["messages"], iteration)
        
        print(f"[검색 #{iteration}] 검색어: {search_query}")
        
        # MCP 검색 도구 찾기
        tools = client.get_tools()
        search_tool = next((tool for tool in tools if tool.name == "searxng_web_search"), None)
        
        if not search_tool:
            search_message = AIMessage(content="검색 도구를 찾을 수 없습니다.")
            return {
                "messages": state["messages"] + [search_message],
                "next_agent": "analysis_node",
                "iteration": iteration
            }
        
        try:
            # 도구에 직접 검색 요청
            search_params = {
                "query": search_query, 
                "language": "ko",
                "safesearch": "0"
            }
            result = await search_tool.ainvoke(search_params)
            
            # 문자열로 변환
            if isinstance(result, str):
                search_result_text = result
            else:
                search_result_text = str(result)
                
            # 검색 결과 메시지 생성
            search_message = AIMessage(content=f"'{search_query}' 검색 결과:\n\n{search_result_text}")
            
            # 검색 결과 저장
            search_results = state.get("search_results", [])
            search_results.append({
                "query": search_query,
                "content": search_result_text,
                "iteration": iteration
            })
            
            # 다음 단계 결정
            next_stage = "search_node" if iteration < 3 else "analysis_node"
            
            return {
                "messages": state["messages"] + [search_message],
                "search_results": search_results,
                "next_agent": next_stage,
                "iteration": iteration + 1
            }
            
        except Exception as e:
            print(f"검색 오류: {e}")
            error_message = AIMessage(content=f"검색 중 오류가 발생했습니다: {str(e)}")
            return {
                "messages": state["messages"] + [error_message],
                "next_agent": "analysis_node",
                "iteration": iteration
            }
    
    return search_node

# ✅ 분석 에이전트 노드
def create_analysis_node(llm):
    def analysis_node(state: AgentState):
        # 검색 결과 추출
        search_results = state.get("search_results", [])
        
        if not search_results:
            analysis_message = AIMessage(content="분석할 검색 결과가 없습니다.")
            return {
                "messages": state["messages"] + [analysis_message],
                "next_agent": END
            }
        
        # 검색 결과 텍스트 구성
        search_texts = []
        for result in search_results:
            query = result.get("query", "검색어 없음")
            content = result.get("content", "")
            search_texts.append(f"검색어: '{query}'\n{content}")
        
        search_text = "\n\n---\n\n".join(search_texts)
        
        # 사용자 원래 질문 추출
        original_query = state["messages"][0].content
        
        # 분석 프롬프트
        analysis_prompt = f"""
        사용자 질문: {original_query}
        
        다음 검색 결과를 분석하여 사용자 질문에 답변해주세요:
        
        {search_text}
        
        중요한 정보를 정리하여 답변하되, 다음을 포함해주세요:
        1. 사용자 질문에 직접적인 답변
        2. 관련된 중요 정보 요약
        3. 정보의 출처 (가능한 경우)
        
        답변은 명확하고 객관적으로 작성해주세요.
        """
        
        # 분석 수행
        analysis_response = llm.invoke([HumanMessage(content=analysis_prompt)])
        
        # 분석 결과 저장
        analysis_results = state.get("analysis_results", [])
        analysis_results.append({"content": analysis_response.content})
        
        return {
            "messages": state["messages"] + [analysis_response],
            "analysis_results": analysis_results,
            "next_agent": END
        }
    
    return analysis_node

# ✅ 멀티 에이전트 워크플로우 그래프 생성
def create_multi_agent_graph(client):
    # LLM 설정
    llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0, max_tokens=20000)
    
    # 노드 생성
    search_node = create_search_node(client)
    analysis_node = create_analysis_node(llm)
    
    # 상태 그래프 정의
    graph_builder = StateGraph(AgentState)
    
    # 노드 추가
    graph_builder.add_node("search_node", search_node)
    graph_builder.add_node("analysis_node", analysis_node)
    
    # 엣지 추가
    graph_builder.add_edge(START, "search_node")
    
    # 조건부 엣지
    def router(state: AgentState):
        return state["next_agent"]
    
    graph_builder.add_conditional_edges("search_node", router)
    graph_builder.add_edge("analysis_node", END)
    
    # 그래프 컴파일
    return graph_builder.compile(checkpointer=memory)

# ✅ 메인 함수
async def main():
    config = RunnableConfig(
        recursion_limit=10,
        configurable={"thread_id": "1"},
        tags=["multi-agent-workflow"]
    )

    async with await create_client() as client:
        agent = create_multi_agent_graph(client)
        
        # 초기 상태 설정
        initial_state = {
            "messages": [HumanMessage(content="이재명에 대해서 검색해줘")],
            "search_results": [],
            "analysis_results": [],
            "next_agent": "search_node",
            "iteration": 1
        }
        
        response = await agent.ainvoke(initial_state, config=config)
        print("📨 최종 응답:", response)
        
        # 그래프 시각화
        visualize_graph(agent, "multi_agent_workflow.png")

# ✅ 실행
if __name__ == "__main__":
    asyncio.run(main())