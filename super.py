import asyncio
import operator
from typing import Annotated, Dict, List, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
import logging

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
from langchain_teddynote import logging as teddy_logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("multi_agent_supervisor")

# 환경 변수 로드
load_dotenv()

# LangSmith 추적 설정 (선택적)
teddy_logging.langsmith("multi_agent_supervisor")

# ===== 상태 정의 =====
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    search_results: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    next_agent: str
    iteration: int

# ===== 클라이언트 생성 함수 =====
async def create_client():
    """MCP 클라이언트 생성"""
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

# ===== 검색어 추출 함수 =====
def extract_search_query(messages, iteration=1):
    """사용자 메시지에서 검색어 추출"""
    if not messages:
        return "최신 뉴스"
        
    # 사용자 질문 추출
    user_message = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
    
    # 검색어 처리 로직
    search_query = user_message
    
    # 반복 횟수에 따른 검색어 변형
    if iteration > 1:
        if iteration == 2:
            search_query += " 최근 소식"
        elif iteration == 3:
            search_query += " 주요 이슈"
    
    logger.info(f"[검색 #{iteration}] 검색어: {search_query}")
    return search_query

# ===== 검색 노드 =====
def create_search_node(client):
    async def search_node(state: AgentState):
        # 현재 반복 횟수
        iteration = state.get("iteration", 1)
        
        # 검색어 추출
        search_query = extract_search_query(state["messages"], iteration)
        
        # MCP 검색 도구 찾기
        tools = client.get_tools()
        search_tool = next((tool for tool in tools if "search" in tool.name.lower()), None)
        
        if not search_tool:
            logger.error("검색 도구를 찾을 수 없습니다.")
            return {
                "messages": state["messages"] + [AIMessage(content="검색 도구를 찾을 수 없습니다.")],
                "next_agent": "analysis_node",
                "iteration": iteration
            }
        
        try:
            # 도구에 직접 검색 요청
            search_params = {
                "query": search_query, 
                "language": "ko"
            }
            result = await search_tool.ainvoke(search_params)
            
            # 문자열로 변환
            search_result_text = str(result) if not isinstance(result, str) else result
            
            # 검색 결과 메시지
            search_message = AIMessage(content=search_result_text)
            
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
            logger.error(f"검색 오류: {e}")
            return {
                "messages": state["messages"] + [AIMessage(content=f"검색 중 오류가 발생했습니다: {str(e)}")],
                "next_agent": "analysis_node",
                "iteration": iteration
            }
    
    return search_node

# ===== 분석 노드 =====
def create_analysis_node(llm):
    def analysis_node(state: AgentState):
        # 검색 결과 추출
        search_results = state.get("search_results", [])
        
        if not search_results:
            logger.warning("분석할 검색 결과가 없습니다.")
            return {
                "messages": state["messages"] + [AIMessage(content="분석할 검색 결과가 없습니다.")],
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

# ===== 그래프 생성 함수 =====
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
    
    # 그래프 컴파일
    return graph_builder.compile(checkpointer=MemorySaver())

# ===== 메인 함수 =====
async def main():
    print("===== 멀티 에이전트 검색 시스템 =====")
    try:
        # 사용자 쿼리 입력 받기 (선택적)
        user_query = input("검색할 내용을 입력하세요: ") or "이재명에 대해서 검색해줘"
        print(f"검색 쿼리: '{user_query}'")
        print("-" * 50)
        
        # 설정 
        config = RunnableConfig(
            recursion_limit=10,
            configurable={"thread_id": "1"},
            tags=["multi-agent-workflow"]
        )

        print("MCP 서버에 연결 중...")
        async with await create_client() as client:
            print("그래프 생성 중...")
            agent = create_multi_agent_graph(client)
            
            # 초기 상태 설정
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "search_results": [],
                "analysis_results": [],
                "next_agent": "search_node",
                "iteration": 1
            }
            
            print("검색 및 분석 중... (시간이 다소 소요될 수 있습니다)")
            response = await agent.ainvoke(initial_state, config=config)
            
            # 최종 결과 출력
            print("\n" + "=" * 50)
            print("검색 결과 분석 완료")
            print("=" * 50)
            
            if response.get("analysis_results"):
                final_result = response["analysis_results"][-1]["content"]
                print(final_result)
            else:
                print("결과를 얻지 못했습니다.")
            
    except Exception as e:
        logger.error(f"오류 발생: {e}")
        print(f"\n오류 발생: {e}")
    
    print("\n===== 시스템 종료 =====")

# ===== 실행 =====
if __name__ == "__main__":
    asyncio.run(main())