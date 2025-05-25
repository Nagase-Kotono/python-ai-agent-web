import asyncio
from typing import Annotated, Dict, List, Any
from typing_extensions import TypedDict
from dotenv import load_dotenv
import logging
import json
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.tree import Tree

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableConfig
from langchain_teddynote import logging as teddy_logging
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Rich Console 설정
console = Console()
error_console = Console(stderr=True, style="bold red")

# Rich 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("multi_agent_supervisor")

# Rich Console로 로깅 함수 추가
def log_info(message: str, style: str = "blue"):
    console.print(f"ℹ️  {message}", style=style)

def log_error(message: str):
    error_console.print(f"❌ {message}")

def log_success(message: str):
    console.print(f"✅ {message}", style="green")

def log_warning(message: str):
    console.print(f"⚠️  {message}", style="yellow")

# 환경 변수 로드
load_dotenv()

# LangSmith 추적 설정
# teddy_logging.langsmith("multi_agent_supervisor") # .env 에서 langsmith 키 설정 필요 LANGCHAIN_API_KEY="lsv2_pt_...""

# ===== 상태 정의 =====
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    search_results: List[Dict[str, Any]]
    file_results: List[Dict[str, Any]]
    code_results: List[Dict[str, Any]]
    analysis_results: List[Dict[str, Any]]
    available_tools: Dict[str, List[str]]
    next_agent: str
    iteration: int
    task_type: str  # 'search', 'file', 'code', 'mixed'
    completed_tasks: List[str]

# ===== MCP 서버 설정 =====
def get_mcp_servers_config():
    """MCP 서버 설정 반환"""
    return {
        "searxng": {
            "command": "npx",
            "args": ["-y", "mcp-searxng"],
            "env": {"SEARXNG_URL": "http://localhost:8000"},
            "description": "웹 검색 엔진"
        },
        "jetbrains": {
            "command": "npx",
            "args": ["-y", "@jetbrains/mcp-proxy"],
            "description": "IDE 코드 편집 및 분석"
        },
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "/Users/kotono/github"
            ],
            "description": "파일 시스템 작업"
        }
    }

# ===== 클라이언트 생성 함수 =====
async def create_client():
    """MCP 클라이언트 생성"""
    servers_config = get_mcp_servers_config()
    
    log_info(f"MCP 서버 연결 중: {list(servers_config.keys())}")
    return MultiServerMCPClient(servers_config)

# ===== 작업 유형 분석 함수 =====
def analyze_task_type(user_query: str) -> Dict[str, Any]:
    """사용자 쿼리를 분석하여 작업 유형 결정"""
    query_lower = user_query.lower()
    
    task_info = {
        "type": "mixed",
        "priority": [],
        "keywords": []
    }
    
    # 검색 관련 키워드
    search_keywords = ["검색", "찾아", "알아봐", "정보", "뉴스", "소식", "최신"]
    
    # 파일 관련 키워드  
    file_keywords = ["파일", "폴더", "디렉토리", "저장", "읽기", "쓰기", "생성", "내부", "뭐가"]
    
    # 코드 관련 키워드
    code_keywords = ["코드", "프로그래밍", "함수", "클래스", "버그", "디버그", "리팩토링"]
    
    if any(keyword in query_lower for keyword in search_keywords):
        task_info["priority"].append("search")
        task_info["keywords"].extend([k for k in search_keywords if k in query_lower])
    
    if any(keyword in query_lower for keyword in file_keywords):
        task_info["priority"].append("file")
        task_info["keywords"].extend([k for k in file_keywords if k in query_lower])
    
    if any(keyword in query_lower for keyword in code_keywords):
        task_info["priority"].append("code")
        task_info["keywords"].extend([k for k in code_keywords if k in query_lower])
    
    # 우선순위가 하나만 있으면 해당 타입으로 설정
    if len(task_info["priority"]) == 1:
        task_info["type"] = task_info["priority"][0]
    elif not task_info["priority"]:
        task_info["type"] = "search"  # 기본값
        task_info["priority"] = ["search"]
    
    log_info(f"🎯 작업 유형 분석: {task_info}")
    return task_info

# ===== 도구 매핑 함수 =====
def get_tools_by_category(client, category: str) -> List[Any]:
    """카테고리별 도구 반환 - MCP 서버별로 구분"""
    all_tools = client.get_tools()
    
    # 각 MCP 서버별 도구 분류
    filesystem_tools = []
    jetbrains_tools = []
    search_tools = []
    
    for tool in all_tools:
        tool_name_lower = tool.name.lower()
        
        # JetBrains 서버 도구들 (우선 분류)
        if any(keyword in tool_name_lower for keyword in [
            "editor", "debugger", "terminal", "reformat", "action",
            "tree_in_folder", "open_file_paths", "breakpoint", "selected_in_editor"
        ]):
            jetbrains_tools.append(tool)
            
        # 검색 도구들
        elif any(keyword in tool_name_lower for keyword in ["searxng", "search", "query", "browse"]):
            search_tools.append(tool)
            
        # Filesystem 서버 도구들 (순수 파일시스템 작업만)
        elif any(keyword in tool_name_lower for keyword in [
            "read_file", "write_file", "list_directory", "create_directory", 
            "move_file", "get_file_info"
        ]) and not any(exclude in tool_name_lower for exclude in [
            "editor", "tree_in_folder", "open_file"
        ]):
            filesystem_tools.append(tool)
    
    # 카테고리별 우선순위 반환
    if category == "search":
        log_info(f"🔍 검색 도구: {[t.name for t in search_tools]}")
        return search_tools
    elif category == "file":
        # Filesystem 도구 우선 사용
        log_info(f"📁 파일 도구 - filesystem: {[t.name for t in filesystem_tools]}, jetbrains: {[t.name for t in jetbrains_tools]}")
        return filesystem_tools if filesystem_tools else jetbrains_tools
    elif category == "code":
        # JetBrains 도구 우선
        log_info(f"💻 코드 도구: {[t.name for t in jetbrains_tools]}")
        return jetbrains_tools
    
    return []

# ===== 검색 노드 =====
def create_search_node(client):
    async def search_node(state: AgentState):
        task_type = state.get("task_type", "search")
        
        # 파일이나 코드 전용 작업이면 검색 건너뛰기
        if task_type in ["file", "code"]:
            completed_tasks = state.get("completed_tasks", [])
            if task_type == "file":
                next_agent = "file_node"
            else:
                next_agent = "code_node"
            
            return {
                "messages": state["messages"],
                "search_results": state.get("search_results", []),
                "next_agent": next_agent,
                "iteration": state.get("iteration", 1),
                "completed_tasks": completed_tasks
            }
        
        iteration = state.get("iteration", 1)
        user_query = state.get("user_query", "")
        
        # 검색 도구 찾기
        search_tools = get_tools_by_category(client, "search")
        
        if not search_tools:
            log_error("검색 도구를 찾을 수 없습니다.")
            return {
                "messages": state["messages"] + [AIMessage(content="검색 도구를 사용할 수 없습니다.")],
                "next_agent": "analysis_node",
                "completed_tasks": state.get("completed_tasks", []) + ["search_failed"]
            }
        
        search_tool = search_tools[0]  # 첫 번째 검색 도구 사용
        log_info(f"🔍 선택된 검색 도구: {search_tool.name}", "blue")
        
        # 검색어 개선
        search_queries = generate_search_queries(user_query, iteration)
        
        search_results = state.get("search_results", [])
        
        for query in search_queries:
            try:
                log_info(f"🔍 검색 실행: {query}")
                search_params = {"query": query, "language": "ko"}
                result = await search_tool.ainvoke(search_params)
                
                search_results.append({
                    "query": query,
                    "content": str(result),
                    "iteration": iteration,
                    "timestamp": datetime.now().isoformat()
                })
                
                log_success(f"🔍 검색 완료: {query}")
                
            except Exception as e:
                log_error(f"검색 오류 ({query}): {e}")
                search_results.append({
                    "query": query,
                    "content": f"검색 실패: {str(e)}",
                    "iteration": iteration,
                    "error": True
                })
        
        # 다음 단계 결정
        task_type = state.get("task_type", "search")
        completed_tasks = state.get("completed_tasks", [])
        
        if task_type == "search" or "search" not in completed_tasks:
            next_agent = "analysis_node" if iteration >= 2 else "search_node"
        else:
            # 다른 작업도 수행해야 하는 경우
            if "file" not in completed_tasks and task_type in ["file", "mixed"]:
                next_agent = "file_node"
            elif "code" not in completed_tasks and task_type in ["code", "mixed"]:
                next_agent = "code_node"
            else:
                next_agent = "analysis_node"
        
        return {
            "messages": state["messages"] + [AIMessage(content=f"검색 완료: {len(search_results)}개 결과")],
            "search_results": search_results,
            "next_agent": next_agent,
            "iteration": iteration + 1,
            "completed_tasks": completed_tasks + ["search"]
        }
    
    return search_node

# ===== 파일 작업 노드 =====
def create_file_node(client):
    async def file_node(state: AgentState):
        user_query = state.get("user_query", "")
        
        # 파일 도구 찾기
        file_tools = get_tools_by_category(client, "file")
        
        if not file_tools:
            log_warning("파일 도구를 찾을 수 없습니다.")
            return {
                "messages": state["messages"] + [AIMessage(content="파일 작업 도구를 사용할 수 없습니다.")],
                "next_agent": "analysis_node",
                "completed_tasks": state.get("completed_tasks", []) + ["file_failed"]
            }
        
        file_results = state.get("file_results", [])
        
        # 파일 작업 수행
        try:
            # 작업 유형에 따른 도구 선택
            selected_tool = None
            
            # 디렉토리 목록 조회가 필요한 경우
            if any(keyword in user_query.lower() for keyword in ["폴더", "내부", "목록", "뭐가", "what", "list"]):
                # list_directory 도구 찾기
                for tool in file_tools:
                    if "list_directory" in tool.name and "tree" not in tool.name:
                        selected_tool = tool
                        break
            
            # 기본 도구 선택 (list_directory 우선 또는 첫 번째)
            if not selected_tool:
                selected_tool = file_tools[0]
            
            log_info(f"🔧 선택된 파일 도구: {selected_tool.name}", "cyan")
            
            # 파일 작업 결정
            file_operations = determine_file_operations(user_query)
            
            for operation in file_operations:
                try:
                    result = await selected_tool.ainvoke(operation)
                    
                    file_results.append({
                        "tool": selected_tool.name,
                        "operation": operation,
                        "result": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    log_success(f"📁 파일 작업 완료: {selected_tool.name} -> {operation}")
                    
                except Exception as e:
                    log_error(f"파일 작업 오류 ({selected_tool.name}, {operation}): {e}")
                    file_results.append({
                        "tool": selected_tool.name,
                        "operation": operation,
                        "result": f"작업 실패: {str(e)}",
                        "error": True
                    })
        
        except Exception as e:
            log_error(f"파일 작업 전체 오류: {e}")
            file_results.append({
                "operation": "error",
                "result": f"파일 작업 실패: {str(e)}",
                "error": True
            })
        
        completed_tasks = state.get("completed_tasks", [])
        task_type = state.get("task_type", "mixed")
        
        # 다음 노드 결정
        if "code" not in completed_tasks and task_type in ["code", "mixed"]:
            next_agent = "code_node"
        else:
            next_agent = "analysis_node"
        
        return {
            "messages": state["messages"] + [AIMessage(content=f"파일 작업 완료: {len(file_results)}개 작업")],
            "file_results": file_results,
            "next_agent": next_agent,
            "completed_tasks": completed_tasks + ["file"]
        }
    
    return file_node

# ===== 코드 작업 노드 =====
def create_code_node(client):
    async def code_node(state: AgentState):
        user_query = state.get("user_query", "")
        
        # JetBrains 도구 사용
        code_tools = get_tools_by_category(client, "code")
        
        if not code_tools:
            log_warning("코드 작업 도구를 찾을 수 없습니다.")
            return {
                "messages": state["messages"] + [AIMessage(content="코드 작업 도구를 사용할 수 없습니다.")],
                "next_agent": "analysis_node",
                "completed_tasks": state.get("completed_tasks", []) + ["code_failed"]
            }
        
        code_results = state.get("code_results", [])
        
        try:
            selected_tool = code_tools[0]
            log_info(f"💻 선택된 코드 도구: {selected_tool.name}", "magenta")
            
            # 코드 작업 결정
            code_operations = determine_code_operations(user_query, selected_tool.name)
            
            for operation in code_operations:
                try:
                    result = await selected_tool.ainvoke(operation)
                    
                    code_results.append({
                        "tool": selected_tool.name,
                        "operation": operation,
                        "result": str(result),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    log_success(f"💻 코드 작업 완료: {selected_tool.name} -> {operation}")
                    
                except Exception as e:
                    log_error(f"코드 작업 오류 ({selected_tool.name}, {operation}): {e}")
                    code_results.append({
                        "tool": selected_tool.name,
                        "operation": operation,
                        "result": f"작업 실패: {str(e)}",
                        "error": True
                    })
        
        except Exception as e:
            log_error(f"코드 작업 전체 오류: {e}")
            code_results.append({
                "operation": "error", 
                "result": f"코드 작업 실패: {str(e)}",
                "error": True
            })
        
        return {
            "messages": state["messages"] + [AIMessage(content=f"코드 작업 완료: {len(code_results)}개 작업")],
            "code_results": code_results,
            "next_agent": "analysis_node",
            "completed_tasks": state.get("completed_tasks", []) + ["code"]
        }
    
    return code_node

# ===== 분석 노드 =====
def create_analysis_node(llm):
    def analysis_node(state: AgentState):
        user_query = state.get("user_query", "")
        search_results = state.get("search_results", [])
        file_results = state.get("file_results", [])
        code_results = state.get("code_results", [])
        
        # 모든 결과 통합
        all_results = {
            "search": search_results,
            "file": file_results, 
            "code": code_results
        }
        
        # 결과가 있는 카테고리만 필터링
        available_results = {k: v for k, v in all_results.items() if v}
        
        if not available_results:
            return {
                "messages": state["messages"] + [AIMessage(content="분석할 데이터가 없습니다.")],
                "next_agent": END
            }
        
        # 통합 분석 프롬프트 생성
        analysis_prompt = create_analysis_prompt(user_query, available_results)
        
        log_info("🧠 AI 분석 수행 중...", "purple")
        
        # 분석 수행
        analysis_response = llm.invoke([
            SystemMessage(content="당신은 다양한 데이터 소스를 분석하여 사용자 질문에 종합적으로 답변하는 전문가입니다. 한국어로 명확하고 구조화된 답변을 제공해주세요."),
            HumanMessage(content=analysis_prompt)
        ])
        
        # 분석 결과 저장
        analysis_results = state.get("analysis_results", [])
        analysis_results.append({
            "content": analysis_response.content,
            "sources": list(available_results.keys()),
            "timestamp": datetime.now().isoformat()
        })
        
        log_success("🧠 AI 분석 완료")
        
        return {
            "messages": state["messages"] + [analysis_response],
            "analysis_results": analysis_results,
            "next_agent": END
        }
    
    return analysis_node

# ===== 헬퍼 함수들 =====
def generate_search_queries(user_query: str, iteration: int) -> List[str]:
    """검색어 생성"""
    base_query = user_query
    queries = [base_query]
    
    if iteration == 1:
        queries.append(f"{base_query} 최신")
        queries.append(f"{base_query} 2024 2025")
    elif iteration == 2:
        queries.append(f"{base_query} 상세")
        queries.append(f"{base_query} 분석")
    
    return queries[:2]  # 최대 2개까지만

def determine_file_operations(user_query: str) -> List[Dict[str, Any]]:
    """파일 작업 결정"""
    operations = []
    base_path = "/Users/kotono/github"  # MCP filesystem 서버 경로와 일치
    
    if "읽기" in user_query or "read" in user_query.lower() or "폴더" in user_query or "내부" in user_query:
        operations.append({"path": base_path})
    
    if "생성" in user_query or "create" in user_query.lower():
        operations.append({"path": f"{base_path}/output.txt", "content": "생성된 파일"})
    
    if not operations:
        # 기본적으로 루트 디렉토리 탐색
        operations.append({"path": base_path})
    
    return operations

def determine_code_operations(user_query: str, tool_name: str = "") -> List[Dict[str, Any]]:
    """코드 작업 결정"""
    operations = []
    
    if "분석" in user_query or "analyze" in user_query.lower():
        operations.append({"action": "analyze_code"})
    
    if "포맷" in user_query or "format" in user_query.lower():
        operations.append({"action": "format_code"})
        
    if "오류" in user_query or "error" in user_query.lower() or "버그" in user_query:
        operations.append({"action": "check_errors"})
    
    if not operations:
        # 기본적으로 코드 상태 확인
        operations.append({"action": "get_editor_info"})
    
    return operations

def create_analysis_prompt(user_query: str, results: Dict[str, List]) -> str:
    """종합 분석 프롬프트 생성"""
    prompt = f"""
사용자 질문: {user_query}

다음은 여러 소스에서 수집된 데이터입니다:

"""
    
    for source, data in results.items():
        source_name = {
            "search": "웹 검색",
            "file": "파일 시스템",
            "code": "코드 분석"
        }.get(source, source.upper())
        
        prompt += f"\n## {source_name} 결과:\n"
        for i, item in enumerate(data, 1):
            if isinstance(item, dict):
                content = item.get("content", item.get("result", str(item)))
                if not item.get("error", False):  # 오류가 아닌 경우만 포함
                    prompt += f"{i}. {content}\n"
            else:
                prompt += f"{i}. {str(item)}\n"
    
    prompt += """

위의 모든 정보를 종합하여 사용자 질문에 대한 완전하고 정확한 답변을 제공해주세요.

답변 구성:
1. 사용자 질문에 대한 직접적인 답변
2. 주요 정보 요약 
3. 관련된 추가 정보
4. 정보의 출처별 신뢰성

답변은 한국어로 명확하고 구조화된 형태로 제공해주세요.
"""
    
    return prompt

# ===== 그래프 생성 함수 =====
def create_multi_agent_graph(client):
    # LLM 설정
    llm = ChatOpenAI(model="gpt-4.1-2025-04-14", temperature=0, max_tokens=20000)
    
    # 노드 생성
    search_node = create_search_node(client)
    file_node = create_file_node(client)
    code_node = create_code_node(client)
    analysis_node = create_analysis_node(llm)
    
    # 상태 그래프 정의
    graph_builder = StateGraph(AgentState)
    
    # 노드 추가
    graph_builder.add_node("search_node", search_node)
    graph_builder.add_node("file_node", file_node)
    graph_builder.add_node("code_node", code_node)
    graph_builder.add_node("analysis_node", analysis_node)
    
    # 시작점을 작업 유형에 따라 설정
    def determine_start_node(state: AgentState):
        task_type = state.get("task_type", "search")
        if task_type == "file":
            return "file_node"
        elif task_type == "code":
            return "code_node"
        else:
            return "search_node"
    
    graph_builder.add_conditional_edges(START, determine_start_node)
    
    # 조건부 라우팅
    def router(state: AgentState):
        return state["next_agent"]
    
    graph_builder.add_conditional_edges("search_node", router)
    graph_builder.add_conditional_edges("file_node", router)
    graph_builder.add_conditional_edges("code_node", router)
    
    # 그래프 컴파일
    return graph_builder.compile(checkpointer=MemorySaver())

# ===== Rich UI 결과 표시 함수들 =====
def display_results(response: Dict[str, Any], user_query: str):
    """Rich를 사용한 결과 표시"""
    
    console.print("\n" + "="*60, style="bold green")
    console.print("📋 [bold cyan]작업 완료 - 종합 분석 결과[/bold cyan]", justify="center")
    console.print("="*60, style="bold green")
    
    # 1. 사용자 질문 표시
    question_panel = Panel(
        f"[bold white]{user_query}[/bold white]",
        title="❓ 사용자 질문",
        title_align="left",
        border_style="blue"
    )
    console.print(question_panel)
    
    # 2. 결과 데이터 요약
    search_count = len(response.get("search_results", []))
    file_count = len(response.get("file_results", []))
    code_count = len(response.get("code_results", []))
    
    summary_table = Table(title="📊 수집된 데이터 요약", show_header=True, header_style="bold yellow")
    summary_table.add_column("분류", style="cyan", width=10)
    summary_table.add_column("개수", justify="right", style="green", width=8)
    summary_table.add_column("상태", style="magenta")
    
    summary_table.add_row("🔍 검색", str(search_count), "✅ 완료" if search_count > 0 else "⚪ 없음")
    summary_table.add_row("📁 파일", str(file_count), "✅ 완료" if file_count > 0 else "⚪ 없음")
    summary_table.add_row("💻 코드", str(code_count), "✅ 완료" if code_count > 0 else "⚪ 없음")
    
    console.print(summary_table)
    
    # 3. AI 분석 결과 표시
    if response.get("analysis_results"):
        final_result = response["analysis_results"][-1]["content"]
        
        analysis_panel = Panel(
            final_result,
            title="🧠 AI 종합 분석 결과",
            title_align="left",
            border_style="green"
        )
        console.print(analysis_panel)
    
    # 4. 상세 결과 표시
    if file_count > 0:
        display_file_results(response["file_results"])
    
    if search_count > 0 and search_count <= 3:  # 검색 결과가 적을 때만 표시
        display_search_results(response["search_results"])
        
    if code_count > 0:
        display_code_results(response["code_results"])
    
    # 5. 작업 요약
    completed = response.get("completed_tasks", [])
    
    summary_panel = Panel(
        f"""
🎯 [bold]수행된 작업:[/bold] {', '.join(completed) if completed else '없음'}
📊 [bold]수집된 데이터:[/bold] 검색 {search_count}개, 파일 {file_count}개, 코드 {code_count}개
⏰ [bold]완료 시간:[/bold] {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """.strip(),
        title="📋 작업 요약",
        title_align="left",
        border_style="green"
    )
    console.print(summary_panel)


def display_file_results(file_results: List[Dict[str, Any]]):
    """파일 결과 표시"""
    console.print("\n📁 [bold cyan]파일 작업 결과[/bold cyan]")
    
    for i, result in enumerate(file_results, 1):
        if result.get("error"):
            panel_style = "red"
            status = "❌ 실패"
        else:
            panel_style = "green"
            status = "✅ 성공"
        
        # 결과 데이터 포맷팅
        result_text = result.get("result", "")
        
        # JSON 데이터인 경우 파싱하여 예쁘게 표시
        try:
            if result_text.startswith('[') or result_text.startswith('{'):
                parsed_data = json.loads(result_text)
                if isinstance(parsed_data, list) and len(parsed_data) > 0:
                    # 파일/폴더 목록인 경우
                    tree = Tree("📂 디렉토리 구조")
                    for item in parsed_data[:10]:  # 최대 10개만 표시
                        if isinstance(item, dict):
                            name = item.get('name', str(item))
                            item_type = item.get('type', 'file')
                            icon = "📁" if item_type == 'directory' else "📄"
                            tree.add(f"{icon} {name}")
                        else:
                            tree.add(f"📄 {item}")
                    
                    if len(parsed_data) > 10:
                        tree.add(f"... 그리고 {len(parsed_data) - 10}개 더")
                    
                    result_text = tree
                else:
                    result_text = Syntax(json.dumps(parsed_data, indent=2, ensure_ascii=False), "json")
        except:
            # JSON이 아닌 경우 그대로 표시
            if len(result_text) > 500:
                result_text = result_text[:500] + "..."
        
        file_panel = Panel(
            result_text,
            title=f"📁 작업 #{i} - {result.get('tool', 'Unknown')} ({status})",
            title_align="left",
            border_style=panel_style
        )
        console.print(file_panel)


def display_search_results(search_results: List[Dict[str, Any]]):
    """검색 결과 표시"""
    console.print("\n🔍 [bold cyan]검색 결과[/bold cyan]")
    
    for i, result in enumerate(search_results, 1):
        if result.get("error"):
            panel_style = "red"
            status = "❌ 실패"
        else:
            panel_style = "blue"
            status = "✅ 성공"
        
        content = result.get("content", "")
        if len(content) > 300:
            content = content[:300] + "..."
        
        search_panel = Panel(
            content,
            title=f"🔍 검색 #{i} - {result.get('query', 'Unknown')} ({status})",
            title_align="left",
            border_style=panel_style
        )
        console.print(search_panel)


def display_code_results(code_results: List[Dict[str, Any]]):
    """코드 결과 표시"""
    console.print("\n💻 [bold cyan]코드 작업 결과[/bold cyan]")
    
    for i, result in enumerate(code_results, 1):
        if result.get("error"):
            panel_style = "red"
            status = "❌ 실패"
        else:
            panel_style = "purple"
            status = "✅ 성공"
        
        code_panel = Panel(
            result.get("result", ""),
            title=f"💻 코드 #{i} - {result.get('tool', 'Unknown')} ({status})",
            title_align="left",
            border_style=panel_style
        )
        console.print(code_panel)

# main() 함수의 수정된 부분

async def main():
    # 시작 메시지
    console.print("\n" + "="*60, style="bold blue")
    console.print("🚀 [bold cyan]멀티 에이전트 MCP 시스템 시작[/bold cyan]", justify="center")
    console.print("="*60 + "\n", style="bold blue")
    
    try:
        # 사용자 쿼리 입력
        user_query = input("작업할 내용을 입력하세요: ") or "현재 폴더 내부에는 뭐가 있니?"
        console.print(f"🎯 [bold]사용자 쿼리:[/bold] '{user_query}'")
        
        # 작업 유형 분석
        task_info = analyze_task_type(user_query)
        console.print(f"📝 [bold]분석된 작업 유형:[/bold] {task_info['type']}")
        console.print(f"📋 [bold]우선순위:[/bold] {', '.join(task_info['priority'])}")
        console.print("-" * 50)
        
        # 설정
        config = RunnableConfig(
            recursion_limit=15,
            configurable={"thread_id": "multi_agent_1"},
            tags=["multi-agent-workflow"]
        )
        
        # MCP 서버 연결 - 단일 Progress 사용
        console.print("🔌 [bold yellow]MCP 서버에 연결 중...[/bold yellow]")
        
        async with await create_client() as client:
            console.print("✅ [bold green]MCP 서버 연결 완료[/bold green]")
            
            console.print("🛠️  [bold yellow]그래프 생성 중...[/bold yellow]")
            agent = create_multi_agent_graph(client)
            console.print("✅ [bold green]그래프 생성 완료[/bold green]")
            
            # 사용 가능한 도구 확인
            available_tools = {}
            for category in ["search", "file", "code"]:
                tools = get_tools_by_category(client, category)
                available_tools[category] = [tool.name for tool in tools]
            
            # 도구 요약 테이블
            tools_table = Table(title="📊 사용 가능한 도구", show_header=True, header_style="bold magenta")
            tools_table.add_column("카테고리", style="cyan", width=12)
            tools_table.add_column("도구 목록", style="green")
            
            tools_table.add_row("🔍 검색", ", ".join(available_tools.get("search", [])))
            tools_table.add_row("📁 파일", ", ".join(available_tools.get("file", [])))
            tools_table.add_row("💻 코드", ", ".join(available_tools.get("code", [])))
            
            console.print(tools_table)
            
            # 초기 상태 설정
            initial_state = {
                "messages": [HumanMessage(content=user_query)],
                "user_query": user_query,
                "search_results": [],
                "file_results": [],
                "code_results": [],
                "analysis_results": [],
                "available_tools": available_tools,
                "next_agent": "search_node",
                "iteration": 1,
                "task_type": task_info["type"],
                "completed_tasks": []
            }
            
            # 작업 실행 - 단순한 메시지로 대체
            console.print(f"\n⚡ [bold yellow]작업 실행 중...[/bold yellow] (시간이 다소 소요될 수 있습니다)")
            
            response = await agent.ainvoke(initial_state, config=config)
            
            console.print("✅ [bold green]작업 실행 완료[/bold green]")
            
            # 결과 출력
            display_results(response, user_query)
        
    except Exception as e:
        log_error(f"시스템 오류: {e}")
        console.print(f"\n💥 [red]오류 발생: {e}[/red]")
    
    # 시스템 종료 메시지
    console.print("\n" + "="*60, style="bold blue")
    console.print("🔚 [bold cyan]시스템 종료[/bold cyan]")
    console.print("="*60, style="bold blue")
    
# ===== 실행 =====
if __name__ == "__main__":
    # Rich Console로 시작 메시지
    console.print("\n🎉 [bold green]MCP 멀티 에이전트 시스템을 시작합니다![/bold green]")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n⚠️  [yellow]사용자에 의해 중단되었습니다.[/yellow]")
    except Exception as e:
        error_console.print(f"\n💥 시스템 오류: {e}")
    finally:
        console.print("\n👋 [blue]시스템을 종료합니다. 안녕히 가세요![/blue]")