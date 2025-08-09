from typing import TypedDict, Annotated, List
from langchain_core.documents import Document
import operator


class GraphState(TypedDict):
    context: Annotated[List[Document], operator.add]
    answer: Annotated[List[Document], operator.add]
    question: Annotated[str, "user question"]
    sql_query: Annotated[str, "sql query"]
    binary_score: Annotated[str, "binary score yes or no"]


def retrieve(state: GraphState) -> GraphState:
    # retrieve: 검색
    documents = "검색된 문서"
    return {"context": documents}


def rewrite_query(state: GraphState) -> GraphState:
    # Query Transform: 쿼리 재작성
    documents = "검색된 문서"
    return GraphState(context=documents)


def llm_gpt_execute(state: GraphState) -> GraphState:
    # LLM 실행
    answer = "GPT 생성된 답변"
    return GraphState(answer=answer)


def llm_claude_execute(state: GraphState) -> GraphState:
    # LLM 실행
    answer = "Claude 의 생성된 답변"
    return GraphState(answer=answer)


def relevance_check(state: GraphState) -> GraphState:
    # Relevance Check: 관련성 확인
    binary_score = "Relevance Score"
    return GraphState(binary_score=binary_score)


def sum_up(state: GraphState) -> GraphState:
    # sum_up: 결과 종합
    answer = "종합된 답변"
    return GraphState(answer=answer)


def search_on_web(state: GraphState) -> GraphState:
    # Search on Web: 웹 검색
    documents = state["context"] = "기존 문서"
    searched_documents = "검색된 문서"
    documents += searched_documents
    return GraphState(context=documents)


def get_table_info(state: GraphState) -> GraphState:
    # Get Table Info: 테이블 정보 가져오기
    table_info = "테이블 정보"
    return GraphState(context=table_info)


def generate_sql_query(state: GraphState) -> GraphState:
    # Make SQL Query: SQL 쿼리 생성
    sql_query = "SQL 쿼리"
    return GraphState(sql_query=sql_query)


def execute_sql_query(state: GraphState) -> GraphState:
    # Execute SQL Query: SQL 쿼리 실행
    sql_result = "SQL 결과"
    return GraphState(context=sql_result)


def validate_sql_query(state: GraphState) -> GraphState:
    # Validate SQL Query: SQL 쿼리 검증
    binary_score = "SQL 쿼리 검증 결과"
    return GraphState(binary_score=binary_score)


def handle_error(state: GraphState) -> GraphState:
    # Error Handling: 에러 처리
    error = "에러 발생"
    return GraphState(context=error)


def decision(state: GraphState) -> GraphState:
    # 의사결정
    decision = "결정"
    # 로직을 추가할 수 가 있고요.

    if state["binary_score"] == "yes":
        return "종료"
    else:
        return "재검색"