from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


# GraphState 상태 정의
class StateStruct(TypedDict, total=False):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    answer: Annotated[str, "Answer"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)
