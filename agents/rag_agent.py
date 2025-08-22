import random

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from structs.node_llm_answer import AnswerNode
from structs.node_retrieval import RetrievalNode
from structs.state_struct import StateStruct


def relevance_check():
    choices = ["valid", "invalid"]
    return random.choice(choices)


# GraphState 상태 정의
class RAGAgent:
    def __init__(self):
        self.app = self.do_build()

    def do_build(self):
        workflow = StateGraph(StateStruct)
        rn = RetrievalNode()
        an = AnswerNode()
        # sn = ...

        workflow.add_node("retrieve", rn.retrieve_document)
        workflow.add_node("llm_answer", an.llm_answer)
        # workflow.add_node("search", sn.search_internet)

        # 엣지 정의
        # workflow.add_conditional_edges(
        #     "retrieve", relevance_check, {"valid": "llm_answer", "invalid": "search"}
        # )
        workflow.add_edge("retrieve", "llm_answer")  # 검색 -> 답변
        workflow.add_edge("llm_answer", END)  # 답변 -> 종료

        # 그래프 진입점 설정
        workflow.set_entry_point("retrieve")

        # 체크포인터 설정
        memory = MemorySaver()

        # 컴파일
        app = workflow.compile(checkpointer=memory)
        return app

    def run_chat(self, prompt: str):
        return self.app.invoke(
            {"question": prompt},
            config={"configurable": {"thread_id": "user-001"}},  # <- 꼭 넣기
        )
