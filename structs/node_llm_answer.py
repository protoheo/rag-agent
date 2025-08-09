from langchain_core.messages import HumanMessage, AIMessage

from models.model_manager import ChainManager
from structs.state_struct import StateStruct


class AnswerNode:
    def __init__(self):
        cm = ChainManager()
        chain = cm.create_chain()
        self.chain = chain

    # 답변 생성 노드
    def llm_answer(self, state: StateStruct) -> StateStruct:
        # 질문을 상태에서 가져옵니다.
        latest_question = state["question"]

        # 검색된 문서를 상태에서 가져옵니다.
        context = state["context"]

        # 체인을 호출하여 답변을 생성합니다.
        response = self.chain.invoke(
            {
                "question": latest_question,
                "context": context,
                "chat_history": messages_to_history(state["messages"]),
            }
        )
        # 생성된 답변, (유저의 질문, 답변) 메시지를 상태에 저장합니다.
        return {
            "answer": response,
            "messages": [("user", latest_question), ("assistant", response)],
        }


def get_role_from_messages(msg):
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    else:
        return "assistant"


def messages_to_history(messages):
    return "\n".join(
        [f"{get_role_from_messages(msg)}: {msg.content}" for msg in messages]
    )