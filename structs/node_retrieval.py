from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from libs.pdf_manager import PDFRetrievalChain
from libs.utils import format_docs
from structs.state_struct import StateStruct


class RetrievalNode:
    def __init__(self, pdf: list[str] = ["documents/doc1.pdf"]):
        # API KEY 정보로드
        load_dotenv()

        # PDF 문서를 로드합니다.
        builder = PDFRetrievalChain(docs=pdf)
        self.pdf_retriever = builder.build_retriever()

    # 문서 검색 노드
    def retrieve_document(self, state: StateStruct) -> StateStruct:
        # 질문을 상태에서 가져옵니다.
        latest_question = state.get("question", "")

        # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
        retrieved_docs = self.pdf_retriever.invoke(latest_question)

        # 검색된 문서를 형식화합니다.(프롬프트 입력으로 넣어주기 위함)
        retrieved_docs = format_docs(retrieved_docs)

        # 검색된 문서를 context 키에 저장합니다.
        state["context"] = retrieved_docs
        return state


def get_role_from_messages(msg: BaseMessage) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    elif isinstance(msg, AIMessage):
        return "assistant"
    else:
        return "assistant"


def messages_to_history(messages: list[BaseMessage]) -> str:
    return "\n".join(
        [f"{get_role_from_messages(msg)}: {msg.content}" for msg in messages]
    )
