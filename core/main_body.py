from agents.rag_agent import RAGAgent


def main_run():
    rag_agent = RAGAgent()
    rag_agent.do_build()
    while True:
        input_str = input("Chat:")
        print(rag_agent.run_chat(input_str))
