from agents.rag_agent import RAGAgent


def main_run():
    rag_agent = RAGAgent()
    while True:
        input_str = input("Chat:")
        outputs = rag_agent.run_chat(input_str)
        for k, v in outputs.items():
            if k == "answer":
                splut = v.split("<|assistant|>")
                answer = splut[-1].replace("<|im_end|>", "").strip()
                print(f"{k}: {answer}")
            else:
                print(f"{k}: {v}")
            print("-----")
