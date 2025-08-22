from typing import Annotated, Any

from langchain_core.runnables import Runnable


class MainRouter(Runnable):
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def invoke(
        self, input: Annotated[dict[Any, Any], "State"], config=None, **kwargs: Any
    ) -> dict:
        while True:
            is_shot = self.model_manager.run(input["messages"], shot_mode=True).strip()
            print("MainRouter Result------->", is_shot)
            if is_shot.lower().replace(".", "") in ["no", "yes"]:
                break

        if "no" in is_shot.lower():
            return {"next": "sLLM", "messages": input["messages"]}
        else:
            return {"next": "CloudLLM", "messages": input["messages"]}

        # 일단은 모델 넣어야되는데 지금은 그냥 단순 분기 생성

        # if len(state["messages"]) > 4:
        #     print("MainRouter --> max_length!")
        #     return {"next": "END", "messages": state["messages"]}
