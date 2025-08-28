from operator import itemgetter

import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


class ChainManager:
    def __init__(self, model=None):
        self.model = model
        self.prompt_dir = "prompts"

    def load_default_model(self):
        model_name = "skt/A.X-3.1-Light"

        llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            pipeline_kwargs={
                "max_new_tokens": 256,
                "top_k": 3,
                "temperature": 0.1,
            },
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=0,
        )
        return llm

    def create_prompt(self):
        prompt_path = self.prompt_dir + "/system/llm.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read()

        prompt = PromptTemplate.from_template(content)
        return prompt

    def create_chain(self):
        if self.model is None:
            model = self.load_default_model()
        else:
            model = self.model

        prompt = self.create_prompt()
        chain = (
            {
                "question": itemgetter("question"),
                "context": itemgetter("context"),
                "chat_history": itemgetter("chat_history"),
            }
            | prompt
            | model
            | StrOutputParser()
        )
        return chain
