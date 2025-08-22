from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


def model_load():
    model_name = "skt/A.X-3.1-Light"

    with open("prompts/system/llm.txt", "r", encoding="utf-8") as f:
        content = f.read()

    prompt = PromptTemplate.from_template(content)

    llm = HuggingFacePipeline.from_model_id(
        model_id=model_name,
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 128,
            "top_k": 50,
            "temperature": 0.1,
        },
    )

    chain = prompt | llm | StrOutputParser()

    return chain
