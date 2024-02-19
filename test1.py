import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

from langchain import HuggingFaceHub
# https://huggingface.co/google/flan-t5-xl
llm = HuggingFaceHub(repo_id="google-bert/bert-large-uncased-whole-word-masking-finetuned-squad", model_kwargs={"temperature":0, "max_length":64})

llm("What would be a good company name for a company that makes colorful socks?")