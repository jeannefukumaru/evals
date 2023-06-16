# Databricks notebook source
# MAGIC %pip install evals langchain huggingface_hub

# COMMAND ----------

import os
os.environ["OPENAI_API_KEY"]="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"

# COMMAND ----------

os.environ["HUGGINGFACEHUB_API_TOKEN"]="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"

# COMMAND ----------

from evals.completion_fns.langchain_llm import LangChainLLMCompletionFn

# COMMAND ----------

from langchain import HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(model_id="bigscience/bloom-1b7", task="text-generation", model_kwargs={"temperature":0, "max_length":64})

# COMMAND ----------

from langchain import PromptTemplate,  LLMChain

template = """Question: {question}

Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "What is electroencephalography?"

print(llm_chain.run(question))

# COMMAND ----------

llm

# COMMAND ----------

cluster_id = '0608-083505-hk6qa5vj'
cfn = LangChainLLMCompletionFn("Databricks", {"cluster_id":cluster_id, "cluster_driver_port":7777, "model_kwargs":{"temperature": .4}})

# COMMAND ----------

cfn.llm("What is Databricks")

# COMMAND ----------

from evals.cli import oaieval

# COMMAND ----------

args = oaieval.OaiEvalArguments(completion_fn="langchain/llm/mpt-7b-instruct",eval="test-math", debug=True, visible=True, max_samples=3, registr)

# COMMAND ----------

# MAGIC %sh oaieval langchain/llm/mpt-7b-instruct test-match

# COMMAND ----------


