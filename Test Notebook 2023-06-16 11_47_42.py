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

cfn.llm("tell me a joke")

# COMMAND ----------

from evals.cli import oaieval

# COMMAND ----------

args = oaieval.OaiEvalArguments(completion_fn="langchain/llm/mpt-7b-instruct", eval="test-match", debug=False, visible=None, max_samples=3, registry_path=None, cache=True, seed=20220722, user="", record_path=None, dry_run=False, local_run=True, dry_run_logging=True, extra_eval_params="")

# COMMAND ----------

oaieval.run(args)

# COMMAND ----------

# MAGIC %sh cat /tmp/evallogs/230616082256XZQYLX7Q_langchain/llm/mpt-7b-instruct_test-match.jsonl

# COMMAND ----------

from evals.cli import oaievalset

# COMMAND ----------

args_eval_set = oaievalset.OaiEvalSetArguments(model="langchain/llm/mpt-7b-instruct", eval_set="test-basic", resume=True, exit_on_error=True)

# COMMAND ----------

oaievalset.run(args_eval_set, unknown_args=[])

# COMMAND ----------

# MAGIC %sh oaievalset gpt-3.5-turbo test-basic

# COMMAND ----------


