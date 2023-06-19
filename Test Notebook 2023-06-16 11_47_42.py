# Databricks notebook source
# MAGIC %md 
# MAGIC # Various setup stuff

# COMMAND ----------

# MAGIC %pip install langchain huggingface_hub

# COMMAND ----------

## need this as a workaround to install the evals package in editable mode 
%sh 
touch setup.cfg
pip install -e .
rm setup.cfg

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

os.environ["DATABRICKS_HOST"]="xxx"
os.environ["DATABRICKS_TOKEN"]=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### This is the completion_fn we will be using. There are others in the registry and we could also consider using our own

# COMMAND ----------

from evals.completion_fns.langchain_llm import LangChainLLMCompletionFn

# COMMAND ----------

cluster_id = '0608-083505-hk6qa5vj'
cfn = LangChainLLMCompletionFn("Databricks", {"cluster_id":cluster_id, "cluster_driver_port":7777, "model_kwargs":{"temperature": .4}})

# COMMAND ----------

cfn.llm("tell me a joke")

# COMMAND ----------

# MAGIC %md 
# MAGIC ### This is normally a cli call, but we can import the functions and use them directly in our notebook

# COMMAND ----------

# MAGIC %md 
# MAGIC #### test with a single eval before trying a set

# COMMAND ----------

from evals.cli import oaieval

# COMMAND ----------

args = oaieval.OaiEvalArguments(completion_fn="langchain/llm/mpt-7b-instruct", eval="test-match", debug=False, visible=None, max_samples=3, registry_path=None, cache=True, seed=20220722, user="", record_path=None, dry_run=False, local_run=True, dry_run_logging=True, extra_eval_params="")

# COMMAND ----------

oaieval.run(args)

# COMMAND ----------

# MAGIC %sh cat /tmp/evallogs/2306170042326BA2IJSB_langchain/llm/mpt-7b-instruct_test-match.jsonl

# COMMAND ----------

# MAGIC %md 
# MAGIC #### test with oaievalset

# COMMAND ----------

from evals.cli import oaievalset

# COMMAND ----------

args_eval_set_mpt = oaievalset.OaiEvalSetArguments(model="langchain/llm/mpt-7b-instruct", eval_set="test-all-dbx", resume=True, exit_on_error=False)

# COMMAND ----------

oaievalset.run(args_eval_set_mpt, unknown_args=[])

# COMMAND ----------

args_eval_set_dolly = oaievalset.OaiEvalSetArguments(model="langchain/llm/dolly-v2-12b", eval_set="test-all-dbx", resume=True, exit_on_error=False)
oaievalset.run(args_eval_set_dolly, unknown_args=[])

# COMMAND ----------

dbutils.fs.cp("file:/tmp/evallogs", "dbfs:/jeanne.choo@databricks.com/oaievallogs/dolly-12b", True)

# COMMAND ----------

dbutils.fs.ls("dbfs:/jeanne.choo@databricks.com/oaievallogs/dolly-12b")

# COMMAND ----------

dbutils.fs.ls("dbfs:/jeanne.choo@databricks.com/oaievallogs/230617094233QEMIJ4WH_langchain/llm/mpt-7b-instruct_pattern_identification.dev.v0.jsonl")

# COMMAND ----------

import os

# COMMAND ----------

base_path = "/dbfs/jeanne.choo@databricks.com/oaievallogs"
directory_ls = os.listdir(base_path)
directory_ls

# COMMAND ----------

events_fps = []
for directory in directory_ls:
  directory_path = os.path.join(base_path, directory)
  for cur_path, directories, files in os.walk(directory_path):
    if len(files) > 0:
      events_fps.append(os.path.join(cur_path, files[0]))

# COMMAND ----------

events_fps

# COMMAND ----------

# MAGIC %pip install jsonlines

# COMMAND ----------

import jsonlines
with jsonlines.open("/dbfs/jeanne.choo@databricks.com/oaievallogs/2306170105127D7W6P3W_langchain/llm/mpt-7b-instruct_test-fuzzy-match.s1.simple-v0.jsonl") as reader:
    for obj in reader:
      print(obj)

# COMMAND ----------

from collections import defaultdict

def def_value():
    return "Not Present"
  

processed_evals = []
for e in events_fps[100:103]:
  with jsonlines.open(e, "r") as fp:
    for obj in fp:
     print(obj)
      # events = fp.read()
      # print(events)
      # processed_evals_row = defaultdict(def_value)
      # print(events["spec"])
      # processed_evals_row["completion_fns"] = events["spec"]["completion_fns"][0]
      # processed_evals_row["eval_name"] = events["spec"]["eval_name"]
      # processed_evals_row["created_at"] = events["spec"]["created_at"]
      # if "final_report" in events["spec"].keys():
      #   processed_evals_row["final_report"] = events["spec"]["final_report"]
      # processed_evals.append(processed_evals_row)

# COMMAND ----------

json.loads(events)

# COMMAND ----------

# MAGIC %pip install evals

# COMMAND ----------

# MAGIC %sh oaieval gpt-3.5-turbo chinese_hard_translations

# COMMAND ----------


