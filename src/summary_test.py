import asyncio

from markdown_it.rules_block import reference
from ragas import Dataset, EvaluationDataset, evaluate
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness
from ragas.experiment import experiment
import dotenv
from langchain_openai import ChatOpenAI
from sqlalchemy.engine import row

from models.summory import Summorizer, MissySummorizer, RudeSummorizer
import os
from openai import AsyncOpenAI
import pandas as pd
from pydantic import BaseModel

dotenv.load_dotenv()

ds = Dataset.load(name='Summarization_summary', backend="local/csv", root_dir="./data")

openai = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
    temperature=0.0,
    model='qwen-3-32b'
)

client = AsyncOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
)

judge_llm = llm_factory('qwen-3-32b', client=client, max_tokens=10000, temperature=0.0)

class ExperimentResult(BaseModel):
    faithfulness_score: float
    faithfulness_reason: str


faithfullness = Faithfulness(llm=judge_llm)

@experiment(ExperimentResult, name_prefix="base_scores")
async def get_faithfulness(row, summarizer):
    summary = summarizer.summorize(row['context'])

    faith_result = await faithfullness.ascore(
        response=summary,
        retrieved_contexts=[row['context']],
        user_input=row['question'],
    )

    return ExperimentResult(
        faithfulness_score=faith_result.value,
        faithfulness_reason = faith_result.reason
    )

async def test_summorizer():
    summorizer = Summorizer(openai)
    exp_result = await get_faithfulness.arun(dataset=ds, name="base_summarizer", summarizer=summorizer)
    print(exp_result)
    for exp in exp_result:
        print(exp)

if __name__ == '__main__':
    asyncio.run(test_summorizer())