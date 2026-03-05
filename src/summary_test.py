import asyncio

from ragas import Dataset
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, SummaryScore
from ragas.experiment import experiment
import dotenv
from langchain_openai import ChatOpenAI

from models.summary import Summarizer, MissySummarizer, RudeSummarizer
import os
from openai import AsyncOpenAI
from pydantic import BaseModel
import time

import evaluate

dotenv.load_dotenv()

ds = Dataset.load(name='Summarization_summary', backend="local/csv", root_dir="./data")

openai = ChatOpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
    temperature=0.0,
    model='qwen3-coder'
)

client = AsyncOpenAI(
    api_key=os.getenv("JUDGE_API_KEY"),
    base_url=os.getenv("JUDGE_BASE_URL"),
    project=os.getenv("JUDGE_PROJECT"),
)

judge_llm = llm_factory(os.getenv("JUDGE_MODEL"), client=client, max_tokens=99000, temperature=0.0)

class ExperimentResult(BaseModel):
    faithfulness_score: float
    summary_score: float
    rouge1_score: float
    rouge2_score: float
    rougeL_score: float
    rougeLsum_score: float
    time_to_summarize: float
    time_to_calc_summary_score: float
    time_to_calc_faithfulness: float
    context: str


faithfulness = Faithfulness(llm=judge_llm)
summary_score = SummaryScore(llm=judge_llm) #TODO use exapnsive llm LATER

@experiment(ExperimentResult, name_prefix="base_scores")
async def get_summary_scores(row, summarizer, lock):
    start = time.time()

    #async with lock:
    summary = await summarizer.summarize(row['context'])
    end = time.time()
    time_to_summarize = end - start

    start = time.time()
    #TODO: uncomment, when ready. It is very expensive
    #summary_score_result_waiter = summary_score.ascore(
    #    reference_contexts=[row['context']],
    #    response=summary,
    #)

    #summary_score_result = await summary_score_result_waiter
    end = time.time()
    time_to_calc_summary_score = end - start

    start = time.time()
    faith_result_waiter = faithfulness.ascore(
        response=summary,
        retrieved_contexts=[row['context']],
        user_input=row['question'],
    )

    faith_result = await faith_result_waiter
    end = time.time()
    time_to_calc_faithfulness = end - start

    rouge_scorer = evaluate.load('rouge')
    rouge = rouge_scorer.compute(predictions=[summary], references=[row['ground_truth']])

    return ExperimentResult(
        faithfulness_score=faith_result.value,
        summary_score=1.0,#summary_score_result.value, #TODO: Uncomment before prod
        rouge1_score=rouge['rouge1'],
        rouge2_score=rouge['rouge2'],
        rougeL_score=rouge['rougeL'],
        rougeLsum_score=rouge['rougeLsum'],
        time_to_summarize=time_to_summarize,
        time_to_calc_summary_score=time_to_calc_summary_score,
        time_to_calc_faithfulness=time_to_calc_faithfulness,
        context=row['context'],
    )

async def test_summarizer():
    summarizer = Summarizer(openai)
    lock = asyncio.Lock()
    exp_result = await get_summary_scores.arun(dataset=ds, name="base_summarizer", summarizer=summarizer, lock=lock)
    exp_df = exp_result.to_pandas()
    print('Faithfulness mean ', exp_df['faithfulness_score'].mean())
    print('Summary score mean ', exp_df['summary_score'].mean())
    print('Rouge1 score mean ', exp_df['rouge1_score'].mean())
    print('Rouge2 score mean ', exp_df['rouge2_score'].mean())
    print('RougeL score mean ', exp_df['rougeL_score'].mean())
    print('RougeLsum score mean ', exp_df['rougeLsum_score'].mean())


async def test_missy_summarizer():
    summarizer = MissySummarizer(openai)
    lock = asyncio.Lock()
    exp_result = await get_summary_scores.arun(dataset=ds, name="missy_summarizer", summarizer=summarizer, lock=lock)
    exp_df = exp_result.to_pandas()
    print('Faithfulness mean ', exp_df['faithfulness_score'].mean())
    print('Summary score mean ', exp_df['summary_score'].mean())
    print('Rouge1 score mean ', exp_df['rouge1_score'].mean())
    print('Rouge2 score mean ', exp_df['rouge2_score'].mean())
    print('RougeL score mean ', exp_df['rougeL_score'].mean())
    print('RougeLsum score mean ', exp_df['rougeLsum_score'].mean())

async def main():
    await test_summarizer()
    await test_missy_summarizer()

if __name__ == '__main__':
    asyncio.run(main())