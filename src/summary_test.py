import asyncio

from ragas import Dataset
from ragas.llms import llm_factory
from ragas.metrics.collections import Faithfulness, SummaryScore, ResponseGroundedness
from ragas.experiment import experiment
import dotenv
from langchain_openai import ChatOpenAI

from models.summary import Summarizer, MissySummarizer
import os
from openai import AsyncOpenAI
from pydantic import BaseModel
import time
import pytest

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


YANDEX_CLOUD_FOLDER = os.getenv("EXPENSIVE_JUDGE_PROJECT")
YANDEX_CLOUD_MODEL = "aliceai-llm/latest"

expensive_client = AsyncOpenAI(
    api_key=os.getenv("EXPENSIVE_JUDGE_API_KEY"),
    base_url=os.getenv("EXPENSIVE_JUDGE_BASE_URL"),
    project=os.getenv("EXPENSIVE_JUDGE_PROJECT"),
)

expensive_judge_llm = llm_factory(os.getenv('EXPENSIVE_JUDGE_MODEL'), client=expensive_client, max_tokens=99000, temperature=0.0)

class ExperimentResult(BaseModel):
    faithfulness_score: float
    summary_score: float
    response_groundedness_score: float
    rouge1_score: float
    rouge2_score: float
    rougeL_score: float
    rougeLsum_score: float
    bleu_score: float
    semantic_similarity_precision: float
    semantic_similarity_recall: float
    semantic_similarity_f1: float
    time_to_summarize: float
    time_to_calc_summary_score: float
    time_to_calc_faithfulness: float
    time_to_response_groundedness: float
    context: str
    summary: str


faithfulness_metric = Faithfulness(llm=judge_llm)
summary_score_metric = SummaryScore(llm=expensive_judge_llm)
response_groundedness_metric = ResponseGroundedness(llm=judge_llm)

semantic_similarity_scorer = evaluate.load('bertscore')
rouge_scorer = evaluate.load('rouge')
bleu_scorer = evaluate.load('bleu')


@experiment(ExperimentResult, name_prefix="base_scores")
async def get_summary_scores(row, summarizer, lock):
    start = time.time()

    #async with lock:
    summary = await summarizer.summarize(row['context'])
    end = time.time()
    time_to_summarize = end - start

    start = time.time()
    summary_score_result_waiter = summary_score_metric.ascore(
        reference_contexts=[row['context']],
        response=summary,
    )

    summary_score_result = await summary_score_result_waiter
    end = time.time()
    time_to_calc_summary_score = end - start

    start = time.time()
    faith_result_waiter = faithfulness_metric.ascore(
        response=summary,
        retrieved_contexts=[row['context']],
        user_input=row['question'],
    )

    faith_result = await faith_result_waiter
    end = time.time()
    time_to_calc_faithfulness = end - start

    start = time.time()
    response_groundedness_waiter = response_groundedness_metric.ascore(
        response=summary,
        retrieved_contexts=[row['context']],
    )

    response_groundedness_result = await response_groundedness_waiter
    end = time.time()
    time_to_response_groundedness = end - start

    rouge = rouge_scorer.compute(predictions=[summary], references=[row['ground_truth']])
    bleu = bleu_scorer.compute(predictions=[summary], references=[row['ground_truth']])
    semantic_similarity = semantic_similarity_scorer.compute(predictions=[summary],
                                                             references=[row['ground_truth']],
                                                             model_type="xlm-roberta-large",
                                                             lang="en")

    return ExperimentResult(
        faithfulness_score=faith_result.value,
        summary_score=summary_score_result.value,
        response_groundedness_score=response_groundedness_result.value,
        rouge1_score=rouge['rouge1'],
        rouge2_score=rouge['rouge2'],
        rougeL_score=rouge['rougeL'],
        rougeLsum_score=rouge['rougeLsum'],
        bleu_score=bleu['bleu'],
        semantic_similarity_precision=semantic_similarity['precision'][0],
        semantic_similarity_recall=semantic_similarity['recall'][0],
        semantic_similarity_f1=semantic_similarity['f1'][0],
        time_to_summarize=time_to_summarize,
        time_to_calc_summary_score=time_to_calc_summary_score,
        time_to_calc_faithfulness=time_to_calc_faithfulness,
        time_to_response_groundedness=time_to_response_groundedness,
        context=row['context'],
        summary=summary,
    )

@pytest.mark.asyncio
async def test_summarizer():
    summarizer = Summarizer(openai)
    lock = asyncio.Lock()
    exp_result = await get_summary_scores.arun(dataset=ds, name="base_summarizer", summarizer=summarizer, lock=lock)
    exp_df = exp_result.to_pandas()

    faithfulness_score = exp_df['faithfulness_score'].mean()
    summary_score = exp_df['summary_score'].mean()
    response_groundedness_score = exp_df['response_groundedness_score'].mean()
    rouge1_score = exp_df['rouge1_score'].mean()
    rouge2_score = exp_df['rouge1_score'].mean()
    rougeL_score = exp_df['rougeL_score'].mean()
    rougeLsum_score = exp_df['rougeLsum_score'].mean()
    bleu_score = exp_df['bleu_score'].mean()
    semantic_similarity_precision = exp_df['semantic_similarity_precision'].mean()
    semantic_similarity_recall = exp_df['semantic_similarity_recall'].mean()
    semantic_similarity_f1 = exp_df['semantic_similarity_f1'].mean()

    print('Faithfulness mean ', faithfulness_score)
    print('Summary score mean ', summary_score)
    print('Response groundedness score mean ', response_groundedness_score)
    print('Rouge1 score mean ', rouge1_score)
    print('Rouge2 score mean ', rouge2_score)
    print('RougeL score mean ', rougeL_score)
    print('RougeLsum score mean ', rougeLsum_score)
    print('Bleu score mean ', bleu_score)
    print('Semantic similarity precision mean ', semantic_similarity_precision)
    print('Semantic similarity recall mean ', semantic_similarity_recall)
    print('Semantic similarity f1 mean ', semantic_similarity_f1)

    assert faithfulness_score > 0.96
    assert summary_score > 0.66
    assert response_groundedness_score > 0.99
    assert rouge1_score > 0.37
    assert rouge2_score > 0.37
    assert rougeL_score > 0.28
    assert rougeLsum_score > 0.28
    assert bleu_score > 0.035
    assert semantic_similarity_precision > 0.85
    assert semantic_similarity_recall > 0.85
    assert semantic_similarity_f1 > 0.85

    faithfulness_score_min = exp_df['faithfulness_score'].min()
    summary_score_min = exp_df['summary_score'].min()
    response_groundedness_score_min = exp_df['response_groundedness_score'].min()

    print('Faithfulness min ', faithfulness_score_min)
    print('Summary score min ', summary_score_min)
    print('Response groundedness min ', response_groundedness_score_min)

    assert faithfulness_score_min > 0.89
    assert summary_score_min > 0.33
    assert response_groundedness_score_min > 0.92


@pytest.mark.asyncio
async def test_missy_summarizer():
    summarizer = MissySummarizer(openai)
    lock = asyncio.Lock()
    exp_result = await get_summary_scores.arun(dataset=ds, name="missy_summarizer", summarizer=summarizer, lock=lock)
    exp_df = exp_result.to_pandas()

    faithfulness_score = exp_df['faithfulness_score'].mean()
    summary_score = exp_df['summary_score'].mean()
    response_groundedness_score = exp_df['response_groundedness_score'].mean()
    rouge1_score = exp_df['rouge1_score'].mean()
    rouge2_score = exp_df['rouge1_score'].mean()
    rougeL_score = exp_df['rougeL_score'].mean()
    rougeLsum_score = exp_df['rougeLsum_score'].mean()
    bleu_score = exp_df['bleu_score'].mean()
    semantic_similarity_precision = exp_df['semantic_similarity_precision'].mean()
    semantic_similarity_recall = exp_df['semantic_similarity_recall'].mean()
    semantic_similarity_f1 = exp_df['semantic_similarity_f1'].mean()

    print('Faithfulness mean ', faithfulness_score)
    print('Summary score mean ', summary_score)
    print('Response groundedness score mean ', response_groundedness_score)
    print('Rouge1 score mean ', rouge1_score)
    print('Rouge2 score mean ', rouge2_score)
    print('RougeL score mean ', rougeL_score)
    print('RougeLsum score mean ', rougeLsum_score)
    print('Bleu score mean ', bleu_score)
    print('Semantic similarity precision mean ', semantic_similarity_precision)
    print('Semantic similarity recall mean ', semantic_similarity_recall)
    print('Semantic similarity f1 mean ', semantic_similarity_f1)

    assert faithfulness_score > 0.82
    assert summary_score > 0.57
    assert response_groundedness_score > 0.82
    assert rouge1_score > 0.3
    assert rouge2_score > 0.3
    assert rougeL_score > 0.2
    assert rougeLsum_score > 0.2
    assert bleu_score > 0.025
    assert semantic_similarity_precision > 0.84
    assert semantic_similarity_recall > 0.84
    assert semantic_similarity_f1 > 0.84

    faithfulness_score_min = exp_df['faithfulness_score'].min()
    summary_score_min = exp_df['summary_score'].min()
    response_groundedness_score_min = exp_df['response_groundedness_score'].min()

    print('Faithfulness min ', faithfulness_score_min)
    print('Summary score min ', summary_score_min)
    print('Response groundedness min ', response_groundedness_score_min)

    assert faithfulness_score_min > 0.4
    assert summary_score_min > 0.33
    assert response_groundedness_score_min > 0.4


async def main():
    await test_summarizer()
    await test_missy_summarizer()

if __name__ == '__main__':
    asyncio.run(main())