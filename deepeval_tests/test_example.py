import os
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

from deepeval_tests.custom_llm import Mistral7B
from transformers import AutoModelForCausalLM, AutoTokenizer
from deepeval.models.base import DeepEvalBaseModel

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is required!")


# run deepeval test run test_example.py


def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
    )
    assert_test(test_case, [answer_relevancy_metric])


def test_custom_metric():
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output="The seller generiously offer a refund or you get a replacement. When you reach out to them, they can give you back a refund or give you a replacement that will fit your need. Let me know if you have any questions, happy to help!",
        # expected_output="You can return the item back to them and they will send you back with one that fits. However, you can get a complete refund as well.",
        expected_output="The seller generiously offer a refund or you get a replacement. When you reach out to them, they can give you back a refund or give you a replacement that will fit your need. Let me know if you have any questions, happy to help!",
    )
    correctness_metric = GEval(
        name="Correctness",
        criteria="Correctness - determine if the actual output is correct according to the expected output.",
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        strict_mode=False,
        threshold=0.5,
    )

    assert_test(test_case, [correctness_metric])


def test_answer_relevance_using_custom_llm():

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

    mistral_7b = Mistral7B(model=model, tokenizer=tokenizer)  # type: ignore
    answer_relevancy_metric = AnswerRelevancyMetric(model=mistral_7b, threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
    )
    assert_test(test_case, [answer_relevancy_metric])
