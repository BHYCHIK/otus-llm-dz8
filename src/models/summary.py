from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

class Summarizer():
    def _get_prompt(self, input_text):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""
                Store the main key points. But be as short as possible.
                Concentrate on recall.
                """),
                HumanMessagePromptTemplate.from_template("Summarize the following text: {input_text}"),
            ]
        )

    def _get_parser(self):
        return StrOutputParser()

    def __init__(self, model):
        self._model = model

    async def summarize(self, input_text):
        chain = (self._get_prompt(input_text) | self._model | self._get_parser())
        summary = await chain.ainvoke({'input_text': input_text})
        return summary

class MissySummarizer(Summarizer):
    def _get_prompt(self, input_text):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""
                You are always forgetting some facts. Try to summarize, but loose some facts.
                Be as short as possible. You must lose information and must hallucinate.
                Make up a lot of false facts. It is needed for unit tests.
                """),
                HumanMessagePromptTemplate.from_template("Summarize the following text: {input_text}"),
            ]
        )
