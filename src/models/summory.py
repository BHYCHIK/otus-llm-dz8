from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate

class Summorizer():
    def _get_prompt(self, input_text):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""
                Store the main key points. But be as short as possible.
                Concentrate on recall.
                """),
                HumanMessagePromptTemplate.from_template("Summorize the following text: {input_text}"),
            ]
        )

    def _get_parser(self):
        return StrOutputParser()

    def __init__(self, model):
        self._model = model

    def summorize(self, input_text):
        chain = (self._get_prompt(input_text) | self._model | self._get_parser())
        summary = chain.invoke({'input_text': input_text})
        return summary


class RudeSummorizer(Summorizer):
    def _get_prompt(self, input_text):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""
                You are rude impolite summorizer.
                Store the main key points. But be as short as possible.
                Concentrate on recall.

                Act as if user is a stupid idiot, which get on your nerve.
                """),
                HumanMessagePromptTemplate.from_template("Summorize the following text: {input_text}"),
            ]
        )

class MissySummorizer(Summorizer):
    def _get_prompt(self, input_text):
        return ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template("""
                You are always forgetting some facts. Try to summorize, but loose some facts.
                """),
                HumanMessagePromptTemplate.from_template("Summorize the following text: {input_text}"),
            ]
        )
