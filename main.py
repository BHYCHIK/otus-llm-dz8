import os
from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate
import dotenv

dotenv.load_dotenv()

openai = OpenAI(
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_BASE_URL"),
    temperature=0.0,
    model='qwen-3-32b'
)

def get_parser():
    return StrOutputParser()

def get_prompt(input_text):
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

def get_chain(input_text):
    return get_prompt(input_text) | openai | get_parser()

def summorize(input_text):
    chain = get_chain(input_text)
    summary = chain.invoke({'input_text': input_text})
    return summary

def main():
    summary = summorize("""
Design documentation is a crucial part of the UX workflow that unfortunately often gets looked upon as something not worthy of wasting time and effort on. Meanwhile, it is the most trustworthy way of bringing order to the process and sharing all the details about the product development with everyone involved. Basically, it is a set of documents that record all the steps, details, descriptions and explanations of every action and decision taken and performed while creating the product. 
The goals of UX design documentation:

    Documenting the process

Documentation helps to write down and preserve everything that took place during the development process, so in case of any issues it serves as a reliable source of information. 

    Sharing info with the team

Documented information can be shared with everyone involved in the UX process, including all the team members and clients. It is especially important for large teams with remote workers. 

    Seeing the whole picture

Writing everything down helps evaluate all the actions and decisions, find mistakes and missing pieces, re-think certain choices and see the project as the whole picture. 
What to include in UX design documentation

There are no certain rules on what exactly should or should not be documented in UX design – it depends on the product, peculiarities of the certain process, team requests and other factors. The more info is written down, the better, but here are the key ideas on what's worthy to include in UX design documentation:

    General information 

As any document, design documentation should start with the general information about the product and its features. Write out the product's purpose, target audience and peculiarities: it should be clear why you're building it and which design methods you're going to apply. 

    Design guidelines 

Design guidelines are principles that the team should mutually stick to when creating the product. They should be carefully thought out depending on the project and clearly articulated for everyone to comprehend, as they will serve as a foundation for the whole process. For example, the team can agree on using certain tools, providing info in a certain way, using certain methods, etc. 

    Key decisions explanation

Documenting rationale means providing written explanation of the exact reason for a certain design decision you've made. It helps you understand your own thinking process better and lets others look at the problem through your eyes. This part of documentation can also be useful in case you're rethinking your design choices later in the process and want to freshen up what led you to this or that decision. 

    Workflow patterns

Throughout the working process, designers tend to stumble upon certain solutions that can be re-applied many times in different parts of the product. Writing these patterns down helps communicate them to the team without wasting time on re-explaining it each time. 

    Issues and edge cases

As weird as it might seem, various issues and edge cases are also worth documenting. It may be useful for the future bug analysis or to prevent any unpleasant "surprises" at the production stage. It's better to write these things down while the product is still being worked on rather than stumble on some issues before the due date without any documented plan on what to do in such cases. 

    Ideas for the product's further development

Apart from documenting everything you're doing with the product now, make sure to also write down your thoughts and ideas on how it can develop in the future — for example, how its current features can be enhanced or which new ones can be added. You won't necessarily use all these ideas later, but it might help the development team preserve continuity in user experience with the future updates. 
How to write UX design documentation more effectively 

Design documentation isn't only about writing down your ideas and actions while developing a product. It also requires mastering the discipline, learning to hear the team's opinion on your documentation and being able to explain everything written in a way that even a non-professional can grasp it clearly. Here are a few tips on how to write design documentation more effectively:

    Discuss documentation with the team 

UX design is team work, so it is necessary to discuss certain ways and methods of documentation with the team beforehand. The team should mutually decide what they think is necessary to document and what's not, how to better integrate documenting in the workflow, which tools to use for sharing documentation with everyone involved, how to form documentation templates to make the process less time-consuming, etc. Then your job is to choose middle-ground on the basis of that discussion and turn the  documentation process into an acceptable routine. 

    Take a disciplined approach

Documentation should become a part of your workflow, which means you must take a disciplined approach to writing it down and make it a habit. It's easier to record things straightaway rather than try to recall all the details when the time has passed. When you don't have enough time to do it properly, at least write down the key info in drafts so as not to miss anything. 

    Show the context before the details

Details are surely a necessary part of design documentation – but it is necessary to give them in a context, otherwise it might cause misinterpretation. For example, when describing a new feature, make sure to explain how it can be reached for from the main menu and how it is integrated into the whole picture. 

    Provide explanations for the third parties

As we said, design documentation is aimed not only at the actual design team, but also at those who are not so familiar with the peculiarities and glossary of UX design, including the third parties like clients and stakeholders. To make your documentation easier to understand, provide extra explanations where necessary – for example, when using special terms and referring to specific features or professional tools. 

    Use visual representations 

In some cases, describing design decisions in a text might be confusing and hard to comprehend – for example, when you have to explain variables or consequential actions to reach a certain feature. Here, visual representation comes in handy: use multi-colored matrix, charts, graphs and other ways to show your design ideas without making the reader have to visualize it in their head by reading long text. 
Conclusion 

Design documentation plays an important role in the UX design, and it should be an integral part of the team's workflow. Making time to write down your design decisions eliminates the need to keep everything in your head and lets you track your progress through the whole process. It also serves as a great tool to present and share design decisions with everyone involved, from the other team members to stakeholders, making it easier to collaborate and share ideas. 
    """)
    print(summary)

if __name__ == "__main__":
    main()