import os

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    max_tokens=1024,
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You answer concisely in one paragraph."),
        ("human", "{question}"),
    ]
)

chain = prompt | llm | StrOutputParser()
