from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    index_name = 'tax-markdown-index'
    database = PineconeVectorStore.from_existing_index(embedding=embedding, index_name = index_name)
    retriever = database.as_retriever(search_kwargs={'k':4})
    return retriever

def get_llm(model='gpt-4o'):
    llm = ChatOpenAI(model=model)
    return llm

def get_rag_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Answer all questions to the best of your ability."),
            ("human", "{question}"),
        ]
    )
    # 문자열로 바로 받으려면 OutputParser를 연결
    chain = prompt | llm | StrOutputParser()
    return chain

def get_ai_message(user_message):
    chain = get_rag_chain()
    ai_message = chain.invoke({"question": user_message})
    return ai_message