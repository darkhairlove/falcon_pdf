import chainlit as cl
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ArxivRetriever
from langchain.chains import RetrievalQA
from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import langchain
langchain.verbose = False

huggingfacehub_api_token = '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'

repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.5, "max_new_tokens": 64})


retriever = ArxivRetriever(load_max_docs=2)
docs = retriever.get_relevant_documents(query="2304.06035")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splits = text_splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings()

vectordb = Chroma.from_documents(documents=splits, embedding=embeddings)


prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say “Sorry, con’t answer your question, try to ask it in a different way”, don't try to make up an answer. Use the only the following pieces of context, 
don't use your own knowledge.
    {context}
    Question: {question}"""


@cl.on_chat_start
def main():
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=['context', "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    llm_chain = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}), chain_type_kwargs=chain_type_kwargs, return_source_documents=True)

    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: str):
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    result = llm_chain({"query": message})
    await cl.Message(content=result["result"]).send()
