import os

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from exp import get_questions, write_ans

load_dotenv()


def load_docs(folder_path: str = "./testdata"):
    loader = DirectoryLoader(folder_path, glob="**/*.md", show_progress=True, use_multithreading=True, loader_cls=UnstructuredMarkdownLoader)
    docs = loader.load()
    return docs


def get_chain(retriever):
    template = """根據下列內容回答問題，請使用繁體中文回答:
    {context}

    問題: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOllama(model="qwen:4b", temperature=0)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) | prompt | model | StrOutputParser()

    chain = RunnableParallel({"context": retriever, "question": RunnablePassthrough()}).assign(answer=rag_chain_from_docs)
    return chain


def parse_answer(answer):
    print(answer["answer"])
    print("Source: ")
    for doc in answer["context"]:
        print(doc.metadata["source"])


def main():
    permanent_db_path = "faiss_index"
    embeddings = OllamaEmbeddings(model="qwen:4b")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50, add_start_index=True)
    docs = load_docs(folder_path="./testdata")
    all_splits = text_splitter.split_documents(docs)

    if os.path.exists(permanent_db_path):
        vectorstore = FAISS.load_local(permanent_db_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(all_splits, embeddings)

    bm25_retriever = BM25Retriever.from_documents(all_splits)
    bm25_retriever.k = 2
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    chain = get_chain(ensemble_retriever)

    questions = get_questions()
    exp_ans = []

    for i, question in enumerate(questions):
        rag_ans = chain.invoke(question[1])["answer"]
        exp_ans.append(questions[i] + [rag_ans])

    write_ans(exp_ans)

    # input_text = input(">>> ")
    # while input_text.lower() != "bye":
    #     parse_answer(chain.invoke(input_text))
    #     input_text = input(">>> ")
    vectorstore.save_local(permanent_db_path)


if __name__ == "__main__":
    main()
