from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

class RswikiBot:
    def __init__(self,rag_prompt = hub.pull("rlm/rag-prompt"),db_loc = './chroma.db'):
        self.rag_prompt = rag_prompt
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        self.chroma_db = Chroma(
            embedding_function=self.embeddings,
            persist_directory=db_loc,  # directory to store the vector store
            collection_name="runescape",  # name of the collection
        )
        self.llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

    def retrieve(self,state: State):
        retrieved_docs = self.chroma_db.similarity_search(state["question"])
        return {"context": retrieved_docs}


    def generate(self,state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.rag_prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def ask_llm(self,query):
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        graph = graph_builder.compile()
        return graph.invoke({"question": query})

def main():
    rswb = RswikiBot()
    result = rswb.ask_llm("List all the quests the Age of Chaos")
    print(f'Context: {result["context"]}\n\n')
    print(f'Answer: {result["answer"]}')

if __name__ == "__main__":
    main()