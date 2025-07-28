from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langchain_chroma import Chroma
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langgraph.graph import START, StateGraph
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from redisvl.extensions.llmcache import SemanticCache
import asyncio

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
        self.llmcache = SemanticCache(
            name="llmcache",
            redis_url="redis://localhost:6379",
            distance_threshold=0.02
        )
        graph_builder = StateGraph(State).add_sequence([self.retrieve, self.generate])
        graph_builder.add_edge(START, "retrieve")
        self.graph = graph_builder.compile()

    async def retrieve(self,state: State):
        retrieved_docs = await self.chroma_db.asimilarity_search(state["question"])
        return {"context": retrieved_docs}


    async def generate(self,state: State):
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = await self.rag_prompt.ainvoke({"question": state["question"], "context": docs_content})
        response = await self.llm.ainvoke(messages)
        return {"answer": response.content}
    
    async def ask_llm(self,query):
        result = await self.llmcache.acheck(
            prompt = query,
            num_results = 1,
            return_fields = ['prompt', 'response']
        )

        if result:
            print('Cache Hit!')
            key_dict = {'prompt':'question','response':'answer'}
            return {key_dict[k]: result[0][k] for k in ('prompt', 'response') if k in result[0]}

        else:
            ans = await self.graph.ainvoke({"question": query})
            await self.llmcache.astore(
                prompt = query,
                response = ans["answer"]
            )
            print('Cache Set!')            
            return ans

async def main():
    rswb = RswikiBot()
    result = await rswb.ask_llm("What all quests are there in Age of Chaos")
    # result = rswb.ask_llm("List all the quests the Age of Chaos")
    # print(f'Context: {result["context"]}\n\n')
    # print(f'Answer: {result["answer"]}')
    print(result)

if __name__ == "__main__":
    asyncio.run(main())