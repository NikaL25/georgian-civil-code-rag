import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_astradb.vectorstores import AstraDBVectorStore
from langchain_groq.chat_models import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv(
    "GROQ_MODEL_NAME", "meta-llama/llama-4-maverick-17b-128e-instruct")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_COLLECTION = os.getenv("ASTRA_DB_COLLECTION", "agentragcoll")

# assert GROQ_API_KEY, "Set GROQ_API_KEY in .env"
# assert OPENAI_API_KEY, "Set OPENAI_API_KEY in .env"


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not set, using fallback if needed")

if ASTRA_TOKEN and ASTRA_ENDPOINT:
    print("Connecting to AstraDB vector store...")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vstore = AstraDBVectorStore(
        api_endpoint=ASTRA_ENDPOINT,
        token=ASTRA_TOKEN,
        collection_name=ASTRA_COLLECTION,
        embedding=emb,
        
    )
else:
    print("AstraDB not configured — load local FAISS from ./faiss_index")
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vstore = FAISS.load_local("./faiss_index", embedding=emb, allow_dangerous_deserialization=True)

retriever = vstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=0.5
)

tools = [
    Tool(
        name="CivilCodeRetrieval",
        func=lambda q: "\n\n".join([doc.page_content for doc in retriever.get_relevant_documents(q)]),  # Возвращает текст чанков как строку
        description="Searches the Georgian Civil Code for relevant articles. Input should be a legal question in Georgian. Returns relevant text chunks."
    )
]

AGENT_PROMPT_TEMPLATE = PromptTemplate(
    template="""
თქვენ ხართ იურიდიული აგენტი, რომელიც პასუხობს მხოლოდ საქართველოს სამოქალაქო კოდექსის საფუძველზე ქართულ ენაზე.

❗ გამოიყენე მხოლოდ CivilCodeRetrieval ინსტრუმენტს კოდექსში ძიებისთვის. არ გამოიგონო ინფორმაცია.
❗ თუ ინფორმაცია არ არის საკმარისი, უპასუხე "არ ვიცი". არ გამოიყენო სხვა წყაროები.
❗ მიუთითე მუხლის ნომერი და გვერდი პასუხში.
❗ პასუხი იყოს მოკლე და ზუსტი.

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}
""",
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    verbose=True, 
    memory=memory,  
    agent_kwargs={"prompt": AGENT_PROMPT_TEMPLATE}  
)

def format_sources(source_docs):
    seen = []
    lines = []
    for d in source_docs:
        art = d.metadata.get("article")
        page = d.metadata.get("page")
        src = ""
        if art:
            src = f"{art}"
            if page:
                src += f" (გვერდი {page})"
        else:
            src = f"გვერდი {page}" if page else d.metadata.get(
                "source", "მონაცემები არ არის")
        if src not in seen:
            seen.append(src)
            lines.append(f"- {src}")
    return "\n".join(lines) if lines else "- წყარო ვერ იქნა იდენტიფიცირებული"


def answer(question: str):
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables")
    res = agent.run(input=question)  
    
    source_docs = retriever.get_relevant_documents(question)
    sources_formatted = format_sources(source_docs)
    
    final = f"{res.strip()}\n\nწყარო(ები):\n{sources_formatted}"
    return final


if __name__ == "__main__":
    print("RAG (Groq) — ask in Georgian. Type 'exit' to quit.")
    while True:
        q = input("Question (ka): ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        print("Thinking...\n")
        print(answer(q))
        print("\n" + "-"*40 + "\n") 