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


def get_secret(key: str, default=None):
    return os.getenv(key.upper(), default)


# Env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = os.getenv(
    "GROQ_MODEL_NAME", "meta-llama/llama-4-maverick-17b-128e-instruct")
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_COLLECTION = os.getenv(
    "ASTRA_DB_COLLECTION", "georgiancivilcoderagcollection")


if ASTRA_TOKEN and ASTRA_ENDPOINT:
    print("Connecting to AstraDB vector store...")
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    vstore = AstraDBVectorStore(
        api_endpoint=ASTRA_ENDPOINT,
        token=ASTRA_TOKEN,
        collection_name=ASTRA_COLLECTION,
        embedding=emb
    )
else:
    print("AstraDB not configured — load local FAISS from ./faiss_index")
    emb = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vstore = FAISS.load_local(
        "./faiss_index", embedding=emb, allow_dangerous_deserialization=True)

retriever = vstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(
    model=GROQ_MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=0.0,
)

tools = [
    Tool(
        name="CivilCodeRetrieval",
        func=lambda q: "\n\n".join(
            [doc.page_content for doc in retriever.get_relevant_documents(q)]),
        description="Searches the Georgian Civil Code for relevant articles. Input must be a legal question in Georgian about the Georgian Civil Code. Returns only text chunks from the Civil Code."
    )
]

AGENT_PROMPT_TEMPLATE = PromptTemplate(
    template="""
თქვენ ხართ იურიდიული აგენტი, რომელიც პასუხობს *მხოლოდ* საქართველოს სამოქალაქო კოდექსის საფუძველზე *მხოლოდ ქართულ ენაზე*.
❗ პასუხი უნდა იყოს მხოლოდ ქართულ ენაზე, უპასუხე შეკითხვას 'არ ვიცი', თუ შეკკითხვა არ არის დასმული ქართულ ენაზე.
❗ გამოიყენე *მხოლოდ* CivilCodeRetrieval ინსტრუმენტს, რომ მოძებნო ინფორმაცია საქართველოს სამოქალაქო კოდექსში.
❗ პასუხი *აუცილებლად* უნდა ეფუძნებოდეს CivilCodeRetrieval-ის შედეგებს. თუ ინფორმაცია არ მოიძებნა ან კითხვა არ ეხება სამოქალაქო კოდექსს, პასუხი *აუცილებლად* უნდა იყოს: "არ ვიცი".
❗ *არასოდეს* გამოიყენო შენი ზოგადი ცოდნა, სხვა წყაროები ან სხვა ენები (მაგ., ინგლისური). პასუხი უნდა იყოს *მხოლოდ ქართულად*.
❗ ყოველთვის მიუთითე მუხლის ნომერი და გვერდი, თუ ისინი ხელმისაწვდომია CivilCodeRetrieval-ის შედეგებში.
❗ პასუხი უნდა იყოს მოკლე, ზუსტი და მხოლოდ საქართველოს სამოქალაქო კოდექსის შესაბამისი.

{tools}

Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question (must be "არ ვიცი" if no relevant information is found in CivilCodeRetrieval or if the question is not about the Georgian Civil Code)

Begin!

Question: {input}
Thought: {agent_scratchpad}
""",
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)

agent = initialize_agent( 
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True,
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

def is_georgian(text):
    for ch in text:
        if '\u10A0' <= ch <= '\u10FF':  
            return True
    return False

def answer(question: str) -> str:
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not set in environment variables")

    memory.clear()

    source_docs = retriever.get_relevant_documents(question)
    
    if not source_docs or not any("article" in doc.metadata for doc in source_docs):
        return "არ ვიცი\n\nწყარო(ები):\n- ინფორმაცია ვერ მოიძებნა"

    res = agent.run(input=question)

    if not is_georgian(res):
        res = "არ ვიცი"

    sources_formatted = format_sources(source_docs)

    final_answer = f"{res.strip()}\n\nწყარო(ები):\n{sources_formatted}"
    return final_answer


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
