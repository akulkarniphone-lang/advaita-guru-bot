import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# 1. Load Secrets
load_dotenv() 

# Get keys explicitly from .env
google_key = os.getenv("GOOGLE_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

if not google_key or not pinecone_key:
    raise ValueError("❌ CRITICAL ERROR: API Keys not found. Check your .env file.")

# --- CRITICAL FIX: FORCE ENVIRONMENT VARIABLES ---
# The Pinecone library looks for this specific variable name automatically.
os.environ["PINECONE_API_KEY"] = pinecone_key
os.environ["GOOGLE_API_KEY"] = google_key 

INDEX_NAME = "gurufast"

app = FastAPI(title="Advaita Guru API")

# 2. Setup Vector Store
print("🔌 Connecting to Pinecone...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_key
)

# FIXED: Removed 'pinecone_api_key' argument. 
# It will now auto-detect the key we set in os.environ above.
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Setup LLM
llm = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest", 
    temperature=0.3,
    google_api_key=google_key
) 

# 4. The Persona Prompt
guru_prompt = ChatPromptTemplate.from_template("""
You are a wise and compassionate teacher of Advaita Vedanta. 
Your goal is to guide the student from ignorance (avidya) to self-knowledge (atma-jnana).

Rules:
1. Answer strictly based on the provided CONTEXT. 
2. Be calm, profound, and clear. Use analogies (rope/snake, pot/clay).
3. If the answer is not in the context, admit it gently, then provide a general Vedantic view.
4. Use Sanskrit terms (Brahman, Atman) but define them.

CONTEXT:
{context}

QUESTION: 
{question}

YOUR ANSWER:
""")

# 5. The Chain
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | guru_prompt
    | llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: QueryRequest):
    print(f"🤔 Received Question: {request.query}")
    try:
        response = rag_chain.invoke(request.query)
        print("✅ Answer Generated.")
        return {"answer": response}
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "GuruBot is awake. Go to /docs to test."}