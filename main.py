import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <--- NEW IMPORT
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
    raise ValueError("❌ CRITICAL ERROR: API Keys not found. Check your .env file or Render Environment Variables.")

# --- CRITICAL CONFIG: FORCE ENVIRONMENT VARIABLES ---
# The Pinecone library looks for this specific variable name automatically.
os.environ["PINECONE_API_KEY"] = pinecone_key
os.environ["GOOGLE_API_KEY"] = google_key 

INDEX_NAME = "gurufast"

app = FastAPI(title="Advaita Guru API")

# --- CORS SETUP (Allow the website to talk to the brain) ---
# This block is VITAL. Without it, your HTML website cannot connect to this server.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows any website to connect (Safe for public apps)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],
)
# -----------------------------------------------------------

# 2. Setup Vector Store
print("🔌 Connecting to Pinecone...")
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=google_key
)

vectorstore = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME, 
    embedding=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 3. Setup LLM
# We use "gemini-flash-latest" as it is the stable production alias 
# that has the active Free Tier quota.
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
5. Format your answer nicely with paragraphs or bullet points if needed.

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