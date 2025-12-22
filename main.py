import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

# Optional: load environment variables from .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# LangChain imports (with compatibility fallbacks)
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_community.vectorstores import FAISS
except Exception:  # pragma: no cover
    from langchain.document_loaders import PyPDFLoader  # type: ignore
    from langchain.vectorstores import FAISS  # type: ignore

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:  # pragma: no cover
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # type: ignore

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except Exception:  # pragma: no cover
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

try:
    # Provided by langchain-community (what you already installed)
    from langchain_community.chat_models import ChatOllama
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "Missing ChatOllama. Install `langchain-community` and ensure your LangChain packages are installed."
    ) from e


# -----------------------------
# Storage / configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "faiss_index"

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Local LLM via Ollama (no API key required)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b-instruct")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# Depending on LangChain/FAISS versions, load_local may require this.
ALLOW_DANGEROUS_DESERIALIZATION = os.getenv("ALLOW_DANGEROUS_DESERIALIZATION", "true").lower() in (
    "1",
    "true",
    "yes",
)


# -----------------------------
# Prompt (EXACT structure requested)
# -----------------------------
GRADING_PROMPT_TEMPLATE = (
    "You are an expert Teaching Assistant for a university-level Operating Systems course. "
    "Your task is to grade a student's answer based only on the provided textbook knowledge.\n"
    "Relevant Knowledge from Course Materials: \"\"\" {context} \"\"\"\n\n"
    "Question: \"\"\" {question} \"\"\"\n\n"
    "Student's Answer: \"\"\" {student_answer} \"\"\"\n\n"
    "Instruction: Based on the provided knowledge, evaluate the student's answer on a scale from 0 to 5. "
    "The score must be an integer. First, provide your reasoning for the score, explaining how the student's answer "
    "aligns with or deviates from the course materials. Then, on a new line, provide the final score in the format: "
    "\"Final Score: [score]\".\n"
)


# -----------------------------
# API models
# -----------------------------
class GradeRequest(BaseModel):
    question: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)


class GradeResponse(BaseModel):
    score: int = Field(..., ge=0, le=5)
    feedback: str


# -----------------------------
# Helpers
# -----------------------------

def _get_embeddings() -> HuggingFaceEmbeddings:
    # Uses SentenceTransformers under the hood.
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def _parse_llm_output(text: str) -> Tuple[str, int]:
    """Parse the LLM output to (feedback, score)."""
    # Accept: Final Score: [3] OR Final Score: 3
    m = re.search(r"Final\s*Score\s*:\s*\[?\s*(\d)\s*\]?", text, flags=re.IGNORECASE)
    if not m:
        raise ValueError("Could not extract 'Final Score: [score]' from model output.")

    score = int(m.group(1))
    score = max(0, min(5, score))

    feedback = text[: m.start()].strip()
    if not feedback:
        feedback = re.sub(
            r"Final\s*Score\s*:\s*\[?\s*\d\s*\]?", "", text, flags=re.IGNORECASE
        ).strip()

    return feedback, score


def _load_vectorstore() -> FAISS:
    if not VECTORSTORE_DIR.exists():
        raise FileNotFoundError(
            f"FAISS vector store not found at {VECTORSTORE_DIR}. Upload materials first via /upload."
        )

    embeddings = _get_embeddings()

    try:
        return FAISS.load_local(
            str(VECTORSTORE_DIR),
            embeddings,
            allow_dangerous_deserialization=ALLOW_DANGEROUS_DESERIALIZATION,
        )
    except TypeError:
        # Older LangChain
        return FAISS.load_local(str(VECTORSTORE_DIR), embeddings)


def _build_chain(vectordb: FAISS):
    """Build an LCEL RAG grading chain using local Ollama (no paid API)."""
    llm = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
    )

    prompt = ChatPromptTemplate.from_template(GRADING_PROMPT_TEMPLATE)

    retriever = vectordb.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    def _format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    get_question = RunnableLambda(lambda x: x["question"])
    get_student_answer = RunnableLambda(lambda x: x["student_answer"])

    chain = (
        {
            "context": get_question | retriever | _format_docs,
            "question": get_question,
            "student_answer": get_student_answer,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Automatic Short Answer Grading (RAG)")


@app.post("/upload")
async def upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Ingest a PDF -> chunk -> embed -> save a local FAISS vector store."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(file.filename).name)
    saved_path = UPLOAD_DIR / safe_name

    # Save upload
    try:
        with saved_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    finally:
        try:
            file.file.close()
        except Exception:
            pass

    try:
        # 1) Load PDF text
        loader = PyPDFLoader(str(saved_path))
        docs = loader.load()

        # 2) Chunk text
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("No chunks produced from the uploaded PDF.")

        # 3) Embed
        embeddings = _get_embeddings()

        # 4) Create and save local FAISS vector store
        vectordb = FAISS.from_documents(chunks, embeddings)
        if VECTORSTORE_DIR.exists():
            shutil.rmtree(VECTORSTORE_DIR)
        vectordb.save_local(str(VECTORSTORE_DIR))

        return {
            "message": "Knowledge base ingested successfully.",
            "filename": safe_name,
            "pages_loaded": len(docs),
            "chunks_created": len(chunks),
            "vectorstore_path": str(VECTORSTORE_DIR),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest PDF: {e}")


@app.post("/grade", response_model=GradeResponse)
async def grade(payload: GradeRequest) -> GradeResponse:
    """Grade a student's answer using RAG over the ingested knowledge base."""
    try:
        vectordb = _load_vectorstore()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        chain = _build_chain(vectordb)

        raw = chain.invoke({"question": payload.question, "student_answer": payload.student_answer})
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Unexpected chain output format.")

        feedback, score = _parse_llm_output(raw)
        return GradeResponse(score=score, feedback=feedback)

    except ValueError as e:
        raise HTTPException(status_code=500, detail=f"Grading failed (parsing): {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grading failed: {e}")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "vectorstore_present": VECTORSTORE_DIR.exists(),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "llm_provider": "ollama",
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
    }
