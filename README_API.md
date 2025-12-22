# Automatic Short Answer Grading System (ASAG) with RAG

A complete Python-based system for automatically grading short answers using Retrieval-Augmented Generation (RAG) with LangChain, FastAPI, and GPT-4.

## Features

- **Knowledge Base Ingestion**: Upload PDF textbooks/course materials
- **RAG-based Grading**: Uses retrieval-augmented generation for context-aware grading
- **Structured Feedback**: Returns both numerical scores (0-5) and detailed reasoning
- **RESTful API**: Easy integration with frontend applications

## Architecture

- **Backend Framework**: FastAPI
- **RAG Pipeline**: LangChain
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS
- **LLM**: OpenAI GPT-4

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set up OpenAI API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-openai-api-key-here
```

Or set it as an environment variable:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="your-openai-api-key-here"
```

**Windows (CMD):**
```cmd
set OPENAI_API_KEY=your-openai-api-key-here
```

**Linux/Mac:**
```bash
export OPENAI_API_KEY=your-openai-api-key-here
```

### 3. Run the Application

```bash
python main.py
```

Or using uvicorn directly:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at: `http://localhost:8000`

## API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative Docs**: http://localhost:8000/redoc

## API Endpoints

### 1. Upload Knowledge Base

**Endpoint**: `POST /upload`

Upload a PDF file containing course materials (textbooks, lecture notes, etc.)

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@textbook.pdf"
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/upload"
files = {"file": open("textbook.pdf", "rb")}
response = requests.post(url, files=files)
print(response.json())
```

**Response:**
```json
{
  "message": "Knowledge base updated successfully",
  "file_name": "textbook.pdf",
  "chunks_processed": 245,
  "total_pages": 42
}
```

### 2. Grade Student Answer

**Endpoint**: `POST /grade`

Submit a question and student answer for grading.

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/grade" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a process in an operating system?",
    "student_answer": "A process is a program in execution with its own memory space."
  }'
```

**Python Example:**
```python
import requests

url = "http://localhost:8000/grade"
payload = {
    "question": "What is a process in an operating system?",
    "student_answer": "A process is a program in execution with its own memory space."
}
response = requests.post(url, json=payload)
print(response.json())
```

**Response:**
```json
{
  "score": 4,
  "feedback": "The student correctly identifies that a process is a program in execution, which is the fundamental definition. The mention of memory space shows understanding of process isolation. However, the answer lacks detail about other key aspects such as the process control block (PCB), program counter, stack, and heap. A more complete answer would mention that a process includes not just memory but also CPU state, file descriptors, and other resources."
}
```

### 3. Health Check

**Endpoint**: `GET /health`

Check system status and configuration.

**Response:**
```json
{
  "status": "healthy",
  "knowledge_base_loaded": true,
  "openai_configured": true
}
```

## Configuration

Edit these constants in `main.py` to customize behavior:

```python
VECTOR_STORE_PATH = "./faiss_index"  # Where to store the vector database
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # SentenceTransformer model
CHUNK_SIZE = 1000  # Text chunk size for embeddings
CHUNK_OVERLAP = 150  # Overlap between chunks
```

## Grading Prompt

The system uses a structured prompt that:
1. Provides relevant context from course materials
2. Shows the question and student answer
3. Requests reasoning-based grading on a 0-5 scale

The LLM is instructed to act as a "Teaching Assistant for a university-level Operating Systems course" and grade based only on provided knowledge.

## Workflow

1. **Upload Course Materials**: Use `/upload` to add PDF textbooks or lecture notes
2. **System Processes PDFs**: Extracts text, chunks it, generates embeddings, stores in FAISS
3. **Submit Grading Request**: Use `/grade` with question and student answer
4. **RAG Retrieval**: System retrieves top-5 most relevant knowledge chunks
5. **LLM Grading**: GPT-4 evaluates answer against retrieved context
6. **Parse & Return**: Extract score and feedback, return structured JSON

## Production Deployment

### Environment Variables

Always use environment variables for sensitive data:

```python
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
```

### Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t asag-system .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key asag-system
```

### Security Considerations

- Never commit `.env` files or API keys to version control
- Use HTTPS in production
- Implement authentication/authorization for API endpoints
- Rate limit API requests
- Validate and sanitize all inputs
- Use secrets management (AWS Secrets Manager, Azure Key Vault, etc.)

## Troubleshooting

### Issue: "OPENAI_API_KEY environment variable not set"

**Solution**: Set your OpenAI API key as described in the setup section.

### Issue: "Knowledge base not found"

**Solution**: Upload at least one PDF using the `/upload` endpoint before grading.

### Issue: Import errors for langchain modules

**Solution**: Update LangChain imports. If using newer versions, change:
```python
# Old
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# New (LangChain >= 0.1.0)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
```

### Issue: FAISS deserialization warning

This is expected. The system uses `allow_dangerous_deserialization=True` for loading local FAISS indexes. Only disable this if loading untrusted index files.

## Future Enhancements

- [ ] Support multiple file formats (TXT, DOCX, etc.)
- [ ] Batch grading endpoint
- [ ] Rubric-based grading
- [ ] Student answer similarity detection
- [ ] Grading history and analytics
- [ ] Fine-tuned models for specific subjects
- [ ] Multi-language support
- [ ] Confidence scores
- [ ] Interactive feedback refinement

## License

MIT

## Contact

For questions or issues, please open a GitHub issue or contact the development team.
