# Quick Start Guide - ASAG System

Get your Automatic Short Answer Grading system running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
- A PDF textbook or course material

## Step 1: Install Dependencies

```powershell
pip install -r requirements.txt
```

**Note**: Installation may take 5-10 minutes due to PyTorch and other ML libraries.

## Step 2: Set Your OpenAI API Key

Choose one method:

### Method A: Environment Variable (Recommended for testing)

**PowerShell:**
```powershell
$env:OPENAI_API_KEY="sk-your-api-key-here"
```

**CMD:**
```cmd
set OPENAI_API_KEY=sk-your-api-key-here
```

### Method B: .env File (Recommended for development)

1. Copy the example file:
   ```powershell
   copy .env.example .env
   ```

2. Edit `.env` and add your key:
   ```
   OPENAI_API_KEY=sk-your-api-key-here
   ```

3. Install python-dotenv if not already installed:
   ```powershell
   pip install python-dotenv
   ```

4. Update `main.py` to load .env (add at top):
   ```python
   from dotenv import load_dotenv
   load_dotenv()
   ```

## Step 3: Start the Server

```powershell
python main.py
```

You should see:
```
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 4: Test the API

### Option A: Use the Web Interface

Open your browser and go to:
```
http://localhost:8000/docs
```

This interactive interface lets you test all endpoints directly!

### Option B: Use the Test Script

In a new terminal:

```powershell
# Test without uploading
python test_api.py

# Test with a PDF upload
python test_api.py "path\to\your\textbook.pdf"
```

### Option C: Use cURL

**Check health:**
```powershell
curl http://localhost:8000/health
```

**Upload a PDF:**
```powershell
curl -X POST "http://localhost:8000/upload" -F "file=@textbook.pdf"
```

**Grade an answer:**
```powershell
curl -X POST "http://localhost:8000/grade" `
  -H "Content-Type: application/json" `
  -d '{\"question\":\"What is a process?\",\"student_answer\":\"A program in execution\"}'
```

## Step 5: Your First Grading Request

### Using Python:

```python
import requests

# 1. Upload your course material
with open("textbook.pdf", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/upload", files=files)
    print(response.json())

# 2. Grade a student answer
payload = {
    "question": "What is a process in an operating system?",
    "student_answer": "A process is a program that is currently executing."
}
response = requests.post("http://localhost:8000/grade", json=payload)
result = response.json()

print(f"Score: {result['score']}/5")
print(f"Feedback: {result['feedback']}")
```

### Using JavaScript (fetch):

```javascript
// 1. Upload PDF
const formData = new FormData();
formData.append('file', pdfFile);

fetch('http://localhost:8000/upload', {
  method: 'POST',
  body: formData
})
.then(res => res.json())
.then(data => console.log(data));

// 2. Grade answer
fetch('http://localhost:8000/grade', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    question: "What is a process in an operating system?",
    student_answer: "A process is a program that is currently executing."
  })
})
.then(res => res.json())
.then(data => {
  console.log('Score:', data.score);
  console.log('Feedback:', data.feedback);
});
```

## Troubleshooting

### ❌ "Module not found" errors

```powershell
pip install --upgrade -r requirements.txt
```

### ❌ "OPENAI_API_KEY not set"

Make sure you set the environment variable before starting the server:

```powershell
$env:OPENAI_API_KEY="your-key"
python main.py
```

### ❌ "Knowledge base not found"

You must upload at least one PDF before grading:

```powershell
curl -X POST "http://localhost:8000/upload" -F "file=@textbook.pdf"
```

### ❌ Server won't start (port 8000 in use)

Change the port:

```powershell
uvicorn main:app --port 8080
```

### ❌ LangChain import errors

If you get import errors like `ImportError: cannot import name 'PyPDFLoader'`, update the imports in `main.py`:

```python
# Change from:
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings

# To:
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
```

## What's Next?

- Read the full [README_API.md](README_API.md) for detailed documentation
- Customize the grading prompt in `main.py` (line 39)
- Adjust chunk size and overlap for better retrieval (lines 23-24)
- Add authentication/authorization for production use
- Deploy to cloud (AWS, Azure, GCP)

## System Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Upload    │────▶│  Chunk Text  │────▶│   Embed &   │
│   PDF       │     │  (1000/150)  │     │ Store FAISS │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Return    │◀────│  Parse Score │◀────│  Retrieval  │
│ Score +     │     │  & Feedback  │     │  + LLM      │
│  Feedback   │     └──────────────┘     │  (GPT-4)    │
└─────────────┘                          └─────────────┘
                                                 ▲
                                                 │
                                         ┌─────────────┐
                                         │   Grade     │
                                         │  Request    │
                                         └─────────────┘
```

## API Endpoints Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root endpoint with API info |
| `/health` | GET | Health check and status |
| `/upload` | POST | Upload PDF to knowledge base |
| `/grade` | POST | Grade student answer |
| `/docs` | GET | Interactive API documentation |

## Need Help?

- Check the [README_API.md](README_API.md) for detailed docs
- Review error messages in the terminal
- Test with the provided `test_api.py` script
- Verify your OpenAI API key is valid and has credits

Happy Grading! 🎓
