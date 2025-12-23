# Autonomous Document Intelligence Platform

An AI-powered Intelligent Document Processing (IDP) system that transforms unstructured documents into structured, analysis-ready datasets through natural language interaction, enabling dynamic data curation and on-demand dataset generation from vast document collections.

Leveraging agentic AI frameworks and multimodal LLMs, the platform interprets natural language queries to autonomously extract, synthesize, and structure information across documents while ensuring privacy through intelligent deidentification.

---

## Project Overview

### The Problem
Enterprises possess vast repositories of unstructured and semi-structured documents holding immense potential value, but transforming this raw information into structured, analysis-ready datasets remains a major bottleneck. Traditional IDP systems are often too rigid, focusing on predefined extraction tasks for individual documents rather than synthesizing information across large corpora to build tailored datasets. 

The core challenge is creating an intelligent system that not only processes documents but allows users to dynamically define, refine, and generate specific datasets from vast document collections using simple natural language commands, moving beyond static processing to interactive, on-demand data curation.

### The Solution
An Intelligent Document Processing (IDP) solution powered by Agentic AI, designed to autonomously create structured datasets guided by user interaction via natural language. The system implements:

- **Natural Language Driven Curation**: Agentic AI framework interpreting user requests to define dataset goals, scope, schema, and filtering criteria
- **Deep Understanding**: Advanced multimodal and reasoning capabilities to understand content relationships within and across documents
- **Query-Adaptive Extraction**: Dynamic extraction and synthesis strategies based on natural language queries
- **On-Demand Assembly**: Automated dataset assembly with quality control and exception handling
- **Dynamic Schema Adaptation**: Conversational interface for clarifying ambiguities and iterative refinement
- **Continuous Learning**: Feedback mechanisms for improving query interpretation and dataset quality
- **Responsible AI**: Privacy-preserving deidentification guided by user specifications

---

## Features

### Intelligent Document Processing
- **Multi-format Support**: PDF, images (PNG/JPG/JPEG), text files, markdown, CSV, and JSON
- **Advanced OCR**: Pytesseract integration for extracting text from scanned documents
- **PDF Processing**: PyMuPDF-based extraction for digital PDF documents

### Natural Language Interaction
- **Conversational AI**: Chat-based interface for querying documents
- **Dynamic Query Processing**: LangChain-powered agent system for understanding user intent
- **Context-Aware Responses**: Maintains conversation context across multiple queries

### Security & Privacy
- **Deidentification**: Presidio-based PII detection and anonymization
- **File Validation**: MIME type checking and malware scanning with ClamAV
- **Rate Limiting**: SlowAPI integration to prevent abuse
- **Authentication**: JWT-based OAuth2 authentication
- **Secure Storage**: PostgreSQL with Prisma ORM for data persistence

### Dataset Generation
- **Structured Output**: Converts unstructured documents into structured datasets
- **Quality Control**: Automated validation and quality issue detection
- **Multi-document Synthesis**: Aggregates information across multiple documents
- **Export Ready**: Generates analysis-ready datasets

### Modern Web Interface
- **Next.js 15**: React-based frontend with server-side rendering
- **Real-time Chat**: Interactive document querying interface
- **File Upload**: Drag-and-drop document upload with preview
- **Responsive Design**: Tailwind CSS-powered modern UI

---

## Architecture

### Backend (FastAPI + Python)
```
backend/
├── main.py                 # FastAPI application & API endpoints
├── agent_manager.py        # LangChain agent orchestration
├── document_processor.py   # Multi-format document extraction
├── deidentification.py     # PII detection & anonymization
└── security_service.py     # Authentication & validation
```

### Frontend (Next.js + TypeScript)
```
src/
├── app/
│   ├── page.tsx           # Landing page
│   ├── chat/              # Chat interface
│   └── api/               # API routes (upload, register)
├── lib/
│   ├── encryption.ts      # Client-side encryption utilities
│   ├── prisma.ts          # Database client
│   └── security.ts        # Security middleware
└── types/                 # TypeScript type definitions
```

---

## Getting Started

### Prerequisites
- **Node.js** 18+ and npm/yarn
- **Python** 3.8+
- **PostgreSQL** database
- **Tesseract OCR** (for image processing)
- **ClamAV** (optional, for malware scanning)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/BhoomiAgrawal12/IDP_Agents.git
cd IDP_Agents
```

#### 2. Backend Setup
```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
python setup_db.py

# Run the backend server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### 3. Frontend Setup
```bash
# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your configuration

# Set up Prisma
npx prisma generate
npx prisma db push

# Run development server
npm run dev
```

### Environment Variables

#### Backend (.env)
```env
DB_PATH=processed_docs.db
HUGGINGFACE_API_KEY=your_key_here
DATABASE_URL=postgresql://user:password@localhost:5432/idp_db
```

#### Frontend (.env.local)
```env
DATABASE_URL=postgresql://user:password@localhost:5432/idp_db
NEXT_PUBLIC_API_URL=http://localhost:8000
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=your_bucket_name
```

---

## Usage

### 1. Start the Application
```bash
# Terminal 1 - Backend
cd backend
uvicorn main:app --reload

# Terminal 2 - Frontend
npm run dev
```

### 2. Access the Platform
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Process Documents
1. Navigate to the chat interface
2. Upload documents (PDF, images, or text files)
3. Ask questions in natural language
4. Receive structured datasets and insights

### Example Queries
- "Extract all customer names and email addresses from these invoices"
- "Create a dataset of product information from these catalogs"
- "Summarize key findings from these research papers"
- "Build a table of dates, amounts, and vendors from these receipts"

---

## Technology Stack

### AI & ML
- **LangChain**: Agent orchestration and LLM integration
- **Transformers**: Hugging Face models for NLP
- **Presidio**: PII detection and anonymization

### Backend
- **FastAPI**: High-performance API framework
- **PyMuPDF**: PDF text extraction
- **Pytesseract**: OCR for images
- **SQLite/PostgreSQL**: Data persistence

### Frontend
- **Next.js 15**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Modern styling
- **Prisma**: Type-safe ORM
- **AWS S3**: File storage

### Security
- **JWT**: Authentication tokens
- **bcrypt**: Password hashing
- **ClamAV**: Malware scanning
- **python-magic**: File type validation
- **SlowAPI**: Rate limiting

---

## API Endpoints

### Document Processing
```http
POST /process
Content-Type: multipart/form-data

Parameters:
- files: List of document files
- query: Natural language query
```

### Deidentification
```http
POST /api/deidentify
Content-Type: application/json

Body:
{
  "content": "Text containing PII",
  "query": "Processing instructions"
}
```

---

## Security Features

- **File Validation**: MIME type and extension verification
- **Malware Scanning**: ClamAV integration for uploaded files
- **Rate Limiting**: 5 requests per minute per IP
- **Authentication**: OAuth2 with JWT tokens
- **PII Protection**: Automatic detection and anonymization
- **CORS Protection**: Configured cross-origin policies
- **Input Sanitization**: HTML sanitization for user inputs

---

## Testing

```bash
# Backend tests
cd backend
python test_server.py

# Frontend tests
npm run test

# E2E tests
npm run test:e2e
```

---

## Abbreviations

- **IDP**: Intelligent Document Processing
- **OCR**: Optical Character Recognition
- **NLP**: Natural Language Processing
- **NLU**: Natural Language Understanding
- **LLM**: Large Language Model
- **PII**: Personally Identifiable Information
- **API**: Application Programming Interface
- **JWT**: JSON Web Token

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License
This project is under the MIT License.

---
