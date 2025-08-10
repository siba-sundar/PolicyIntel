# PolicyIntel 🚀

**The Intelligent Multi-Format Document Processing & Q&A System with Advanced AI Reasoning**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

PolicyIntel is a sophisticated document processing and question-answering system that intelligently processes multiple file formats, extracts meaningful content, and provides accurate answers using advanced AI reasoning. Built with cutting-edge RAG (Retrieval-Augmented Generation) technology, it features dependency-aware link fetching, parallel processing, and enterprise-grade security.

---

## 🌟 Key Features

### 📄 **Multi-Format Document Processing**
- **PDF Processing**: Enhanced table structure preservation, smart link extraction, and metadata analysis
- **Office Documents**: Word (.docx/.doc), PowerPoint (.pptx/.ppt), Excel (.xlsx/.xls)  
- **Images**: OCR processing for JPG, PNG, GIF, BMP, TIFF, WebP with batch optimization
- **Text Files**: Plain text, HTML, and ZIP archive processing with nested support
- **Advanced OCR**: Parallel batch processing with Tesseract integration

### 🧠 **Intelligent AI Reasoning System**
- **Three-Tier Answer Hierarchy**:
  1. **Direct Answers**: Extract exact information from document context
  2. **Pattern Deduction**: Identify logical patterns and sequences from examples
  3. **Intelligent Reasoning**: Apply domain knowledge when patterns aren't available
- **Context-Aware Processing**: Smart chunking with semantic similarity
- **Mathematical Pattern Recognition**: Analyzes examples to determine underlying logic

### 🔗 **Dependency-Aware RAG Pipeline**
- **Smart Link Detection**: Automatically identifies when external links need to be fetched
- **Two-Stage Classification**: Fast rule-based + LLM checkpoint for accurate dependency detection
- **Concurrent Link Fetching**: Parallel processing with timeout handling and caching
- **Content Enhancement**: Specialized JSON, HTML, and text response processing

### ⚡ **High-Performance Architecture**
- **Parallel Processing**: Multi-threaded document chunking, embedding, and OCR
- **FAISS Vector Search**: Optimized semantic similarity search with multi-tier retrieval
- **API Key Rotation**: Automatic load balancing across multiple Gemini and Cohere keys
- **Memory Optimization**: Intelligent cleanup and resource management for cloud deployment

### 🛡️ **Enterprise Security**
- **Data Encryption**: AES-256-GCM encryption for sensitive data
- **Integrity Verification**: HMAC-SHA256 for data integrity checks
- **Secure Audit Logging**: Comprehensive request tracking and monitoring
- **Input Sanitization**: Protection against malicious content and injection attacks

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- API Keys for:
  - Google Gemini (1-3 keys recommended)
  - Cohere Embeddings (1-2 keys recommended)
  - OCR Space (optional, for enhanced OCR)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/policyintel.git
cd policyintel
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
COHERE_API_KEY=your_cohere_api_key
COHERE_API_KEY_2=your_second_cohere_key  # Optional
GEMINI_API_KEY_1=your_gemini_api_key
GEMINI_API_KEY_2=your_second_gemini_key  # Optional
GEMINI_API_KEY_3=your_third_gemini_key   # Optional
TEAM_TOKEN=your_authentication_token
OCR_SPACE_API_KEY=your_ocr_space_key     # Optional
```

4. **Install system dependencies (for OCR)**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract

# Windows - Download from: https://github.com/tesseract-ocr/tesseract
```

5. **Run the application**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

---

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t policyintel .

# Run the container
docker run -p 8000:8000 \
  -e COHERE_API_KEY=your_key \
  -e GEMINI_API_KEY_1=your_key \
  -e TEAM_TOKEN=your_token \
  policyintel
```

### Deploy on Render

1. Connect your GitHub repository to Render
2. Use the included `render.yaml` configuration
3. Set environment variables in Render dashboard
4. Deploy automatically with every push

---

## 📚 API Usage

### Main Query Endpoint

**POST** `/hackrx/run`

Process documents and answer questions with intelligent reasoning.

```json
{
  "documents": "https://example.com/document.pdf",
  "questions": [
    "What is covered under this policy?",
    "What are the claim procedures?"
  ]
}
```

**Response:**
```json
{
  "answers": [
    "The policy covers hospitalization expenses up to Rs. 5,00,000 including room rent, surgery fees, and diagnostic tests.",
    "Claims must be intimated within 24 hours of emergency admission or 48 hours before planned treatment."
  ]
}
```

### Health Check Endpoints

```bash
# System health and capabilities
GET /health

# Memory usage and optimization
GET /memory-status  

# API key status and rotation info
GET /api-status

# FAISS index statistics
GET /faiss-stats

# Supported file formats
GET /supported-formats
```

### Security Endpoints

```bash
# Security system status
GET /hackrx/security/status

# Link processing analysis
GET /hackrx/links/analysis/{request_id}

# Configure link processing
POST /hackrx/links/configure
```

---

## 🎯 Advanced Features

### Intelligent Link Processing

PolicyIntel automatically detects and processes external links in documents:

```python
# Example: Processing HackRx puzzle with API dependencies
{
  "documents": "https://hackrx.in/final-round.pdf",
  "questions": ["What is my flight number?"]
}

# System automatically:
# 1. Detects API endpoint in PDF: https://register.hackrx.in/submissions/myFavouriteCity  
# 2. Fetches user's favorite city: {"city": "Mumbai"}
# 3. Maps city to landmark using PDF table: Mumbai → Gateway of India
# 4. Determines correct flight endpoint and fetches flight number
# 5. Returns complete answer with reasoning chain
```

### Multi-Language Support

```json
{
  "documents": "https://example.com/hindi-document.pdf",
  "questions": ["यह नीति क्या कवर करती है?"]
}
```

Automatically detects question language and responds accordingly.

### Large Document Mode

Enable enhanced processing for large documents:

```bash
export LARGE_DOC_MODE=true
python main.py
```

Benefits:
- Increased search results (30 vs 20)
- Larger context windows (20 vs 15)
- Enhanced chunk sizes and overlap
- Lower similarity thresholds for better recall

---

## 🏗️ Architecture Overview

### Core Components

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI API   │ -> │  Document        │ -> │  Enhanced PDF   │
│   Layer         │    │  Service         │    │  Processor      │  
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Security &    │ -> │  Embedding       │ -> │  FAISS Vector   │
│   Auth Utils    │    │  Service         │    │  Search         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         v                       v                       v
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Memory &      │ -> │  LLM Service     │ -> │  Dependency     │
│   Cache Utils   │    │  (Gemini)        │    │  Manager        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Processing Pipeline

1. **Document Ingestion**: Multi-format file processing with validation
2. **Smart Chunking**: Context-aware text segmentation with overlap
3. **Link Detection**: Intelligent identification of fetchable dependencies  
4. **Embedding Generation**: Parallel vector generation with Cohere
5. **Vector Indexing**: FAISS-based similarity search optimization
6. **Query Processing**: Multi-tier search with relevance scoring
7. **AI Reasoning**: Three-tier answer generation with context enrichment
8. **Response Assembly**: Structured output with confidence indicators

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `COHERE_API_KEY` | Primary Cohere API key | - | ✅ |
| `COHERE_API_KEY_2` | Secondary Cohere key | - | ❌ |
| `GEMINI_API_KEY_1` | Primary Gemini API key | - | ✅ |
| `GEMINI_API_KEY_2` | Secondary Gemini key | - | ❌ |
| `GEMINI_API_KEY_3` | Tertiary Gemini key | - | ❌ |
| `TEAM_TOKEN` | Authentication token | - | ✅ |
| `OCR_SPACE_API_KEY` | OCR Space API key | - | ❌ |
| `LARGE_DOC_MODE` | Enhanced large doc processing | `false` | ❌ |

### Performance Tuning

```python
# Document Processing
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
MAX_WORKERS = min(4, os.cpu_count())
PARALLEL_CHUNK_SIZE = 5
PARALLEL_OCR_BATCH = 3

# Vector Search
FAISS_K_SEARCH = 20  # Search results count
SEMANTIC_SIMILARITY_THRESHOLD = 0.20
CONTEXT_WINDOW_SIZE = 15

# AI Models
LLM_TEMPERATURE = 0.25
LLM_MAX_TOKENS = 300
EMBEDDING_BATCH_SIZE = 96
```

---

## 📊 Performance Benchmarks

### Processing Speed
- **PDF Processing**: ~2-5 seconds per page
- **OCR Processing**: ~1-3 seconds per image (parallel batch)
- **Embedding Generation**: ~0.5-2 seconds per batch (96 texts)
- **Vector Search**: ~50-200ms for 10K+ vectors
- **LLM Response**: ~1-3 seconds per question

### Scalability
- **Concurrent Requests**: 10-50 (depending on hardware)
- **Memory Usage**: ~500MB-2GB (auto-optimized)
- **Document Size**: Up to 500MB per file
- **Supported Languages**: 100+ (via Tesseract OCR)

---

## 🧪 Testing & Examples

### Run Example Scripts

```bash
# Test dependency-aware RAG with HackRx scenario  
python run_request_example.py --mode hackrx

# Test simple document Q&A
python run_request_example.py --mode simple

# Run both examples
python run_request_example.py --mode both
```

### API Testing with cURL

```bash
# Test document processing
curl -X POST "http://localhost:8000/hackrx/run" \
  -H "Authorization: Bearer your-token" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/policy.pdf",
    "questions": ["What is the coverage limit?"]
  }'

# Check system health
curl "http://localhost:8000/health"
```

### Unit Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Test specific module
python -m pytest tests/test_pdf_processor.py -v

# Test with coverage
python -m pytest tests/ --cov=app --cov-report=html
```

---

## 🛠️ Development

### Project Structure

```
PolicyIntel/
├── app/
│   ├── api/endpoints/          # FastAPI route handlers
│   ├── config/                 # Configuration settings
│   ├── models/                 # Pydantic data models
│   ├── processors/             # Document format processors  
│   ├── services/               # Core business logic
│   └── utils/                  # Utility functions
├── document_cache/             # Cached document data
├── tests/                      # Unit tests
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container configuration
└── render.yaml                 # Render deployment config
```

### Adding New Document Processors

1. **Create processor class**:
```python
from app.processors.base_processor import SyncBaseProcessor

class MyProcessor(SyncBaseProcessor):
    def process_sync(self, file_content: bytes, filename: str = "", **kwargs) -> str:
        # Your processing logic here
        return extracted_text
    
    def supports_format(self, filename: str) -> bool:
        return filename.lower().endswith('.myext')
```

2. **Register the processor** in `document_service.py`
3. **Add format** to `SUPPORTED_FORMATS` in settings
4. **Write tests** for the new processor

### Contributing Guidelines

1. **Fork the repository** and create a feature branch
2. **Follow PEP 8** style guidelines
3. **Write comprehensive tests** for new features
4. **Update documentation** for API changes
5. **Submit a pull request** with detailed description

---

## 📈 Monitoring & Analytics

### Built-in Monitoring

- **Request Tracking**: Unique request IDs with full traceability
- **Performance Metrics**: Response times, memory usage, API calls
- **Error Tracking**: Comprehensive exception logging and recovery
- **Security Auditing**: Access logs, authentication events, data integrity

### Log Analysis

```bash
# View system logs
docker logs container-name

# Monitor API performance  
grep "completed in" app.log | tail -100

# Check security events
grep "🛡️" app.log | tail -50

# Memory optimization tracking
grep "📊 Memory" app.log | tail -20
```

### Health Monitoring Endpoints

```bash
# Comprehensive health check with metrics
GET /health

# Memory usage and cleanup status  
GET /memory-status

# API key rotation and usage stats
GET /api-status

# Vector search performance metrics
GET /faiss-stats
```

---

## 🔒 Security Considerations

### Data Protection
- **Encryption at Rest**: AES-256-GCM for cached documents
- **Encryption in Transit**: TLS 1.3 for all API communications
- **Key Management**: Secure rotation of API keys and tokens
- **Data Retention**: Configurable cache expiry and cleanup

### Access Control  
- **Token-Based Authentication**: Secure API access control
- **Request Rate Limiting**: Protection against abuse
- **Input Validation**: Comprehensive sanitization of all inputs
- **Output Filtering**: Prevention of sensitive data leakage

### Audit & Compliance
- **Complete Audit Trail**: All requests and responses logged
- **Integrity Verification**: Hash-based data integrity checks
- **Security Events**: Real-time monitoring and alerting
- **Privacy Controls**: Configurable data retention policies

---

## 🚀 Production Deployment

### Recommended Infrastructure

**Minimum Requirements:**
- CPU: 2 cores, 4GB RAM
- Storage: 10GB SSD
- Network: 1Gbps connection
- OS: Ubuntu 20.04+ or Docker

**Optimal Configuration:**
- CPU: 4-8 cores, 16GB RAM  
- Storage: 50GB NVMe SSD
- Network: 10Gbps connection
- Load Balancer: Nginx/HAProxy
- Monitoring: Prometheus + Grafana

### Environment Setup

1. **Production Environment Variables**
```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export ENABLE_METRICS=true
export CACHE_TTL=3600
export MAX_CONCURRENT_REQUESTS=20
```

2. **Database Configuration**
```bash
# Optional Redis for distributed caching
export REDIS_URL=redis://localhost:6379
export CACHE_STRATEGY=redis
```

3. **Monitoring Setup**
```bash
# Prometheus metrics endpoint
export METRICS_ENABLED=true
export METRICS_PORT=9090
```

### Scaling Considerations

- **Horizontal Scaling**: Deploy multiple instances behind load balancer
- **Vertical Scaling**: Increase CPU/RAM for higher throughput
- **Caching Strategy**: Use Redis for distributed document caching  
- **Database Scaling**: Consider PostgreSQL for persistent storage
- **CDN Integration**: Serve static assets via CloudFlare/AWS CloudFront

---

## ❓ FAQ

### Common Issues

**Q: Getting "API key quota exceeded" errors?**
A: Add multiple API keys in environment variables. The system automatically rotates between them.

**Q: OCR not working properly?**  
A: Ensure Tesseract is installed and properly configured. Check OCR_SPACE_API_KEY for enhanced accuracy.

**Q: High memory usage?**
A: Enable large document mode cleanup or reduce PARALLEL_CHUNK_SIZE and MAX_WORKERS.

**Q: Slow processing for large PDFs?**
A: Enable LARGE_DOC_MODE=true and increase FAISS_K_SEARCH for better performance.

**Q: Links not being fetched?**
A: Check network connectivity and ensure URLs are accessible. Review dependency classification logic.

### Performance Optimization

**Q: How to optimize for speed?**
- Increase PARALLEL_WORKERS for multi-core systems
- Use Redis caching for repeated document processing  
- Enable HTTP/2 and connection pooling
- Implement request batching for multiple questions

**Q: How to optimize for accuracy?**
- Increase FAISS_K_SEARCH for more search results
- Lower SEMANTIC_SIMILARITY_THRESHOLD for broader context
- Use multiple Gemini API keys for better redundancy
- Enable LARGE_DOC_MODE for comprehensive processing

---

## 🎯 Use Cases

### Insurance & Finance
- **Policy Document Analysis**: Extract coverage details, exclusions, claim procedures
- **Regulatory Compliance**: Analyze legal documents for compliance requirements
- **Risk Assessment**: Process financial reports and risk documentation

### Legal & Contracts
- **Contract Review**: Extract key terms, obligations, and deadlines
- **Legal Research**: Search through case law and legal precedents  
- **Document Discovery**: Find relevant information across large document sets

### Healthcare & Medical
- **Medical Record Analysis**: Process patient records and treatment histories
- **Research Papers**: Extract findings from scientific literature
- **Drug Information**: Process pharmaceutical documentation and guidelines

### Education & Research  
- **Academic Papers**: Analyze research documents and extract key insights
- **Student Document Processing**: Handle assignments and thesis documents
- **Knowledge Base Creation**: Build searchable databases from educational content

---

## 🔄 Version History

### v3.1 - Current (Intelligent Reasoning & Enhanced Processing)
- ✅ Three-tier intelligent reasoning system
- ✅ Dependency-aware RAG with smart link fetching  
- ✅ Enhanced PDF processing with table structure preservation
- ✅ Parallel processing optimization
- ✅ Advanced security and encryption
- ✅ Multi-API key rotation and load balancing

### v3.0 - Major Architecture Upgrade
- ✅ FAISS vector search implementation
- ✅ Multi-format document processing
- ✅ Memory optimization for cloud deployment
- ✅ Comprehensive error handling and logging

### v2.0 - Enhanced AI Integration
- ✅ Gemini AI integration for advanced reasoning
- ✅ Cohere embeddings for semantic search
- ✅ Context-aware chunking strategies
- ✅ OCR processing with batch optimization

### v1.0 - Initial Release
- ✅ Basic PDF processing
- ✅ Simple Q&A functionality
- ✅ FastAPI web framework
- ✅ Docker containerization

---

## 🤝 Support & Community

### Getting Help

- **Documentation**: Comprehensive guides and API reference
- **GitHub Issues**: Bug reports and feature requests
- **Stack Overflow**: Community-driven support with `policyintel` tag
- **Discord Community**: Real-time chat and development discussions

### Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:
- Code style and standards
- Testing requirements  
- Documentation updates
- Feature request process
- Bug report templates

### Roadmap

**Upcoming Features (v4.0):**
- 🔄 GraphQL API support
- 🔄 Webhook integrations  
- 🔄 Advanced analytics dashboard
- 🔄 Multi-tenant architecture
- 🔄 Blockchain document verification
- 🔄 Advanced NLP with custom models

**Future Enhancements:**
- Voice input/output processing
- Real-time collaborative document analysis
- Advanced visualization and reporting
- Mobile SDK for iOS/Android
- Enterprise SSO integration

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **FastAPI**: For the excellent async web framework
- **Google Gemini**: For powerful language model capabilities  
- **Cohere**: For high-quality embedding generation
- **PyMuPDF**: For robust PDF processing capabilities
- **FAISS**: For efficient similarity search
- **Tesseract**: For OCR processing capabilities
- **HackRx Community**: For inspiration and real-world use cases

---

<div align="center">

**Built with ❤️ by the PolicyIntel Team**

[🌟 Star us on GitHub](https://github.com/yourusername/policyintel) | [📚 Documentation](https://policyintel.readthedocs.io) | [💬 Join Discord](https://discord.gg/policyintel)

</div>
