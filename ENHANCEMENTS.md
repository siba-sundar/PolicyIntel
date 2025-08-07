# PolicyIntel v3.1 - Parallel Processing & Intelligent Reasoning Enhancements

## New Features Added

### 1. Parallel Processing
- **Parallel Chunking**: Large documents are split into segments and processed concurrently
- **Parallel Embedding**: Embedding requests are batched and processed concurrently 
- **Parallel OCR**: PowerPoint images are collected and processed with OCR in parallel batches
- **Configurable Workers**: Settings to control parallelism levels for memory management

### 2. Cohere API Key Rotation
- **Dual Key Support**: Now supports 2 Cohere API keys (`COHERE_API_KEY` and `COHERE_API_KEY_2`)
- **Request-Level Rotation**: Keys rotate after each complete `/hackrx/run` request
- **Load Balancing**: Distributes API calls across keys to avoid rate limits

### 3. Intelligent Reasoning System
- **Three-Tier Answer Hierarchy**:
  1. **Direct Answer**: Extract exact answers from context when available
  2. **Pattern Deduction**: Find patterns, sequences, and logical rules from context examples
  3. **Intelligent Reasoning**: Use domain knowledge only when no patterns can be deduced

- **Priority**: Pattern deduction from context over standard knowledge
- **Mathematical Pattern Recognition**: Analyzes examples to determine underlying logic
- **Transparent Reasoning**: Clear indicators of reasoning level used

### 4. Enhanced PowerPoint Processing
- **Batch Image Collection**: Collects all images from slides first
- **Parallel OCR Processing**: Processes multiple images simultaneously
- **Improved Text Extraction**: Better handling of tables and text shapes

### 5. Improved ZIP File Processing
- **Single-Level Deep Processing**: Goes only one level deep into nested ZIPs
- **First-ZIP-Only Strategy**: If multiple ZIPs found, processes only the first one
- **Detailed Reporting**: Comprehensive summary of what was found and processed
- **Path Tracking**: Reports final file paths and processing status

## Configuration Settings Added

```python
# Parallel Processing Settings
MAX_WORKERS = min(4, os.cpu_count())              # Conservative worker count
PARALLEL_CHUNK_SIZE = 5                           # Chunks processed in parallel
PARALLEL_OCR_BATCH = 3                           # Images processed simultaneously  
PARALLEL_EMBEDDING_CONCURRENT = 2                # Concurrent embedding requests

# Enhanced LLM Settings
LLM_TEMPERATURE = 0.25                           # Higher for better reasoning
LLM_MAX_TOKENS = 300                            # More tokens for detailed responses
LLM_TOP_K = 30                                  # Increased for creativity
```

## API Key Management

### Gemini Keys (per-request rotation)
- Rotates after 12 requests per key
- Tracks: `current_gemini_key_index`, `gemini_request_count`

### Cohere Keys (per-complete-request rotation) 
- Rotates after each complete `/hackrx/run` request
- Tracks: `current_cohere_key_index`

## Intelligent Answer Processing

### Answer Hierarchy Implementation
1. **Context-Based**: "According to the document..."
2. **Pattern-Based**: "Following the pattern shown..." / "Based on similar examples..."
3. **Knowledge-Based**: "Using standard principles..." (only when no patterns found)

### Fallback Mechanisms
- Low relevance scores trigger hybrid reasoning approach
- No search results trigger intelligent reasoning with general knowledge
- Transparent confidence indicators in responses

## Performance Improvements

### Chunking Performance
- **Parallel Text Segmentation**: Large texts split into segments for concurrent processing
- **Multi-Strategy Parallel Processing**: Different chunking strategies run on segments simultaneously
- **Quality Scoring**: Parallel quality assessment and ranking

### Embedding Performance  
- **Concurrent Batch Processing**: Multiple embedding batches processed simultaneously
- **API Key Load Balancing**: Distributes load across available Cohere keys
- **Memory Optimization**: Efficient batch management and cleanup

### OCR Performance
- **Batch Collection**: All PowerPoint images collected before processing
- **ThreadPoolExecutor**: Parallel OCR processing with configurable workers
- **Error Isolation**: Individual image failures don't affect others

## Health Check Updates

The `/health` endpoint now reports:
- Both Gemini and Cohere key availability and current usage
- Parallel processing capabilities
- Intelligent reasoning features
- Memory optimization status

## Version Information
- **Version**: v3.1 - intelligent_reasoning_v3.1  
- **Status**: "Intelligent Multi-Format Document Processor with Parallel Processing & Reasoning"
- **New Features**: `intelligent_reasoning`, `context_deduction`, `fallback_intelligence`

## Environment Variables Required

```bash
COHERE_API_KEY=your_first_cohere_key
COHERE_API_KEY_2=your_second_cohere_key
GEMINI_API_KEY_1=your_first_gemini_key
GEMINI_API_KEY_2=your_second_gemini_key  
GEMINI_API_KEY_3=your_third_gemini_key
```

## Usage Notes

1. **Parallel Processing**: Automatically activates for large documents and multiple images
2. **API Key Rotation**: Happens automatically - no manual intervention needed
3. **Intelligent Answers**: Always provides best possible answer using the three-tier hierarchy
4. **Memory Management**: Enhanced cleanup with parallel processing considerations
5. **Performance**: Significantly faster processing for large documents with multiple images
