import time
import re
import logging
import hashlib
from fastapi import APIRouter, HTTPException, Header
from app.models.schemas import QueryRequest, QueryResponse
from app.config.settings import qa_storage, FAISS_K_SEARCH
from app.utils.auth_utils import verify_token, rotate_cohere_key
from app.utils.memory_utils import log_memory_usage, cleanup_variables, clear_memory
from app.utils.security_utils import SimpleSecurityManager
from app.services.document_service import DocumentService
from app.services.chunking import get_enhanced_chunks
from app.services.embedding_service import get_embeddings
from app.services.faiss_search import FAISSSearchService
from app.services.llm_service import ask_llm, build_prompt
from app.config.settings import document_cache
from app.processors.pdf_processor import EnhancedPDFProcessor

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize security manager
security_manager = SimpleSecurityManager()

# Your original endpoint with enhanced PDF processing
startup_clear = document_cache.clear_cache(confirm=True)
logger.info(f"🔒 Startup with enhanced PDF processing - cache clear: {startup_clear}")

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    """Main query processing endpoint with enhanced PDF processing and smart link fetching"""
    global qa_storage
    
    total_start_time = time.time()
    request_id = f"req_{int(total_start_time * 1000)}"
    
    # Log initial memory state
    initial_memory = log_memory_usage("Request start")
    logger.info(f"📥 🔒 Received request {request_id} with {len(request.questions)} questions")
    
    # Auth check
    if not verify_token(authorization):
        audit_log = security_manager.create_audit_log({
            'request_id': request_id,
            'action': 'auth_failed',
            'error': 'invalid_token'
        })
        logger.error(f"🚫 Auth failed: {audit_log}")
        
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")
        else:
            raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # Basic encryption of sensitive data (non-blocking)
        logger.info("🔐 Encrypting input data...")
        encrypted_questions = security_manager.encrypt_data(request.questions, "questions")
        logger.info("✅ Input data encrypted successfully")
        
        # Log security event
        security_audit = security_manager.create_audit_log({
            'request_id': request_id,
            'action': 'data_encrypted',
            'questions_count': len(request.questions)
        })
        logger.info(f"🛡️ {security_audit}")
        
        # Initialize services
        document_service = DocumentService()
        faiss_service = FAISSSearchService()
        enhanced_pdf_processor = EnhancedPDFProcessor()
        
        # Process document with enhanced PDF processing
        logger.info("📄 Starting enhanced document processing")
        log_memory_usage("Before document processing")
        
        cached_result = document_cache.load_from_cache(request.documents)
        if cached_result:
            document_text, chunks, chunk_embeddings, extracted_links, fetched_content = cached_result
            logger.info("Using cached document processing results")
            log_memory_usage("After cache load")
        else:
            # Create question context for smart link prioritization
            question_context = " ".join(request.questions)
            
            # Check if we have PDF documents
            has_pdf = request.documents.lower().endswith('.pdf')
            
            if has_pdf:
                logger.info("🔗 Processing PDF with enhanced link extraction")
                
                # Use enhanced PDF processor
                document_text = ""
                extracted_links = []
                fetched_content = {}
                
                # Download PDF content
                pdf_content = await document_service.download_and_process_document(request.documents)
                document_text = pdf_content

                if enhanced_pdf_processor:
                    # Process with enhanced PDF processor
                    enhanced_text = enhanced_pdf_processor.process_sync(
                        file_content=pdf_content,
                        filename='document.pdf',
                        fetch_links=True,
                        max_links_to_fetch=15,  # Configurable
                        question_context=question_context
                    )
                    
                    document_text = enhanced_text
                    extracted_links.extend(enhanced_pdf_processor.get_extracted_links())
                    fetched_content.update(enhanced_pdf_processor.get_fetched_content())
            else:
                # Use standard document processing for non-PDF documents
                document_text = await document_service.download_and_process_document(request.documents)
                extracted_links = []
                fetched_content = {}
            
            log_memory_usage("After document download and link processing")
            
            if not document_text.strip():
                logger.error("No text extracted from document")
                raise HTTPException(status_code=400, detail="No text could be extracted from the document")
            
            # Create integrity hash for document (including fetched content)
            doc_hash = hashlib.sha256(document_text.encode()).hexdigest()[:16]
            logger.info(f"📋 Document integrity hash: {doc_hash}")
            
            logger.info("🧩 Starting document chunking")
            chunks = get_enhanced_chunks(document_text)
            log_memory_usage("After chunking")
            
            if not chunks:
                logger.error("No chunks created from document")
                raise HTTPException(status_code=400, detail="No meaningful chunks could be created from the document")
            
            logger.info("🤖 Getting embeddings")
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_embeddings = await get_embeddings(chunk_texts, input_type="search_document")
            log_memory_usage("After chunk embeddings")
            
            # Save to cache (including link data)
            document_cache.save_to_cache(request.documents, document_text, chunks, 
                                      chunk_embeddings, extracted_links, fetched_content)
        
        # Clear document text from memory after chunking
        cleanup_variables(document_text)
        clear_memory("Document text cleanup")
        
        # Decrypt questions for processing
        decrypted_questions = security_manager.decrypt_data(encrypted_questions)
        if not decrypted_questions:
            decrypted_questions = request.questions  # Fallback to original
        
        question_embeddings = await get_embeddings(decrypted_questions, input_type="search_query")
        log_memory_usage("After question embeddings")
        
        # Cleanup intermediate variables
        cleanup_variables(chunk_texts)
        clear_memory("Embeddings cleanup")
        
        # Create FAISS index with chunk embeddings
        logger.info("📊 Creating FAISS index")
        faiss_service.create_index(chunk_embeddings, chunks)
        log_memory_usage("After FAISS index creation")
        
        # Process questions with enhanced context awareness
        logger.info("❓ Processing questions with enhanced context")
        answers = []
        
        for i, (question, q_emb) in enumerate(zip(decrypted_questions, question_embeddings)):
            try:
                logger.info(f"📝 Processing question {i+1}/{len(decrypted_questions)}: {question[:50]}...")
                question_start_memory = log_memory_usage(f"Question {i+1} start")
                
                search_results = faiss_service.multi_tier_search(q_emb, question, FAISS_K_SEARCH)
                
                # Enhanced search with link context awareness
                try:
                    from app.services.enhanced_search import apply_enhanced_search_if_enabled
                    # Pass link information to enhanced search if available
                    search_context = {
                        'extracted_links': extracted_links if 'extracted_links' in locals() else [],
                        'fetched_content': fetched_content if 'fetched_content' in locals() else {}
                    }
                    search_results = apply_enhanced_search_if_enabled(search_results, question, search_context)
                except ImportError:
                    logger.info("Enhanced search module not available, using standard search")
                
                if not search_results:
                    logger.warning(f"No search results found for question: {question}")
                    
                    # Enhanced fallback that considers fetched content
                    fallback_context = ""
                    if 'fetched_content' in locals() and fetched_content:
                        # Try to find relevant fetched content for fallback
                        question_lower = question.lower()
                        for url, content_data in list(fetched_content.items())[:3]:
                            if any(word in content_data.get('content', '').lower() 
                                  for word in question_lower.split()[:3]):
                                fallback_context += f"\n\nRelevant external data from {url}:\n{content_data.get('content', '')[:500]}..."
                    
                    fallback_prompt = f"""You are an intelligent analyst. No specific context was found in the main document, but try to reason logically about the question.

**QUESTION:** {question}

**ADDITIONAL CONTEXT:**{fallback_context if fallback_context else " None available"}

**INSTRUCTIONS:**
- Look for mathematical patterns, logical sequences, or domain-specific rules
- If additional context is provided, use it to inform your answer
- If it's a mathematical question, try to find underlying logic or patterns
- Only use general knowledge if no logical patterns can be determined
- Be transparent about your reasoning approach
- Keep answer concise (40-80 words)
- Use figures (1, 2, 3) instead of words for numbers

**ANSWER (logical reasoning with available context):**"""
                    
                    try:
                        answer = await ask_llm(fallback_prompt)
                        if answer and not answer.startswith("Unable to generate"):
                            prefix = "Based on available context and general principles: " if fallback_context else "Based on general principles: "
                            answer = f"{prefix}{answer}"
                        else:
                            answer = "I'd need more specific information to provide a definitive answer for this question."
                    except Exception as e:
                        logger.error(f"Fallback answer generation failed: {str(e)}")
                        answer = "I'd need more specific information to provide a definitive answer for this question."
                    
                    answers.append(answer)
                    qa_storage.append([question, answer])
                    continue
                
                best_score = search_results[0].get('final_score', 0) if hasattr(search_results[0], 'get') and 'final_score' in search_results[0] else search_results[0].get('similarity_score', 0)
                
                if best_score < 0.15:
                    logger.info(f"Low relevance score ({best_score:.3f}) for question: {question[:50]}... Using hybrid approach with link context")
                    relevant_chunks = faiss_service.enhance_results(search_results[:5], question)
                    limited_context = "\n\n---\n\n".join(relevant_chunks[:5])
                    
                    # Enhance context with relevant fetched content
                    link_context = ""
                    if 'fetched_content' in locals() and fetched_content:
                        question_keywords = set(question.lower().split())
                        for url, content_data in fetched_content.items():
                            content_words = set(content_data.get('content', '').lower().split())
                            if len(question_keywords.intersection(content_words)) >= 2:
                                link_context += f"\n\nRelevant linked data from {url}:\n{content_data.get('content', '')[:400]}..."
                                break
                    
                    hybrid_prompt = f"""You are an intelligent data analyst. The available context has limited relevance, but analyze it carefully for patterns, rules, or examples.

**CONTEXT TO ANALYZE FOR PATTERNS:**
{limited_context}

**ADDITIONAL LINKED CONTEXT:**{link_context if link_context else " None available"}

**QUESTION:** {question}

**ANSWER (look for patterns first, then reasoning):**"""
                    
                    response = await ask_llm(hybrid_prompt)
                else:
                    relevant_chunks = faiss_service.enhance_results(search_results, question)
                    
                    # Enhance prompt with link context if relevant
                    link_enhancement = ""
                    if 'fetched_content' in locals() and fetched_content:
                        question_keywords = set(question.lower().split())
                        for url, content_data in list(fetched_content.items())[:2]:
                            content_words = set(content_data.get('content', '').lower().split())
                            if len(question_keywords.intersection(content_words)) >= 1:
                                link_enhancement += f"\n\nSupplementary data from {url}:\n{content_data.get('content', '')[:300]}...\n"
                    
                    if link_enhancement:
                        enhanced_prompt = f"""Based on the following context and supplementary data, please answer the question accurately and concisely.

**PRIMARY CONTEXT:**
{chr(10).join(relevant_chunks)}

**SUPPLEMENTARY LINKED DATA:**
{link_enhancement}

**QUESTION:** {question}

**INSTRUCTIONS:**
- Use the primary context as your main source
- Supplement with linked data when relevant
- Be precise and factual
- Use figures (1, 2, 3) instead of words for numbers
- Keep answer concise but complete

**ANSWER:**"""
                        response = await ask_llm(enhanced_prompt)
                    else:
                        prompt = build_prompt(question, relevant_chunks)
                        response = await ask_llm(prompt)
                
                # Clean response
                response = response.strip()
                response = re.sub(r'^(Answer:|ANSWER:|Response:)\s*', '', response, flags=re.IGNORECASE)
                response = re.sub(r'\s*\.$', '', response)
                
                final_answer = response.strip()
                answers.append(final_answer)
                qa_storage.append([question, final_answer])
                
                # Cleanup question-specific variables
                cleanup_variables(search_results, relevant_chunks, response)
                
                logger.info(f"✅ Question {i+1} processed successfully")
                clear_memory(f"Question {i+1} completion")
                log_memory_usage(f"Question {i+1} end")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1} '{question}': {str(e)}")
                error_answer = "Error processing the question. Please try again."
                answers.append(error_answer)
                qa_storage.append([question, error_answer])
        
        # Basic security: encrypt answers for logging
        encrypted_answers = security_manager.encrypt_data(answers, "answers")
        
        # Final cleanup of major variables
        cleanup_variables(chunks, chunk_embeddings, question_embeddings, decrypted_questions)
        if 'extracted_links' in locals():
            cleanup_variables(extracted_links, fetched_content)
        
        total_time = time.time() - total_start_time
        final_memory = log_memory_usage("Request completion")
        
        # Calculate memory usage statistics
        memory_delta = 0
        if initial_memory and final_memory:
            memory_delta = final_memory['process_memory_mb'] - initial_memory['process_memory_mb']
            logger.info(f"📊 Memory Delta: {memory_delta:+.2f}MB from start to finish")
        
        # Enhanced audit log with link processing info
        link_stats = {}
        if 'extracted_links' in locals():
            link_stats = {
                'total_links_found': len(extracted_links) if extracted_links else 0,
                'links_fetched': len(fetched_content) if fetched_content else 0,
                'high_priority_links': len([l for l in extracted_links if l.fetch_priority == 1]) if extracted_links else 0
            }
        
        final_audit = security_manager.create_audit_log({
            'request_id': request_id,
            'processing_time': total_time,
            'questions_processed': len(answers),
            'memory_delta_mb': memory_delta,
            'action': 'request_completed',
            **link_stats
        })
        
        logger.info(f"⏱️ 🔒 Request {request_id} completed in {total_time:.2f}s")
        logger.info(f"🛡️ {final_audit}")
        
        # Log enhanced Q&A results
        logger.info("📋 ALL QUESTIONS AND ANSWERS (Enhanced with Link Data):")
        for i, (q, a) in enumerate(qa_storage):
            logger.info(f"Q{i+1}: {q}")
            logger.info(f"A{i+1}: {a}")
            logger.info("---")
        
        qa_storage.clear()
        
        # Final comprehensive memory cleanup
        clear_memory("Complete request cleanup")
        
        # Rotate Cohere API key for the next request
        rotate_cohere_key()
        
        logger.info("clearing cache after successful request completion")
        cache_clear_result = document_cache.clear_cache(confirm=True)
        logger.info(f"Cache cleared: {cache_clear_result}")
        
        return QueryResponse(answers=answers)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        error_audit = security_manager.create_audit_log({
            'request_id': request_id,
            'error_type': 'http_exception',
            'error': str(he.detail),
            'action': 'request_failed'
        })
        logger.error(f"🚫 {error_audit}")
        
        cache_clear_result = document_cache.clear_cache(confirm=True)
        logger.info(f"Cache Cleared: {cache_clear_result}")
        raise he
        
    except Exception as e:
        logger.error(f"Unexpected error in run_query: {str(e)}", exc_info=True)
        error_audit = security_manager.create_audit_log({
            'request_id': request_id,
            'error_type': 'unexpected_error',
            'error': str(e),
            'action': 'system_failure'
        })
        logger.error(f"💥 {error_audit}")
        
        cache_clear_result = document_cache.clear_cache(confirm=True)
        logger.info(f"Cache Cleared: {cache_clear_result}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/hackrx/security/status")
async def get_security_status():
    """Get basic security system status"""
    return {
        "encryption": "AES-256-GCM",
        "integrity_verification": "HMAC-SHA256", 
        "audit_logging": "Enabled",
        "enhanced_pdf_processing": "Enabled",
        "smart_link_fetching": "Enabled",
        "status": "Operational",
        "features": [
            "Input data encryption",
            "Document integrity verification", 
            "Secure audit logging",
            "Memory cleanup",
            "Error handling",
            "Smart PDF link extraction",
            "Context-aware link prioritization",
            "Automatic API/data source integration"
        ]
    }


@router.get("/hackrx/links/analysis/{request_id}")
async def get_link_analysis(request_id: str):
    """Get detailed link analysis for a specific request (for debugging/monitoring)"""
    # This would typically fetch from a database or cache
    # For now, return a sample response structure
    return {
        "request_id": request_id,
        "message": "Link analysis endpoint - implement with persistent storage for production use",
        "expected_fields": [
            "total_links_extracted",
            "links_by_priority",
            "fetching_decisions", 
            "successful_fetches",
            "failed_fetches",
            "content_enhancement_impact"
        ]
    }


@router.post("/hackrx/links/configure")
async def configure_link_processing(
    max_links_to_fetch: int = 10,
    fetch_timeout: int = 10,
    enable_smart_fetching: bool = True,
    priority_keywords: list = None
):
    """Configure link processing parameters"""
    
    if priority_keywords is None:
        priority_keywords = ["api", "data", "current", "live", "real-time"]
    
    # In production, save these to configuration storage
    config = {
        "max_links_to_fetch": max_links_to_fetch,
        "fetch_timeout": fetch_timeout,
        "enable_smart_fetching": enable_smart_fetching,
        "priority_keywords": priority_keywords,
        "updated_at": time.time()
    }
    
    logger.info(f"Link processing configuration updated: {config}")
    
    return {
        "status": "success",
        "message": "Link processing configuration updated",
        "config": config
    }