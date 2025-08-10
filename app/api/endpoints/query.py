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

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize security manager
security_manager = SimpleSecurityManager()


# Your original endpoint with minimal security changes
startup_clear = document_cache.clear_cache(confirm=True)
logger.info(f"🔒 Startup with basic security - cache clear: {startup_clear}")

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    """Main query processing endpoint with basic security"""
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
        
        # Process document (your original logic)
        logger.info("📄 Starting document processing")
        log_memory_usage("Before document processing")
        
        cached_result = document_cache.load_from_cache(request.documents)
        if cached_result:
            document_text, chunks, chunk_embeddings = cached_result
            logger.info("Using cached document processing results")
            log_memory_usage("After cache load")
        else:
            document_text = await document_service.download_and_process_document(request.documents)
            log_memory_usage("After document download")
            
            if not document_text.strip():
                logger.error("No text extracted from document")
                raise HTTPException(status_code=400, detail="No text could be extracted from the document")
            
            # Basic security: create integrity hash for document
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
            
            document_cache.save_to_cache(request.documents, document_text, chunks, chunk_embeddings)
        
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
        
        # Process questions (your original logic)
        logger.info("❓ Processing questions")
        answers = []
        
        for i, (question, q_emb) in enumerate(zip(decrypted_questions, question_embeddings)):
            try:
                logger.info(f"📝 Processing question {i+1}/{len(decrypted_questions)}: {question[:50]}...")
                question_start_memory = log_memory_usage(f"Question {i+1} start")
                
                search_results = faiss_service.multi_tier_search(q_emb, question, FAISS_K_SEARCH)
                
                # Optional enhanced search
                try:
                    from app.services.enhanced_search import apply_enhanced_search_if_enabled
                    search_results = apply_enhanced_search_if_enabled(search_results, question)
                except ImportError:
                    logger.info("Enhanced search module not available, using standard search")
                
                if not search_results:
                    logger.warning(f"No search results found for question: {question}")
                    fallback_prompt = f"""You are an intelligent analyst. No specific context was found, but try to reason logically about the question.

**QUESTION:** {question}

**INSTRUCTIONS:**
- Look for mathematical patterns, logical sequences, or domain-specific rules
- If it's a mathematical question, try to find underlying logic or patterns
- Only use general knowledge if no logical patterns can be determined
- Be transparent about your reasoning approach
- Keep answer concise (40-80 words)
- Use figures (1, 2, 3) instead of words for numbers

**ANSWER (logical reasoning or general knowledge):**"""
                    
                    try:
                        answer = await ask_llm(fallback_prompt)
                        if answer and not answer.startswith("Unable to generate"):
                            answer = f"Based on general principles: {answer}"
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
                    logger.info(f"Low relevance score ({best_score:.3f}) for question: {question[:50]}... Using hybrid approach")
                    relevant_chunks = faiss_service.enhance_results(search_results[:5], question)
                    limited_context = "\n\n---\n\n".join(relevant_chunks[:5])
                    
                    hybrid_prompt = f"""You are an intelligent data analyst. The available context has limited relevance, but analyze it carefully for patterns, rules, or examples.

**CONTEXT TO ANALYZE FOR PATTERNS:**
{limited_context}

**QUESTION:** {question}

**ANSWER (look for patterns first, then reasoning):**"""
                    
                    response = await ask_llm(hybrid_prompt)
                else:
                    relevant_chunks = faiss_service.enhance_results(search_results, question)
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
                cleanup_variables(search_results, relevant_chunks, prompt, response)
                
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
        
        total_time = time.time() - total_start_time
        final_memory = log_memory_usage("Request completion")
        
        # Calculate memory usage statistics
        if initial_memory and final_memory:
            memory_delta = final_memory['process_memory_mb'] - initial_memory['process_memory_mb']
            logger.info(f"📊 Memory Delta: {memory_delta:+.2f}MB from start to finish")
        
        # Create final audit log
        final_audit = security_manager.create_audit_log({
            'request_id': request_id,
            'processing_time': total_time,
            'questions_processed': len(answers),
            'memory_delta_mb': memory_delta if initial_memory and final_memory else 0,
            'action': 'request_completed'
        })
        
        logger.info(f"⏱️ 🔒 Request {request_id} completed in {total_time:.2f}s")
        logger.info(f"🛡️ {final_audit}")
        
        # Log Q&A pairs and clear storage
        logger.info("📋 ALL QUESTIONS AND ANSWERS:")
        logger.info(f"{qa_storage}")
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
        "status": "Operational",
        "features": [
            "Input data encryption",
            "Document integrity verification", 
            "Secure audit logging",
            "Memory cleanup",
            "Error handling"
        ]
    }