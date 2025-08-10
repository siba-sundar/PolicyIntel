# app/api/endpoints/query.py
import time
import re
import logging
from fastapi import APIRouter, HTTPException, Header
from app.models.schemas import QueryRequest, QueryResponse
from app.config.settings import qa_storage, FAISS_K_SEARCH
from app.utils.auth_utils import verify_token, rotate_cohere_key
from app.utils.memory_utils import log_memory_usage, cleanup_variables, clear_memory
from app.services.document_service import DocumentService
from app.services.chunking import get_enhanced_chunks
from app.services.embedding_service import get_embeddings
from app.services.faiss_search import FAISSSearchService
from app.services.llm_service import ask_llm, build_prompt
from app.config.settings import document_cache

logger = logging.getLogger(__name__)
router = APIRouter()



startup_clear = document_cache.clear_cache(confirm=True)
logger.info(f"startup cache clear: {startup_clear}")

@router.post("/hackrx/run", response_model=QueryResponse)
async def run_query(request: QueryRequest, authorization: str = Header(None)):
    """Main query processing endpoint"""
    global qa_storage
    
    total_start_time = time.time()
    
    # Log initial memory state
    initial_memory = log_memory_usage("Request start")
    logger.info(f"📥 Received request with {len(request.questions)} questions")
    
    # Auth check
    if not verify_token(authorization):
        if not authorization or not authorization.startswith("Bearer "):
            logger.error("Missing or malformed authorization header")
            raise HTTPException(status_code=401, detail="Authorization header missing or malformed.")
        else:
            logger.error("Invalid token provided")
            raise HTTPException(status_code=403, detail="Invalid token")

    try:
        # Initialize services
        document_service = DocumentService()
        faiss_service = FAISSSearchService()
        
        # Process document with enhanced multi-format support
        logger.info("📄 Starting enhanced document processing")
        log_memory_usage("Before document processing")
        
        
        #check if document is cached
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
            
            # Use improved semantic chunking
            logger.info("🧩 Starting document chunking")
            chunks = get_enhanced_chunks(document_text)
            log_memory_usage("After chunking")
            
            if not chunks:
                logger.error("No chunks created from document")
                raise HTTPException(status_code=400, detail="No meaningful chunks could be created from the document")
            
            # Get embeddings for chunks
            logger.info("🤖 Getting embeddings")
            chunk_texts = [chunk["text"] for chunk in chunks]
            chunk_embeddings = await get_embeddings(chunk_texts, input_type="search_document")
            log_memory_usage("After chunk embeddings")
            
            # Save to cache if document is large enough
            document_cache.save_to_cache(request.documents, document_text, chunks, chunk_embeddings)
            
            
        # Clear document text from memory after chunking
        cleanup_variables(document_text)
        clear_memory("Document text cleanup")
        
        question_embeddings = await get_embeddings(request.questions, input_type="search_query")
        log_memory_usage("After question embeddings")
        
        # Cleanup intermediate variables
        cleanup_variables(chunk_texts)
        clear_memory("Embeddings cleanup")
        
        # Create FAISS index with chunk embeddings
        logger.info("📊 Creating FAISS index")
        faiss_service.create_index(chunk_embeddings, chunks)
        log_memory_usage("After FAISS index creation")
        
        # Process questions
        logger.info("❓ Processing questions")
        answers = []
        for i, (question, q_emb) in enumerate(zip(request.questions, question_embeddings)):
            try:
                logger.info(f"📝 Processing question {i+1}/{len(request.questions)}: {question[:50]}...")
                question_start_memory = log_memory_usage(f"Question {i+1} start")
                
                # Use multi-tier FAISS search for comprehensive results
                search_results = faiss_service.multi_tier_search(q_emb, question, FAISS_K_SEARCH)
                
                # Optional enhanced search (only if USE_ENHANCED_SEARCH=true)
                try:
                    from app.services.enhanced_search import apply_enhanced_search_if_enabled
                    search_results = apply_enhanced_search_if_enabled(search_results, question)
                except ImportError:
                    logger.info("Enhanced search module not available, using standard search")
                
                if not search_results:
                    logger.warning(f"No search results found for question: {question}")
                    # Use intelligent reasoning even without search results
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
                            # Prefix to indicate this is based on general knowledge
                            answer = f"Based on general insurance principles: {answer}"
                        else:
                            answer = "I'd need more specific policy information to provide a definitive answer for this question."
                    except Exception as e:
                        logger.error(f"Fallback answer generation failed: {str(e)}")
                        answer = "I'd need more specific policy information to provide a definitive answer for this question."
                    
                    answers.append(answer)
                    qa_storage.append([question, answer])
                    continue
                
                # Check if search results have good relevance scores
                best_score = search_results[0].get('final_score', 0) if hasattr(search_results[0], 'get') and 'final_score' in search_results[0] else search_results[0].get('similarity_score', 0)
                
                if best_score < 0.15:  # Low relevance threshold
                    logger.info(f"Low relevance score ({best_score:.3f}) for question: {question[:50]}... Using hybrid approach")
                    
                    # Combine context with intelligent reasoning
                    relevant_chunks = faiss_service.enhance_results(search_results[:5], question)  # Use fewer chunks
                    limited_context = "\n\n---\n\n".join(relevant_chunks[:5])
                    
                    hybrid_prompt = f"""You are an intelligent data analyst. The available context has limited relevance, but analyze it carefully for patterns, rules, or examples.

**APPROACH:**
1. Look for ANY patterns, sequences, or mathematical operations in the context
2. Find similar examples and deduce the underlying rule or logic
3. Apply the discovered pattern to answer the question
4. If no patterns exist, then provide reasoning based on the domain
5. Be transparent: "Following the pattern...", "Based on similar examples...", "The data shows..."

**CONTEXT TO ANALYZE FOR PATTERNS:**
{limited_context}

**QUESTION:** {question}

**ANSWER (look for patterns first, then reasoning):**"""
                    
                    response = await ask_llm(hybrid_prompt)
                else:
                    # Good relevance - use normal approach
                    relevant_chunks = faiss_service.enhance_results(search_results, question)
                    prompt = build_prompt(question, relevant_chunks)
                    response = await ask_llm(prompt)
                
                # Clean response
                response = response.strip()
                response = re.sub(r'^(Answer:|ANSWER:|Response:)\s*', '', response, flags=re.IGNORECASE)
                response = re.sub(r'\s*\.$', '', response)
                
                final_answer = response.strip()
                answers.append(final_answer)
                
                # Store Q&A pair
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
        
        # Final cleanup of major variables
        cleanup_variables(chunks, chunk_embeddings, question_embeddings)
        
        total_time = time.time() - total_start_time
        final_memory = log_memory_usage("Request completion")
        
        # Calculate memory usage statistics
        if initial_memory and final_memory:
            memory_delta = final_memory['process_memory_mb'] - initial_memory['process_memory_mb']
            logger.info(f"📊 Memory Delta: {memory_delta:+.2f}MB from start to finish")
        
        logger.info(f"⏱️ Total request processed in {total_time:.2f}s")
        
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
        logger.info(f"Cache cleared:{cache_clear_result}")
        
        return QueryResponse(answers=answers)
        
    except HTTPException as he:
        logger.error(f"HTTP Exception: {he.detail}")
        cache_clear_result = document_cache.clear_cache(confirm=True)
        logger.info(f"Cache Cleared:{cache_clear_result}")
        raise he
    except Exception as e:
        logger.error(f"Unexpected error in run_query: {str(e)}", exc_info=True)
        cache_clear_result = document_cache.clear_cache(confirm=True)
        logger.info(f"Cache Cleared:{cache_clear_result}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")