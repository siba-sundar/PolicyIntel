#!/usr/bin/env python3
"""
Example script demonstrating the dependency-aware RAG system for HackRx puzzle.

This script shows how to:
1. Process a document with URLs
2. Ask questions that require external link fetching
3. Handle the HackRx-style puzzle logic
"""

import asyncio
import logging
import json
import sys
import time
from typing import Dict, List

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the app directory to Python path for imports
sys.path.append('./app')

try:
    from app.services.depedency_aware_retrieval import DependencyManager
    from app.services.llm_service import ask_llm
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error("Make sure you're running this from the PolicyIntel directory")
    sys.exit(1)


class SimpleDocumentProcessor:
    """Simple document processor for demonstration purposes"""
    
    def __init__(self):
        self.dependency_manager = DependencyManager(
            max_concurrent_requests=3,
            request_timeout=10,
            enable_security_checks=False  # Disable for demo
        )
    
    def chunk_text(self, text: str) -> List[Dict[str, str]]:
        """Simple text chunking by paragraphs"""
        paragraphs = text.split('\n\n')
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append({
                    'text': para.strip(),
                    'chunk_id': i,
                    'source': 'document'
                })
        
        return chunks
    
    async def process_question(self, document_text: str, question: str) -> Dict:
        """Process a single question with dependency-aware RAG"""
        logger.info(f"Processing question: {question}")
        
        # Simple chunking
        chunks = self.chunk_text(document_text)
        logger.info(f"Created {len(chunks)} chunks from document")
        
        # Apply dependency-aware processing
        try:
            enriched_chunks, dependencies = await self.dependency_manager.process_dependencies(
                question, chunks
            )
            
            # Get summary
            dep_summary = self.dependency_manager.get_dependency_summary(dependencies)
            
            # Log results
            logger.info(f"Dependency processing complete: {dep_summary}")
            
            if dependencies:
                logger.info("Link processing details:")
                for dep in dependencies:
                    status = "✅ Fetched" if dep.content else ("❌ Failed" if dep.is_required else "⏭️ Skipped")
                    logger.info(f"  - {dep.url}: {status}")
                    if dep.processing_steps:
                        for step in dep.processing_steps:
                            logger.info(f"    {step}")
            
            # Generate answer using enriched context
            if len(enriched_chunks) > len(chunks):
                logger.info("Using enriched context with fetched link data")
                answer_context = '\n\n'.join(enriched_chunks[:10])  # Limit context
            else:
                logger.info("Using original document chunks")
                answer_context = '\n\n'.join(enriched_chunks[:10])
            
            answer_prompt = f"""Based on the provided context, answer the following question concisely and accurately.

CONTEXT:
{answer_context}

QUESTION: {question}

ANSWER:"""
            
            try:
                answer = await ask_llm(answer_prompt)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                answer = "Unable to generate answer due to LLM error."
            
            return {
                'question': question,
                'answer': answer.strip(),
                'dependencies_found': len(dependencies),
                'dependencies_required': dep_summary['required'],
                'dependencies_fetched': dep_summary['fetched'],
                'success_rate': dep_summary['success_rate'],
                'enriched_chunks_count': len(enriched_chunks),
                'original_chunks_count': len(chunks)
            }
            
        except Exception as e:
            logger.error(f"Error in dependency processing: {e}")
            return {
                'question': question,
                'answer': f"Error processing question: {str(e)}",
                'error': True
            }


async def run_hackrx_example():
    """Run example with HackRx-style puzzle"""
    
    # Sample document with URLs (simulating HackRx PDF content)
    sample_document = """
HackRx Final Round Instructions

Welcome to the final round! Follow these steps carefully:

1. First, call https://register.hackrx.in/submissions/myFavouriteCity to get your assigned city.

2. Based on the city you receive, look up the corresponding landmark from the table below:
   
   City Mappings:
   - Mumbai -> Gateway of India  
   - Delhi -> Red Fort
   - Bangalore -> Lalbagh
   - Chennai -> Charminar
   - Kolkata -> Victoria Memorial

3. Next, determine your flight endpoint:
   - If city is Mumbai or Delhi: use getFifthCityFlightNumber 
   - If city is Bangalore: use getBangaloreFlightDetails
   - Otherwise: use getFifthCityFlightNumber

4. Call the appropriate endpoint at https://register.hackrx.in/teams/public/flights/{endpoint}

5. Extract your flight number from the response.

Additional Resources:
- Documentation: https://hackrx.in/docs
- Support: https://hackrx.in/support
"""
    
    # Initialize processor
    processor = SimpleDocumentProcessor()
    
    # Test questions
    test_questions = [
        "What is my flight number?",  # Should trigger dependency fetching
        "What are the city mappings?",  # Should not require external links
        "How many steps are there in the instructions?",  # Should not require external links
        "What endpoint should I use for Chennai?",  # Should not require external links but might fetch for verification
    ]
    
    logger.info("=" * 60)
    logger.info("HACKRX DEPENDENCY-AWARE RAG DEMONSTRATION")
    logger.info("=" * 60)
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n🔸 Question {i}/{len(test_questions)}")
        logger.info("-" * 40)
        
        start_time = time.time()
        result = await processor.process_question(sample_document, question)
        processing_time = time.time() - start_time
        
        result['processing_time'] = round(processing_time, 2)
        results.append(result)
        
        # Display result
        logger.info(f"❓ Q: {result['question']}")
        logger.info(f"💡 A: {result['answer']}")
        logger.info(f"⏱️  Processing time: {processing_time:.2f}s")
        
        if not result.get('error'):
            logger.info(f"🔗 Dependencies: {result['dependencies_found']} found, {result['dependencies_required']} required, {result['dependencies_fetched']} fetched")
            if result['success_rate'] > 0:
                logger.info(f"📊 Success rate: {result['success_rate']:.1f}%")
        
        logger.info("")  # Empty line for readability
    
    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    
    total_dependencies = sum(r.get('dependencies_found', 0) for r in results)
    total_required = sum(r.get('dependencies_required', 0) for r in results)
    total_fetched = sum(r.get('dependencies_fetched', 0) for r in results)
    
    logger.info(f"📋 Total questions processed: {len(results)}")
    logger.info(f"🔗 Total dependencies found: {total_dependencies}")
    logger.info(f"📥 Total dependencies required: {total_required}")
    logger.info(f"✅ Total dependencies fetched: {total_fetched}")
    
    if total_required > 0:
        overall_success = (total_fetched / total_required) * 100
        logger.info(f"🎯 Overall fetch success rate: {overall_success:.1f}%")
    
    return results


async def run_simple_example():
    """Run a simple example with mock data"""
    
    # Simple document with some URLs
    simple_doc = """
Company Policy Document

For employee verification, please check the status at https://company.example.com/verify
For more information about benefits, visit https://company.example.com/benefits

Our main office is located in New York. 
Contact HR at hr@company.com for questions.
"""
    
    processor = SimpleDocumentProcessor()
    
    logger.info("=" * 60)
    logger.info("SIMPLE DEPENDENCY-AWARE RAG EXAMPLE")
    logger.info("=" * 60)
    
    questions = [
        "How can I verify my employee status?",  # Should identify link dependency
        "Where is the main office located?",  # Should not require external links
    ]
    
    for question in questions:
        logger.info(f"\nProcessing: {question}")
        result = await processor.process_question(simple_doc, question)
        
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Dependencies: {result.get('dependencies_found', 0)} found, {result.get('dependencies_required', 0)} required")


def main():
    """Main function to run examples"""
    print("🚀 Starting Dependency-Aware RAG System Demo")
    print("=" * 60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Demo the dependency-aware RAG system")
    parser.add_argument('--mode', choices=['hackrx', 'simple', 'both'], default='both',
                      help='Which example to run')
    
    args = parser.parse_args()
    
    async def run_examples():
        if args.mode in ['simple', 'both']:
            await run_simple_example()
            
        if args.mode in ['hackrx', 'both']:
            if args.mode == 'both':
                print("\n" + "=" * 60 + "\n")
            await run_hackrx_example()
    
    try:
        asyncio.run(run_examples())
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        return 1
    
    logger.info("🎉 Demo completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())
