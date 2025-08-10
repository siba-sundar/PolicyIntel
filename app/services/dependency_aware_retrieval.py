import re
import json
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Union
from urllib.parse import urlparse, urljoin
import aiohttp
from dataclasses import dataclass, field
from app.services.llm_service import ask_llm

logger = logging.getLogger(__name__)

@dataclass
class DynamicAPICall:
    """Represents a dynamically discovered API call"""
    name: str
    url: str
    method: str = "GET"
    parameters: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None
    priority: int = 1  # 1=highest (execute first), 2=conditional, 3=lowest
    executed: bool = False
    result: Optional[Dict] = None
    error: Optional[str] = None

@dataclass
class ExecutionContext:
    """Holds the execution state and results"""
    results: Dict[str, Any] = field(default_factory=dict)
    execution_order: List[str] = field(default_factory=list)
    failed_calls: List[str] = field(default_factory=list)

class SmartDynamicAPIExecutor:
    """
    Smart system that can work with ANY document and API structure.
    Uses LLM intelligence to understand context and execute appropriately.
    """
    
    def __init__(self, max_concurrent: int = 3, timeout: int = 15):
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.execution_cache = {}
        
        # Generic patterns for API detection
        self.api_patterns = [
            r'GET\s+(https?://[^\s<>"{}|\\^`\[\]]+)',
            r'POST\s+(https?://[^\s<>"{}|\\^`\[\]]+)', 
            r'call.*?endpoint.*?(https?://[^\s<>"{}|\\^`\[\]]+)',
            r'API.*?(https?://[^\s<>"{}|\\^`\[\]]+)',
            r'request.*?(https?://[^\s<>"{}|\\^`\[\]]+)',
            r'fetch.*?(https?://[^\s<>"{}|\\^`\[\]]+)',
            r'query.*?(https?://[^\s<>"{}|\\^`\[\]]+)'
        ]
        
        # Question patterns that likely need real data
        self.execution_triggers = [
            r'what\s+is\s+my\s+\w+',
            r'find\s+my\s+\w+',
            r'get\s+my\s+\w+', 
            r'my\s+\w+\s+(?:number|id|details)',
            r'what\s+\w+\s+(?:am\s+I|do\s+I)',
            r'current\s+\w+',
            r'actual\s+\w+',
            r'real\s+\w+'
        ]

    async def should_execute_apis(self, question: str, document_content: str) -> bool:
        """
        Use LLM to intelligently determine if API execution is needed for ANY context
        """
        question_lower = question.lower()
        content_lower = document_content.lower()
        
        # Quick pattern check first
        asks_for_specific_data = any(re.search(pattern, question_lower) for pattern in self.execution_triggers)
        has_api_endpoints = any(re.search(pattern, content_lower) for pattern in self.api_patterns)
        
        if not (asks_for_specific_data and has_api_endpoints):
            logger.info("❌ Quick check: No API execution needed")
            return False
        
        # Use LLM for intelligent decision making
        decision_prompt = f"""You are an API execution decision maker. Analyze if the question requires calling external APIs to get actual data.

**QUESTION:** {question}

**DOCUMENT CONTENT:** {document_content[:2000]}

**DECISION CRITERIA:**
1. Does the question ask for SPECIFIC data that only an API can provide?
2. Does the document describe a process that requires API calls?
3. Can the question be answered from document alone, or does it need live data?

**EXAMPLES:**
- "What is my flight number?" + Document with flight API steps → EXECUTE (needs real data)
- "How do I find my flight number?" + Document with instructions → NO_EXECUTE (just explaining process)
- "What is my balance?" + Document with balance API → EXECUTE (needs current balance)
- "How does authentication work?" + Document with auth API → NO_EXECUTE (explaining concept)

**RESPONSE FORMAT:** 
- "EXECUTE" if APIs should be called to get real data
- "NO_EXECUTE" if question can be answered from document content

**DECISION:**"""

        try:
            decision = await ask_llm(decision_prompt)
            should_execute = "EXECUTE" in decision.upper() and "NO_EXECUTE" not in decision.upper()
            
            logger.info(f"🤖 LLM Decision: {decision.strip()}")
            logger.info(f"🎯 Final decision: EXECUTE={should_execute}")
            
            return should_execute
            
        except Exception as e:
            logger.error(f"❌ LLM decision failed: {str(e)}")
            # Fallback to pattern-based decision
            return asks_for_specific_data and has_api_endpoints

    async def extract_dynamic_api_calls(self, question: str, document_content: str) -> List[DynamicAPICall]:
        """
        Use LLM to intelligently extract API calls and their execution logic from ANY document
        """
        
        # First extract all URLs
        all_urls = set()
        for pattern in self.api_patterns:
            matches = re.finditer(pattern, document_content, re.IGNORECASE)
            for match in matches:
                url = match.group(1) if match.groups() else match.group(0)
                if self._is_valid_url(url.strip()):
                    all_urls.add(url.strip())
        
        if not all_urls:
            logger.info("❌ No valid URLs found in document")
            return []
        
        logger.info(f"🔍 Found {len(all_urls)} potential API endpoints")
        
        # Use LLM to understand the API execution logic
        analysis_prompt = f"""You are an API execution planner. Analyze the document to understand the API calling logic.

**QUESTION:** {question}

**DOCUMENT:** {document_content[:3000]}

**FOUND APIs:** 
{json.dumps(list(all_urls), indent=2)}

**INSTRUCTIONS:**
Analyze the document to understand:
1. Which API should be called FIRST (usually to get initial data)
2. Which APIs are CONDITIONAL (called based on first API result)
3. The LOGICAL ORDER of execution
4. Any CONDITIONS or dependencies between calls

**RESPONSE FORMAT (JSON):**
```json
{
  "execution_plan": [
    {
      "name": "descriptive_name",
      "url": "actual_url",
      "method": "GET_or_POST",
      "priority": 1_or_2_or_3,
      "condition": "condition_description_or_null",
      "depends_on": ["list_of_prerequisite_call_names"]
    }
  ],
  "reasoning": "explanation_of_the_execution_logic"
}
```

**PRIORITY LEVELS:**
- 1: Execute FIRST (initial data calls)
- 2: Execute CONDITIONALLY (based on first call results) 
- 3: Execute OPTIONALLY (fallback or additional data)

**ANALYSIS:**"""

        try:
            analysis = await ask_llm(analysis_prompt)
            logger.info(f"🧠 LLM Analysis: {analysis[:500]}...")
            
            # Parse LLM response
            execution_plan = self._parse_execution_plan(analysis, all_urls)
            
            logger.info(f"📋 Parsed {len(execution_plan)} API calls from LLM analysis")
            return execution_plan
            
        except Exception as e:
            logger.error(f"❌ LLM analysis failed: {str(e)}")
            # Fallback: create simple execution plan
            return self._create_fallback_plan(all_urls)

    def _parse_execution_plan(self, analysis: str, available_urls: set) -> List[DynamicAPICall]:
        """Parse LLM execution plan into DynamicAPICall objects"""
        api_calls = []
        
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group(1))
            else:
                # Try to find JSON without code blocks
                json_match = re.search(r'(\{.*?"execution_plan".*?\})', analysis, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group(1))
                else:
                    raise ValueError("No valid JSON found")
            
            execution_plan = plan_data.get('execution_plan', [])
            
            for i, call_info in enumerate(execution_plan):
                url = call_info.get('url', '')
                if url in available_urls or any(available_url in url for available_url in available_urls):
                    api_call = DynamicAPICall(
                        name=call_info.get('name', f'api_call_{i}'),
                        url=url,
                        method=call_info.get('method', 'GET'),
                        priority=call_info.get('priority', 2),
                        condition=call_info.get('condition'),
                        depends_on=call_info.get('depends_on', [])
                    )
                    api_calls.append(api_call)
                    
            logger.info(f"✅ Successfully parsed {len(api_calls)} API calls from LLM")
            
        except Exception as e:
            logger.error(f"❌ Failed to parse LLM plan: {str(e)}")
            # Fallback to simple plan
            return self._create_fallback_plan(available_urls)
        
        return api_calls

    def _create_fallback_plan(self, urls: set) -> List[DynamicAPICall]:
        """Create simple fallback execution plan"""
        api_calls = []
        urls_list = list(urls)
        
        for i, url in enumerate(urls_list):
            # First URL gets priority 1, others get priority 2
            priority = 1 if i == 0 else 2
            depends_on = [f'api_call_0'] if i > 0 else []
            
            api_call = DynamicAPICall(
                name=f'api_call_{i}',
                url=url,
                method='GET',
                priority=priority,
                depends_on=depends_on
            )
            api_calls.append(api_call)
        
        logger.info(f"🔄 Created fallback plan with {len(api_calls)} API calls")
        return api_calls

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL"""
        try:
            result = urlparse(url)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except Exception:
            return False

    async def execute_dynamic_api_calls(self, api_calls: List[DynamicAPICall], question: str) -> ExecutionContext:
        """
        Execute API calls with smart conditional logic
        """
        context = ExecutionContext()
        
        if not api_calls:
            return context
        
        logger.info(f"🚀 EXECUTING {len(api_calls)} dynamic API calls")
        
        # Sort by priority (1=first, 2=conditional, 3=optional)
        sorted_calls = sorted(api_calls, key=lambda x: (x.priority, x.name))
        
        for api_call in sorted_calls:
            try:
                # Check dependencies
                if not self._check_dependencies(api_call, context):
                    logger.info(f"⏭️ Skipping {api_call.name} - dependencies not ready")
                    continue
                
                # Check conditions for priority 2+ calls
                if api_call.priority > 1 and not await self._should_execute_conditional_call(api_call, context, question):
                    logger.info(f"⏭️ Skipping {api_call.name} - conditions not met")
                    continue
                
                logger.info(f"🔗 EXECUTING: {api_call.name} -> {api_call.url}")
                
                # Execute the call
                result = await self._execute_single_api_call(api_call, context)
                
                if result:
                    api_call.result = result
                    api_call.executed = True
                    context.results[api_call.name] = result
                    context.execution_order.append(api_call.name)
                    
                    logger.info(f"✅ {api_call.name} SUCCESS - Data received")
                    logger.info(f"   Result preview: {str(result)[:200]}...")
                    
                    # If this was a priority 1 call, evaluate which conditional calls to make
                    if api_call.priority == 1 and len(sorted_calls) > 1:
                        await self._evaluate_conditional_calls(result, sorted_calls[1:], context, question)
                else:
                    context.failed_calls.append(api_call.name)
                    logger.error(f"❌ {api_call.name} FAILED - No data received")
                    
            except Exception as e:
                logger.error(f"💥 ERROR executing {api_call.name}: {str(e)}")
                api_call.error = str(e)
                context.failed_calls.append(api_call.name)
        
        logger.info(f"🏁 EXECUTION COMPLETE: {len(context.results)} successful, {len(context.failed_calls)} failed")
        return context

    def _check_dependencies(self, api_call: DynamicAPICall, context: ExecutionContext) -> bool:
        """Check if API call dependencies are satisfied"""
        if not api_call.depends_on:
            return True
        
        for dep in api_call.depends_on:
            if dep and dep not in context.results:
                return False
        
        return True

    async def _should_execute_conditional_call(self, api_call: DynamicAPICall, context: ExecutionContext, question: str) -> bool:
        """
        Use LLM to decide if conditional API call should be executed based on previous results
        """
        if api_call.priority == 1:  # Always execute priority 1 calls
            return True
        
        if not context.results:  # No previous results to base decision on
            return False
        
        # Use LLM to make conditional decision
        conditional_prompt = f"""You are deciding whether to execute a conditional API call based on previous API results.

**ORIGINAL QUESTION:** {question}

**PREVIOUS API RESULTS:**
{json.dumps(context.results, indent=2)}

**CONDITIONAL API TO CONSIDER:**
- Name: {api_call.name}
- URL: {api_call.url}
- Condition: {api_call.condition or 'Based on previous results'}

**DECISION CRITERIA:**
Does the previous API result contain data that would determine which conditional API to call?
Should this specific API be called based on the data received?

**RESPONSE:** 
- "YES" if this API should be called
- "NO" if this API should be skipped

**DECISION:**"""

        try:
            decision = await ask_llm(conditional_prompt)
            should_execute = "YES" in decision.upper()
            
            logger.info(f"🤖 Conditional decision for {api_call.name}: {decision.strip()}")
            return should_execute
            
        except Exception as e:
            logger.error(f"❌ Conditional decision failed: {str(e)}")
            return False

    async def _evaluate_conditional_calls(self, first_result: Dict, remaining_calls: List[DynamicAPICall], context: ExecutionContext, question: str):
        """
        Evaluate which conditional calls to make based on first API result
        """
        if not isinstance(first_result, dict):
            return
        
        logger.info(f"🔍 Evaluating {len(remaining_calls)} conditional calls based on: {str(first_result)[:100]}...")
        
        for api_call in remaining_calls:
            if api_call.priority == 2:  # Only evaluate conditional calls
                should_execute = await self._should_execute_conditional_call(api_call, context, question)
                if should_execute:
                    logger.info(f"🎯 Conditional match found - executing {api_call.name}")
                    result = await self._execute_single_api_call(api_call, context)
                    if result:
                        api_call.result = result
                        api_call.executed = True
                        context.results[api_call.name] = result
                        context.execution_order.append(api_call.name)
                    break  # Usually only execute one conditional call

    async def _execute_single_api_call(self, api_call: DynamicAPICall, context: ExecutionContext) -> Optional[Dict]:
        """Execute a single API call"""
        cache_key = hashlib.md5(f"{api_call.url}_{api_call.method}".encode()).hexdigest()
        
        # Check cache
        if cache_key in self.execution_cache:
            logger.info(f"📦 Using cached result for {api_call.url}")
            return self.execution_cache[cache_key]
        
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; SmartAPIExecutor/1.0)',
                'Accept': 'application/json, text/html, text/plain, */*',
                **api_call.headers
            }
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                logger.info(f"🌐 HTTP {api_call.method} {api_call.url}")
                
                if api_call.method.upper() == "GET":
                    async with session.get(api_call.url, headers=headers, ssl=False) as response:
                        result = await self._process_api_response(response)
                else:  # POST
                    data = api_call.parameters.get("data", {})
                    async with session.post(api_call.url, json=data, headers=headers, ssl=False) as response:
                        result = await self._process_api_response(response)
                
                # Cache successful results
                if result and response.status == 200:
                    self.execution_cache[cache_key] = result
                
                return result
                
        except Exception as e:
            logger.error(f"💥 HTTP request failed for {api_call.url}: {str(e)}")
            return None

    async def _process_api_response(self, response) -> Optional[Dict]:
        """Process HTTP response and extract meaningful data"""
        try:
            if response.status != 200:
                logger.warning(f"⚠️ HTTP {response.status} - {response.reason}")
                return None
            
            content_type = response.headers.get('content-type', '').lower()
            text_content = await response.text()
            
            logger.info(f"📄 Response: {len(text_content)} chars, Type: {content_type}")
            
            # Try JSON first
            if 'application/json' in content_type or text_content.strip().startswith(('{', '[')):
                try:
                    json_data = json.loads(text_content)
                    logger.info(f"✅ Parsed JSON: {json_data}")
                    return json_data
                except json.JSONDecodeError:
                    pass
            
            # Extract meaningful data from text
            extracted_data = self._extract_data_from_text(text_content)
            
            if extracted_data:
                logger.info(f"✅ Extracted data: {extracted_data}")
                return extracted_data
            
            # Return raw content as fallback
            return {
                "raw_content": text_content.strip()[:500],
                "full_text": text_content.strip()
            }
            
        except Exception as e:
            logger.error(f"💥 Response processing error: {str(e)}")
            return None

    def _extract_data_from_text(self, text: str) -> Dict[str, Any]:
        """Extract structured data from text response"""
        extracted = {}
        
        # Common patterns for data extraction
        patterns = {
            'flight_number': r'(?:flight\s+)?(?:number|id)[\s:]*([A-Z0-9]+)',
            'city': r'(?:city|location)[\s:]*([A-Za-z\s]+)',
            'landmark': r'(?:landmark|monument)[\s:]*([A-Za-z\s]+)',
            'number': r'(?:number|id|code)[\s:]*([A-Z0-9]+)',
            'name': r'(?:name)[\s:]*([A-Za-z\s]+)',
            'value': r'(?:value|result)[\s:]*([A-Za-z0-9\s]+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted[key] = match.group(1).strip()
        
        # If text is short and looks like a single value, use it directly
        if len(text.strip()) < 50 and not extracted:
            extracted['value'] = text.strip()
        
        return extracted

    def create_enhanced_context(self, original_content: str, context: ExecutionContext, api_calls: List[DynamicAPICall]) -> str:
        """Create context enhanced with REAL API execution results"""
        
        if not context.results:
            return original_content
        
        enhanced_content = original_content + "\n\n=== LIVE API EXECUTION RESULTS ===\n"
        
        executed_calls = [call for call in api_calls if call.executed]
        
        for api_call in executed_calls:
            enhanced_content += f"\n--- {api_call.name.upper()} EXECUTED ---\n"
            enhanced_content += f"URL: {api_call.url}\n"
            enhanced_content += f"Status: SUCCESS\n"
            
            if api_call.result:
                enhanced_content += f"Result: {json.dumps(api_call.result, indent=2)}\n"
            
            enhanced_content += "---\n"
        
        enhanced_content += f"\n=== EXECUTION SUMMARY ===\n"
        enhanced_content += f"Total API Calls: {len(api_calls)}\n"
        enhanced_content += f"Successfully Executed: {len(context.results)}\n"
        enhanced_content += f"Failed: {len(context.failed_calls)}\n"
        enhanced_content += f"Execution Order: {' -> '.join(context.execution_order)}\n"
        enhanced_content += "=== END LIVE RESULTS ===\n"
        
        return enhanced_content

    async def process_with_aggressive_execution(self, question: str, document_chunks: List[Dict[str, Any]]) -> Tuple[str, List[DynamicAPICall], ExecutionContext]:
        """
        Main processing with smart dynamic execution for ANY document/API structure
        """
        
        logger.info("🧠 Starting SMART dynamic API execution")
        
        # Combine document content
        document_content = "\n\n".join([chunk.get('text', '') for chunk in document_chunks])
        
        # Step 1: Intelligent decision on whether to execute APIs
        should_execute = await self.should_execute_apis(question, document_content)
        
        execution_context = ExecutionContext()
        api_calls = []
        
        if should_execute:
            logger.info("🚀 EXECUTION REQUIRED - analyzing document for API structure")
            
            # Step 2: Smart extraction of API calls and execution logic
            api_calls = await self.extract_dynamic_api_calls(question, document_content)
            
            if api_calls:
                logger.info(f"🎯 Found {len(api_calls)} API calls - executing with smart logic")
                # Step 3: Execute APIs with conditional logic
                execution_context = await self.execute_dynamic_api_calls(api_calls, question)
            else:
                logger.warning("⚠️ Execution required but no valid API calls found")
        else:
            logger.info("❌ No API execution needed - can answer from document")
        
        # Step 4: Create enhanced context with real results
        enhanced_content = self.create_enhanced_context(document_content, execution_context, api_calls)
        
        logger.info(f"🏁 Smart execution completed: {len(execution_context.results)} real API calls executed")
        
        return enhanced_content, api_calls, execution_context

    def clear_cache(self):
        """Clear execution cache"""
        cache_size = len(self.execution_cache)
        self.execution_cache.clear()
        logger.info(f"🗑️ Cleared execution cache: {cache_size} items")
        return cache_size