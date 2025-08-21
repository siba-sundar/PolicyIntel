"""
Service layer for processing insurance claims
Integrates with existing backend processing logic
"""
import asyncio
import time
from datetime import datetime
from django.utils import timezone
from .models import InsuranceQuery, ClaimReason

# Import your existing backend logic
import sys
import os

# Add the app directory to Python path to import existing modules
app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app')
if app_path not in sys.path:
    sys.path.append(app_path)

try:
    from app.api.endpoints.query import process_hackrx_query
    from app.config.settings import qa_storage
except ImportError as e:
    print(f"Warning: Could not import backend modules: {e}")
    process_hackrx_query = None


class InsuranceClaimService:
    """Service for processing insurance claims using existing backend logic"""
    
    @classmethod
    def process_query(cls, query_text: str) -> InsuranceQuery:
        """
        Process an insurance query using the existing backend logic
        
        Args:
            query_text: The insurance query text
            
        Returns:
            InsuranceQuery: The processed query with results
        """
        # Create the initial query record
        query = InsuranceQuery.objects.create(
            query_text=query_text,
            status='PENDING'
        )
        
        start_time = time.time()
        
        try:
            if process_hackrx_query:
                # Use existing backend logic
                result = cls._call_existing_backend(query_text)
            else:
                # Fallback to mock processing if backend not available
                result = cls._mock_process_query(query_text)
            
            # Update query with results
            processing_time = time.time() - start_time
            query.processed_at = timezone.now()
            query.processing_time_seconds = processing_time
            
            # Extract information from result
            if isinstance(result, dict):
                query.age = result.get('age')
                query.procedure = result.get('procedure')
                query.location = result.get('location')
                query.policy_age = result.get('policy_age')
                query.status = result.get('status', 'PENDING')
                query.payout = result.get('payout', 0)
                
                query.save()
                
                # Save reasoning
                reasons = result.get('reasons', [])
                for i, reason in enumerate(reasons):
                    ClaimReason.objects.create(
                        query=query,
                        reason_text=reason,
                        order=i
                    )
            
        except Exception as e:
            query.status = 'ERROR'
            query.processed_at = timezone.now()
            query.processing_time_seconds = time.time() - start_time
            query.save()
            
            # Create error reason
            ClaimReason.objects.create(
                query=query,
                reason_text=f"Processing error: {str(e)}",
                order=0
            )
        
        return query
    
    @classmethod
    def _call_existing_backend(cls, query_text: str) -> dict:
        """Call the existing FastAPI backend logic"""
        try:
            # Create a mock request object that matches what the backend expects
            class MockRequest:
                def __init__(self, query):
                    self.query = query
            
            request = MockRequest(query_text)
            
            # This would need to be adapted based on your exact backend structure
            # The process_hackrx_query function might need some modifications
            # to work in this context
            result = asyncio.run(process_hackrx_query(request))
            return result
            
        except Exception as e:
            print(f"Error calling existing backend: {e}")
            return cls._mock_process_query(query_text)
    
    @classmethod
    def _mock_process_query(cls, query_text: str) -> dict:
        """
        Mock processing for development/testing
        This simulates the backend processing logic
        """
        # Simple keyword-based analysis for demo purposes
        query_lower = query_text.lower()
        
        # Extract basic information
        age = None
        if 'age' in query_lower:
            # Simple age extraction
            import re
            age_match = re.search(r'age\s+(\d+)', query_lower)
            if age_match:
                age = int(age_match.group(1))
        
        # Extract procedure
        procedure = None
        if 'surgery' in query_lower:
            if 'knee' in query_lower:
                procedure = 'Knee surgery'
            elif 'heart' in query_lower:
                procedure = 'Heart surgery'
            else:
                procedure = 'Surgery'
        elif 'treatment' in query_lower:
            procedure = 'Treatment'
        
        # Extract location
        location = None
        cities = ['mumbai', 'delhi', 'bangalore', 'pune', 'chennai', 'kolkata']
        for city in cities:
            if city in query_lower:
                location = city.capitalize()
                break
        
        # Extract policy age
        policy_age = None
        if 'month' in query_lower:
            import re
            month_match = re.search(r'(\d+)\s*month', query_lower)
            if month_match:
                policy_age = f"{month_match.group(1)} months"
        elif 'year' in query_lower:
            import re
            year_match = re.search(r'(\d+)\s*year', query_lower)
            if year_match:
                policy_age = f"{year_match.group(1)} years"
        
        # Simple decision logic
        status = 'APPROVED'
        payout = 5000
        reasons = []
        
        # Basic approval logic
        if procedure and location:
            reasons.append(f"Medical procedure '{procedure}' is covered under policy")
            reasons.append(f"Treatment location '{location}' is within network")
        
        if policy_age:
            if 'month' in policy_age and int(policy_age.split()[0]) >= 3:
                reasons.append("Policy waiting period requirements met")
            elif 'year' in policy_age:
                reasons.append("Policy waiting period requirements met")
            else:
                status = 'REJECTED'
                payout = 0
                reasons = ["Policy waiting period not met (minimum 3 months required)"]
        
        # Extract amount if mentioned
        import re
        amount_match = re.search(r'rs\.?\s*(\d+(?:,\d+)*)', query_lower)
        if amount_match and status == 'APPROVED':
            amount_str = amount_match.group(1).replace(',', '')
            claimed_amount = int(amount_str)
            payout = min(claimed_amount, 75000)  # Cap at policy limit
            
            if payout < claimed_amount:
                reasons.append(f"Payout capped at policy limit of ₹75,000")
        
        return {
            'age': age,
            'procedure': procedure,
            'location': location,
            'policy_age': policy_age,
            'status': status,
            'payout': payout,
            'reasons': reasons
        }
    
    @classmethod
    def get_recent_queries(cls, limit: int = 10) -> list:
        """Get recent queries for display"""
        return InsuranceQuery.objects.select_related().prefetch_related('reasons').order_by('-submitted_at')[:limit]
    
    @classmethod
    def search_queries(cls, search_term: str = None, status: str = None) -> list:
        """Search queries based on text and status"""
        queryset = InsuranceQuery.objects.select_related().prefetch_related('reasons')
        
        if search_term:
            queryset = queryset.filter(query_text__icontains=search_term)
        
        if status:
            queryset = queryset.filter(status=status)
        
        return queryset.order_by('-submitted_at')[:50]
