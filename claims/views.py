from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.views.generic import ListView
import json

from .forms import InsuranceQueryForm, ClaimSearchForm
from .models import InsuranceQuery
from .services import InsuranceClaimService


def home(request):
    """Main page for insurance claims"""
    form = InsuranceQueryForm()
    context = {
        'form': form,
        'page_title': 'HackRx AI Claims Assistant'
    }
    return render(request, 'claims/home.html', context)


def submit_query(request):
    """Handle insurance query submission"""
    if request.method == 'POST':
        form = InsuranceQueryForm(request.POST)
        if form.is_valid():
            query_text = form.cleaned_data['query_text']
            
            # Process the query using our service
            query = InsuranceClaimService.process_query(query_text)
            
            # Add success message
            messages.success(request, 'Your query has been processed successfully!')
            
            # Redirect to results page
            return redirect('claims:query_result', query_id=query.id)
        else:
            messages.error(request, 'Please correct the errors below.')
    else:
        form = InsuranceQueryForm()
    
    return render(request, 'claims/home.html', {'form': form})


def query_result(request, query_id):
    """Display the results of a processed query"""
    query = get_object_or_404(InsuranceQuery, id=query_id)
    
    context = {
        'query': query,
        'reasons': query.reasons.all().order_by('order'),
        'page_title': f'Query Result #{query.id}'
    }
    
    return render(request, 'claims/result.html', context)


@csrf_exempt
@require_http_methods(["POST"])
def api_submit_query(request):
    """API endpoint for submitting queries (for AJAX requests)"""
    try:
        data = json.loads(request.body)
        query_text = data.get('query')
        
        if not query_text:
            return JsonResponse({
                'error': 'Query text is required'
            }, status=400)
        
        # Process the query
        query = InsuranceClaimService.process_query(query_text)
        
        # Return the results in the expected format
        response_data = {
            'age': query.age,
            'procedure': query.procedure,
            'location': query.location,
            'policy_age': query.policy_age,
            'status': query.status,
            'payout': float(query.payout) if query.payout else 0,
            'reasons': [reason.reason_text for reason in query.reasons.all().order_by('order')]
        }
        
        return JsonResponse(response_data)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        return JsonResponse({
            'error': f'Processing error: {str(e)}'
        }, status=500)


class QueryHistoryView(ListView):
    """View for displaying query history"""
    model = InsuranceQuery
    template_name = 'claims/history.html'
    context_object_name = 'queries'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = InsuranceQuery.objects.prefetch_related('reasons').order_by('-submitted_at')
        
        # Handle search
        search_term = self.request.GET.get('search')
        status_filter = self.request.GET.get('status')
        
        if search_term:
            queryset = queryset.filter(query_text__icontains=search_term)
        
        if status_filter:
            queryset = queryset.filter(status=status_filter)
        
        return queryset
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['search_form'] = ClaimSearchForm(self.request.GET)
        context['page_title'] = 'Query History'
        return context


def query_detail(request, query_id):
    """Detailed view of a specific query"""
    query = get_object_or_404(InsuranceQuery, id=query_id)
    
    context = {
        'query': query,
        'reasons': query.reasons.all().order_by('order'),
        'page_title': f'Query #{query.id} Details'
    }
    
    return render(request, 'claims/detail.html', context)
