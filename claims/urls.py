from django.urls import path
from . import views

app_name = 'claims'

urlpatterns = [
    # Main pages
    path('', views.home, name='home'),
    path('submit/', views.submit_query, name='submit_query'),
    path('result/<int:query_id>/', views.query_result, name='query_result'),
    path('history/', views.QueryHistoryView.as_view(), name='history'),
    path('detail/<int:query_id>/', views.query_detail, name='detail'),
    
    # API endpoints
    path('api/submit/', views.api_submit_query, name='api_submit_query'),
]
