from django import forms
from .models import InsuranceQuery

class InsuranceQueryForm(forms.ModelForm):
    """Form for submitting insurance queries"""
    
    class Meta:
        model = InsuranceQuery
        fields = ['query_text']
        widgets = {
            'query_text': forms.Textarea(attrs={
                'rows': 4,
                'class': 'w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-custom-blue focus:border-custom-blue resize-none',
                'placeholder': 'Example: I had a knee surgery for Rs 75,000 in Mumbai. My policy has been active for 6 months. Will my claim be approved?',
                'required': True,
                'id': 'queryInput'
            })
        }
        labels = {
            'query_text': 'Enter your insurance query'
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['query_text'].label = 'Enter your insurance query'
        self.fields['query_text'].help_text = None


class ClaimSearchForm(forms.Form):
    """Form for searching previous claims"""
    
    search = forms.CharField(
        max_length=200,
        required=False,
        widget=forms.TextInput(attrs={
            'class': 'w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500',
            'placeholder': 'Search previous queries...'
        })
    )
    
    status = forms.ChoiceField(
        choices=[('', 'All Statuses')] + InsuranceQuery.CLAIM_STATUS_CHOICES,
        required=False,
        widget=forms.Select(attrs={
            'class': 'px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500'
        })
    )
