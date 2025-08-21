from django.contrib import admin
from .models import InsuranceQuery, ClaimReason

class ClaimReasonInline(admin.TabularInline):
    model = ClaimReason
    extra = 0
    readonly_fields = ('order',)

@admin.register(InsuranceQuery)
class InsuranceQueryAdmin(admin.ModelAdmin):
    list_display = ('id', 'status', 'procedure', 'location', 'payout', 'submitted_at')
    list_filter = ('status', 'submitted_at', 'processed_at', 'location')
    search_fields = ('query_text', 'procedure', 'location')
    readonly_fields = ('submitted_at', 'processed_at', 'processing_time_seconds')
    inlines = [ClaimReasonInline]
    
    fieldsets = (
        ('Query Information', {
            'fields': ('query_text', 'submitted_at')
        }),
        ('Extracted Details', {
            'fields': ('age', 'procedure', 'location', 'policy_age')
        }),
        ('Decision', {
            'fields': ('status', 'payout')
        }),
        ('Processing Info', {
            'fields': ('processed_at', 'processing_time_seconds'),
            'classes': ('collapse',)
        })
    )

@admin.register(ClaimReason)
class ClaimReasonAdmin(admin.ModelAdmin):
    list_display = ('query', 'reason_text_short', 'order')
    list_filter = ('query__status',)
    search_fields = ('reason_text', 'query__query_text')
    
    def reason_text_short(self, obj):
        return obj.reason_text[:100] + '...' if len(obj.reason_text) > 100 else obj.reason_text
    reason_text_short.short_description = 'Reason Text'
