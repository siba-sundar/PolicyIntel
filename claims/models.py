from django.db import models
from django.utils import timezone

class InsuranceQuery(models.Model):
    """Model to store insurance queries and their results"""
    CLAIM_STATUS_CHOICES = [
        ('PENDING', 'Pending'),
        ('APPROVED', 'Approved'),
        ('REJECTED', 'Rejected'),
    ]
    
    # Input data
    query_text = models.TextField(help_text="Original query text from user")
    submitted_at = models.DateTimeField(default=timezone.now)
    
    # Extracted information
    age = models.IntegerField(null=True, blank=True)
    procedure = models.CharField(max_length=200, null=True, blank=True)
    location = models.CharField(max_length=100, null=True, blank=True)
    policy_age = models.CharField(max_length=50, null=True, blank=True)
    
    # Decision results
    status = models.CharField(
        max_length=10, 
        choices=CLAIM_STATUS_CHOICES, 
        default='PENDING'
    )
    payout = models.DecimalField(
        max_digits=10, 
        decimal_places=2, 
        null=True, 
        blank=True,
        help_text="Payout amount in INR"
    )
    
    # Processing metadata
    processed_at = models.DateTimeField(null=True, blank=True)
    processing_time_seconds = models.FloatField(null=True, blank=True)
    
    class Meta:
        ordering = ['-submitted_at']
        verbose_name = "Insurance Query"
        verbose_name_plural = "Insurance Queries"
    
    def __str__(self):
        return f"Query {self.id} - {self.status} ({self.submitted_at.strftime('%Y-%m-%d %H:%M')})"


class ClaimReason(models.Model):
    """Model to store reasoning for claim decisions"""
    query = models.ForeignKey(
        InsuranceQuery, 
        on_delete=models.CASCADE,
        related_name='reasons'
    )
    reason_text = models.TextField()
    order = models.PositiveIntegerField(default=0)
    
    class Meta:
        ordering = ['order']
        verbose_name = "Claim Reason"
        verbose_name_plural = "Claim Reasons"
    
    def __str__(self):
        return f"Reason {self.order + 1} for Query {self.query.id}"
