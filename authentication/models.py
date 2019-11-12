from django.db import models

# Create your models here.


class Auth(models.Model):
    """
    Model to map authentication data in DB
    """
    team_id = models.IntegerField(null=True, blank=True)
    name = models.CharField(max_length=50, null=True, blank=True)
    auth_token = models.TextField(null=True, blank=True)
    team_token = models.CharField(max_length=50, null=True, blank=True)
