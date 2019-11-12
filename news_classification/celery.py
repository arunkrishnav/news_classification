import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'news_classification.settings')

app = Celery('news_classification')
app.config_from_object('django.conf:settings', namespace='CELERY')
app.autodiscover_tasks()
