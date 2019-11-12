import logging

from django.db.models.signals import post_save
from django.dispatch import receiver

from classification.models import Article
from classification.tasks import predict_data_for_batch

CALLBACK_LOG = logging.getLogger('django')


@receiver(post_save, sender=Article)
def send_article_batch_for_prediction(sender, instance, created, *args, **kwargs):
    """
    Sends a batch of articles for prediction
    """
    batch_count = 50

    if not created:
        return

    articles = Article.objects.filter(status='not_processed')
    if articles.count() == batch_count:
        try:
            data = [(article.problem_id, article.title + " " + article.description) for article in articles]
            articles.update(status='processing')
            predict_data_for_batch.delay(data)
        except Exception as error:
            CALLBACK_LOG.error("Failed to send batch for prediction from callbacks: %s", error)


def complete_remaining():
    """
    Sends remaining set of articles for prediction
    NOTE: This function should be invoked only after `Article.fetch_problems()` is finished
    """
    try:
        status = ["not_processed", "processing"]
        articles = Article.objects.filter(status__in=status)
        data = [(article.problem_id, article.title + " " + article.description) for article in articles]
        articles.update(status='processing')
        predict_data_for_batch.delay(data)
    except Exception as error:
        CALLBACK_LOG.error("Failed to send remaining data-set for prediction: %s", error)
