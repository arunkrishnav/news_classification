import logging

from celery import shared_task

from classification.models import Article
from classification.controller import n_gram_predict_from_ip

TASK_LOG = logging.getLogger('celery_task')


@shared_task
def send_result_and_update_article(batch):
    """
    Update articles with predicted category and submit results to end-point
    :param batch: [tuple] problem_id, category
    """
    for problem_id, category in batch.items():
        try:
            article = Article.objects.get(problem_id=problem_id)
            article.category = category
            article.status = 'completed'
            article.save()
            success = article.submit_result()
            if not success:
                print("Failed to submit results of %s" % problem_id)
        except Exception as error:
            print("Error occurred in submitting results of %s: %s" % (problem_id, error))
            TASK_LOG.error("Error occurred in submitting results of %s: %s\n", problem_id, error)


@shared_task
def predict_data_for_batch(batch):
    """
    :param batch:
    :return:
    """
    try:
        predicted_dict = n_gram_predict_from_ip(batch)
        send_result_and_update_article.delay(predicted_dict)
    except Exception as error:
        TASK_LOG.error("Error occurred in predicting data for following docs: %s \n %s\n",
                       [x[0] for x in batch], error)
