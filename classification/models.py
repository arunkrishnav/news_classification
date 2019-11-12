import logging
import re
from collections import OrderedDict

from django.db import models

from authentication.handlers import APIHandler
from scraper.models import Article as ScraperArticle, Category

# Create your models here.

DJANGO_LOGS = logging.getLogger('django')
SCRIPT_LOGS = logging.getLogger('script')


class Article(models.Model):
    """
    Model to map articles
    """
    CATEGORY_CHOICES = (
        ('sports', 'Sports'),
        ('automobile', 'Automobile'),
        ('technology', 'Technology'),
        ('entertainment', 'Entertainment'),
        ('weather', 'Weather'),
        ('health', 'Health'),
        ('politics', 'Politics'),
        ('business', 'Business')
    )

    STATUS_CHOICES = (
        ('not_processed', 'Not Processed'),
        ('processing', 'Processing'),
        ('completed', 'Completed')
    )

    problem_id = models.CharField(max_length=50, null=True, blank=True)
    title = models.TextField(null=True, blank=True)
    description = models.TextField(null=True, blank=True)
    iteration = models.IntegerField(null=True, blank=True)
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, null=True, blank=True)
    score = models.IntegerField(null=True, blank=True)
    status = models.CharField(max_length=50, choices=STATUS_CHOICES, default="not_processed")
    is_cleaned = models.NullBooleanField(default=False)

    @classmethod
    def fetch_problems(cls):
        """
        Fetches and creates articles for evaluation
        """
        iteration = APIHandler.get_current_iteration()
        if not iteration:
            SCRIPT_LOGS.error("Failed to fetch current-iteration")
            return

        problem_set = APIHandler.get_problem_set()
        if not problem_set:
            SCRIPT_LOGS.error("Failed to fetch problem-set")
            return

        for problem in problem_set:
            try:
                problem_id = problem["id"]
                if not cls.objects.filter(problem_id=problem_id):
                    score = problem["score"]
                    problem_data = APIHandler.get_problem(problem_id)
                    if not problem_data:
                        SCRIPT_LOGS.error("Article not created. Failed to fetch"
                                          " problem-details for %s" % problem_id)
                        continue
                    title = problem_data.get("title")
                    description = problem_data.get("description")
                    category = None
                    Article.objects.create(title=title, description=description, iteration=iteration,
                                           category=category, score=score, problem_id=problem_id)
                    print(problem_data)
            except Exception as e:
                print("Failed to create Article for problem %s: %s" % (problem, e))
                continue

    def submit_result(self):
        """
        Submits the result to end-point
        :returns success status
        """
        data = {
            "problem": self.problem_id,
            "solution": {"category": self.category},
        }
        success_status, response = APIHandler.submit_result(data)
        return success_status

    @classmethod
    def generate_test_articles(cls):
        for i in range(300, 400):
            cls.objects.create(problem_id=i)

    def get_data_dict(self):
        """
        Returns an ordered dict of data to write in csv
        """
        return OrderedDict([
            ('category', self.category),
            ('text', re.sub('[\n\r]', ' ', self.title + ' ' + self.description)),
        ])

    @classmethod
    def export_articles_to_scraping_table(cls, iteration):
        """
        Exports problems to articles table in scraper app
        """
        articles_to_export = cls.objects.filter(iteration=iteration, is_cleaned=True)
        iter_string = 'iteration_%s' % iteration
        db_cache = []
        for article in articles_to_export:
            try:
                scraper_article = ScraperArticle()
                scraper_article.category = Category.objects.get(name=article.category)
                scraper_article.source = iter_string
                scraper_article.title = article.title
                scraper_article.content = article.description
                scraper_article.is_cleaned = True
                db_cache.append(scraper_article)
            except Exception as error:
                print('Failed to save article to scraper articles table: %s' % error)
        objects = ScraperArticle.objects.bulk_create(db_cache)
        if len(objects) == len(db_cache):
            print("Successfully created articles")
        else:
            print("Failed to create all articles"
                  "\tSuccess: %s"
                  "\tFailed: %s" % (len(objects), (len(db_cache)-len(objects))))
