import time

from django.conf import settings
from selenium import webdriver
from pyvirtualdisplay import Display

from .models import Category, Article

DRIVER_PATH = settings.BASE_DIR + '/bin/geckodriver'


class ArticleUrlScraper(object):
    """
    Scrapes the url of news articles of the related category
    """
    page_limit = 500  # 5000 articles
    max_retries = 3
    category_url_mappings = {
        'politics': 'https://southlive.in/category/newsroom/politics/',
    }
    link_xpath = '//ul[contains(@class, "mvp-blog-story-list")]/li[contains(@class, "mvp-blog-story-wrap")]/a'

    def __init__(self, category):
        self.driver = None
        self.extracted_urls = set()
        self._set_category(category)
        self.driver = webdriver.Firefox(executable_path=DRIVER_PATH)
        self.url = self.get_category_url()

    def _set_category(self, category):
        """
        Sets the category
        """
        if isinstance(category, str) and category.lower() in self.category_url_mappings.keys():
            self.category = category.lower()
        else:
            raise ValueError('Invalid category')

    def get_category_url(self):
        """
        Get the base url for a category
        """
        return self.category_url_mappings.get(self.category)

    def scrape_urls(self):
        """
        Fetches article urls and updates `self.extracted_urls`
        """
        page_link = 'page/{}/'
        page_count = 1
        while page_count < self.page_limit:
            retry_count = 0  # Tries to load page `max_retries` times before giving up
            while retry_count < self.max_retries:
                try:
                    self.driver.get(self.url + page_link.format(page_count))
                    break
                except:
                    retry_count += 1
            else:
                break
            time.sleep(2)
            if self.url not in self.driver.current_url:
                break
            urls = [elem.get_attribute('href') for elem in
                    self.driver.find_elements_by_xpath(self.link_xpath)]
            self.extracted_urls.update(urls)
            page_count += 1
        return self.extracted_urls

    def save_to_db(self):
        category = Category.objects.get(name=self.category)
        db_cache = []
        for url in self.extracted_urls:
            article = Article()
            article.url = url
            article.category = category
            article.source = Article.SOURCE_SL
            db_cache.append(article)
        objects = Article.objects.bulk_create(db_cache)
        if len(objects) == len(db_cache):
            print("Successfully created articles")
        else:
            print("Failed to create all articles"
                  "\tSuccess: %s"
                  "\tFailed: %s" % (len(objects), (len(db_cache)-len(objects))))


class ArticleDetailsScraper(object):
    """
    Scrapes the article url
    """
    driver = webdriver.Firefox(executable_path=DRIVER_PATH)
    title_xpath = '//header[@id="mvp-post-head"]/h1'
    content_xpath = '//div[@id="mvp-content-main"]/p'
    model_class = Article

    def __init__(self, article_url):
        self.url = article_url
        self.title = None
        self.content = None
        self.virtual_display = Display()
        self.virtual_display.start()
        self._initialize_driver()

    def _initialize_driver(self):
        """
        Initializes driver with article url
        """
        self.driver.get(self.url)

    def get_article_data(self):
        """
        Extracts article title and content from articles page
        """
        try:
            title_element = self.driver.find_element_by_xpath(self.title_xpath)
        except:
            try:
                title_element = self.driver.find_element_by_xpath(self.title_xpath)
            except Exception:
                return
        self.title = self.get_true_text(title_element).strip()

        try:
            content_elements = self.driver.find_elements_by_xpath(self.content_xpath)
        except:
            try:
                content_elements = self.driver.find_elements_by_xpath(self.content_xpath)
            except Exception:
                return
        self.content = '\n'.join([self.get_true_text(elem) for elem in content_elements])

    @staticmethod
    def get_true_text(tag):
        children = tag.find_elements_by_xpath('*')
        original_text = tag.text
        for child in children:
            original_text = original_text.replace(child.text, '', 1)
        return original_text

    @classmethod
    def update_article_details(cls):
        """
        Performs scrapping and updates details of article objects
        """
        articles = Article.objects.filter(title__isnull=True, content__isnull=True,
                                          source=Article.SOURCE_SL)
        updated_articles = []
        failed_articles = []
        errors = []
        for article in articles:
            try:
                try:
                    scraper = cls(article.url)
                except Exception as error:
                    try:
                        scraper = cls(article.url)
                    except Exception as error:
                        scraper = None
                if scraper:
                    scraper.get_article_data()
                    article.title = scraper.title
                    article.content = scraper.content
                    article.save()
                    updated_articles.append(article.id)
                else:
                    print("Failed to load page for %s" % article.url)
                    failed_articles.append(article.id)
            except Exception as error:
                errors.append(article.id)
        print("Updated articles: %s" % len(updated_articles))
        print("Failed to load %s articles" % len(failed_articles))
        print("Error occurred in scraping %s articles" % len(errors))
