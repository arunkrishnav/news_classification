import time
from http.client import BadStatusLine

from django.conf import settings
from pyvirtualdisplay import Display
from selenium import webdriver

from scraper.models import Category, Article


class UrlScraper(object):
    """
    Scrapes the url of news articles of the related category
    """
    scroll_limit = 500000
    category_topic_url_mappings = {
        'sports': 'http://m.dailyhunt.in/news/india/malayalam/kayikam-topics-17',
        'automobile': 'https://m.dailyhunt.in/news/india/malayalam/ottomobail-topics-101',
        'technology': 'https://m.dailyhunt.in/news/india/malayalam/deknealaji-topics-102',
        'entertainment': 'https://m.dailyhunt.in/news/india/malayalam/vinodham-topics-8',
        'weather': '',
        'health': '',
        'politics': '',
        'business': 'https://m.dailyhunt.in/news/india/malayalam/bisinas+dhanakaryam-topics-2'
    }

    category_newspaper_url_mappings = {
        'sports': [],
        'automobile': [
            ('https://m.dailyhunt.in/news/india/malayalam/drivespark+malayalam-epaper-drivemal',
             '//*[@id="lang_home"]'),
            ('https://m.dailyhunt.in/news/india/malayalam/drivespark+malayalam-epaper-drivemal',
             '//*[@id="lang_fourwheelers"]'),
            ('https://m.dailyhunt.in/news/india/malayalam/drivespark+malayalam-epaper-drivemal',
             '//*[@id="lang_twowheelers"]'),
            ('https://m.dailyhunt.in/news/india/malayalam/drivespark+malayalam-epaper-drivemal',
             '//*[@id="lang_offbeat"]'),
        ],
        'technology': [],
        'entertainment': [],
        'weather': [],
        'health': [
            ('https://m.dailyhunt.in/news/india/malayalam/janam+tv-epaper-janamtv',
             '//*[@id="lang_health"]'),
            ('https://m.dailyhunt.in/news/india/malayalam/news60+malayalam-epaper-nwsixtym',
             '//*[@id="lang_health"]'),
            ('https://m.dailyhunt.in/news/india/malayalam/falconpost-epaper-falcon',
             '//*[@id="lang_health"]'),
            ('https://m.dailyhunt.in/news/india/malayalam/evening+kerala-epaper-evekeral',
             '//*[@id="lang_health"]'),
            # ('https://m.dailyhunt.in/news/india/malayalam/malayalam+breaking+news-epaper-malbrne',
            #  '//*[@id="lang_health"]'),
            # ('https://m.dailyhunt.in/news/india/malayalam/rashtradeepika-epaper-rasdep',
            #  '//*[@id="lang_health"]'),
            # ('https://m.dailyhunt.in/news/india/malayalam/samakalikamalayalam-epaper-samaka',
            #  '//*[@id="lang_health"]'),
        ],
        'politics': [
            # ('https://m.dailyhunt.in/news/india/malayalam/express+kerala-epaper-exkerala',
            #  '//*[@id="lang_politics"]'),
            # ('https://m.dailyhunt.in/news/india/malayalam/evening+kerala-epaper-evekeral',
            #  '//*[@id="lang_politics"]'),
            ('https://m.dailyhunt.in/news/india/malayalam/malayalivartha+new-epaper-malvart',
             '//*[@id="lang_politics"]'),
        ],
        'business': []
    }
    topic_scraping_id = 'topicHeadline'
    paper_scraping_id = 'newspaperHeadline'
    link_xpath = '//ul[@id="{}"]/li[@class="lang_ml"]/figure/figcaption/h2/a'

    def __init__(self, category, url=None):
        self.driver = None
        self.extracted_urls = []
        self.virtual_display = Display()
        self.virtual_display.start()
        self._set_category(category)
        self._initialize_driver(url)

    def _set_category(self, category):
        """
        Sets the category
        """
        if isinstance(category, str) and category.lower() in self.category_topic_url_mappings.keys():
            self.category = category.lower()
        else:
            raise ValueError('Invalid category')

    def _initialize_driver(self, url=None):
        """
        Initializes driver with category url
        """
        driver_path = settings.BASE_DIR + '/bin/geckodriver'
        self.driver = webdriver.Firefox(executable_path=driver_path)
        if url:
            self.link_xpath = self.get_link_xpath(False)
            self.driver.get(url)
        else:
            self.link_xpath = self.get_link_xpath()
            self.driver.get(self.get_category_url())

    def get_category_url(self):
        """
        Get the base url for a category
        """
        return self.category_topic_url_mappings.get(self.category)

    def get_link_xpath(self, is_topic_scraping=True):
        """
        Get the link xpath for an article group
        """
        if is_topic_scraping:
            return self.link_xpath.format(self.topic_scraping_id)
        else:
            return self.link_xpath.format(self.paper_scraping_id)

    def scroll_to_page_end(self):
        """
        Scrolls to page end
        :return: len of the page
        """
        try:
            len_of_page = self.driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);"
                "var lenOfPage=document.body.scrollHeight;"
                "return lenOfPage;"
            )
            time.sleep(2)
        except Exception as error:
            print("Exception while scrolling %s" % error)
            len_of_page = 0
        return len_of_page

    def scrape_urls(self):
        """
        Fetches article urls and updates `self.extracted_urls`
        """
        try:
            len_of_page = self.scroll_to_page_end()
            while True:
                page_height = len_of_page
                len_of_page = self.scroll_to_page_end()
                if len_of_page == page_height or len_of_page >= self.scroll_limit:
                    print("Scrolled to a page length %s" % page_height)
                    break
            while True:
                try:
                    self.extracted_urls = [elem.get_attribute('href') for elem
                                           in self.driver.find_elements_by_xpath(self.link_xpath)]
                    break
                except BadStatusLine:
                    continue
            return self.extracted_urls
        finally:
            self.driver.close()
            self.virtual_display.stop()

    @classmethod
    def scrape_urls_from_newspapers(cls, category):
        """
        Fetches article urls and updates `self.extracted_urls`
        """
        newspaper_info_set = cls.category_newspaper_url_mappings.get(category.lower(), [])
        for news_paper_url, elem_xpath in newspaper_info_set:
            try:
                paper_scraper = cls(category, url=news_paper_url)
                retry = 0
                while True:
                    try:
                        elem = paper_scraper.driver.find_element_by_xpath(elem_xpath)
                        if elem.is_displayed():
                            elem.click()
                        elif retry < 3:
                            paper_scraper.driver.find_element_by_xpath('//a[@class="more"]').click()
                            retry += 1
                            continue
                        break
                    except BadStatusLine:
                        continue
                time.sleep(3)
                paper_scraper.scrape_urls()
                paper_scraper.save_to_db()
            except Exception as error:
                print('Exception occurred while scraping %s from %s. Error: %s'
                      % (category, news_paper_url, error))

    def save_to_db(self):
        category = Category.objects.get(name=self.category)
        db_cache = []
        for url in self.extracted_urls:
            article = Article()
            article.url = url
            article.category = category
            article.source = Article.SOURCE_DH
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
    title_xpath = '//div[@class="details_data"]/h1'
    content_xpath = '//div[@class="details_data"]/div[@class="data"]/p'
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
        driver_path = settings.BASE_DIR + '/bin/geckodriver'
        self.driver = webdriver.Firefox(executable_path=driver_path)
        self.driver.get(self.url)

    def get_article_data(self):
        """
        Extracts article title and content from articles page
        """
        try:
            while True:
                try:
                    title_element = self.driver.find_element_by_xpath(self.title_xpath)
                    content_elements = self.driver.find_elements_by_xpath(self.content_xpath)
                    break
                except BadStatusLine:
                    continue
            self.title = self.get_true_text(title_element).strip()
            self.content = '\n'.join([self.get_true_text(elem) for elem in content_elements])
        finally:
            self.driver.close()
            self.virtual_display.stop()

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
                                          source=Article.SOURCE_DH, category__name="health")
        updated_articles = []
        failed_articles = []
        errors = []
        for article in articles:
            try:
                while True:
                    try:
                        scraper = cls(article.url)
                        break
                    except BadStatusLine:
                        continue
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
                print('\tError occurred for %s' % article.id)
                errors.append(article.id)
        print("Updated articles: %s" % len(updated_articles))
        print("Failed to load %s articles" % len(failed_articles))
        print("Error occurred in scraping %s articles" % len(errors))
