import json

from scraper.models import Article


def clean_data_with_empty_content(category=None):
    try:
        articles = Article.objects.all()
        if category:
            articles = articles.filter(category__name=category)
        for article in articles:
            if article.content is None or article.content == "":
                article.delete()
    except Exception as e:
        print(str(e))


def remove_duplicates_with_url(category=None):
    try:
        url_list = list()
        articles = Article.objects.all()
        if category:
            articles = articles.filter(category__name=category)
        for article in articles:
            if article.url in url_list:
                article.delete()
            else:
                url_list.append(article.url)
    except Exception as e:
        print(str(e))


def remove_duplicates_with_title(category=None):
    try:
        title_list = list()
        articles = Article.objects.filter(title__isnull=False)
        if category:
            articles = articles.filter(category__name=category)
        for article in articles:
            if article.title in title_list:
                article.delete()
            else:
                title_list.append(article.title)
    except Exception as e:
        print(str(e))


def export_article_data_to_json(category):
    try:
        articles = Article.objects.filter(category__name=category, title__isnull=False,
                                          content__isnull=False).order_by('-id')[:1000]
        article_list = list()
        for article in articles:
            article_dict = dict()
            article_dict["title"] = article.title
            article_dict["description"] = article.content
            article_dict["url"] = article.url
            article_dict["source"] = article.source
            article_list.append(article_dict)

        with open('data/%s.json' % category, 'w') as outfile:
            json.dump(article_list, outfile)
    except Exception as e:
        print(str(e))
