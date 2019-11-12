import csv
import os
from collections import OrderedDict

import re
from django.db import models
from django.conf import settings


class Category(models.Model):
    """
    Model to hold category info
    """
    name = models.CharField(max_length=1024)

    @classmethod
    def populate(cls, seeder_path=None):
        """
        Seeder method to populate categories
        header row of csv must be 'category'
        :param seeder_path: path to csv seeder file
        """
        if seeder_path is None:
            seeder_path = settings.BASE_DIR + '/data/categories.csv'
        if os.path.exists(seeder_path):
            with open(seeder_path, 'r') as seeder_file:
                data = csv.DictReader(seeder_file)
                count = 0
                for row in data:
                    try:
                        cls.objects.create(name=row['category'])
                        count += 1
                    except Exception as error:
                        print("Failed to create category for %s: %s" % (row['category'], error))
                print("Added %s categories" % count)
        else:
            print("Failed to seed categories: seeder file does not exists")

    def __str__(self):
        return self.name.title() if self.name else ''


class Article(models.Model):
    """
    Model to store info of articles
    """
    SOURCE_DH = 'dailyhunt'
    SOURCE_SL = 'southlive'
    SOURCE_OI = 'oneindia'
    SOURCE_MM = 'manorama'
    SOURCE_IT1 = 'iteration_1'
    SOURCE_IT2 = 'iteration_2'
    SOURCE_IT3 = 'iteration_3'
    SOURCE_CHOICES = (
        (SOURCE_DH, 'DailyHunt'),
        (SOURCE_SL, 'SouthLive'),
        (SOURCE_OI, 'OneIndia'),
        (SOURCE_MM, 'Manorama'),
        (SOURCE_IT1, 'Iteration 1'),
        (SOURCE_IT2, 'Iteration 2'),
        (SOURCE_IT3, 'Iteration 3'),
    )

    category = models.ForeignKey('Category', on_delete=models.CASCADE)
    title = models.TextField(null=True, blank=True)
    url = models.URLField(null=True, blank=True, max_length=2048)
    content = models.TextField(null=True, blank=True)
    is_cleaned = models.BooleanField(default=False)
    source = models.CharField(max_length=2048, choices=SOURCE_CHOICES)
    is_mapped = models.BooleanField(default=False)

    def __str__(self):
        return self.title if self.title else ''

    def get_data_dict(self):
        """
        Returns an ordered dict of data to write in csv
        """
        return OrderedDict([
            ('category', self.category.name),
            ('text', re.sub('[\n\r]', ' ', self.title + ' ' + self.content)),
        ])

    @classmethod
    def export_to_csv(cls, categories=list(), limit=500, cleaned_only=False):
        """
        Method to export existing articles to a csv
        :param categories: list of categories required
        :param limit: count limit per category. If minimum count for any category
         is less than this, will generate csv with that minimum count for all categories
        :param cleaned_only: Decides whether the csv should contain only cleaned values
        """
        out_file_path = settings.BASE_DIR + "/data/demo_train_data.csv"
        query_sets = []

        if not categories:
            categories = ['sports', 'automobile', 'technology', 'entertainment',
                          'weather', 'health', 'politics', 'business']

        for category in categories:
            query_set = cls.objects.filter(category__name=category, title__isnull=False, content__isnull=False)\
                    .exclude(content='')
            if cleaned_only:
                query_set.filter(is_cleaned=True)
            query_sets.append(query_set)

        with open(out_file_path, 'w', encoding='utf-8') as csvfile:
            field_names = list(cls.objects.filter(title__isnull=False, content__isnull=False)
                               .last().get_data_dict().keys())
            writer = csv.DictWriter(csvfile, fieldnames=field_names)
            writer.writeheader()

            for query_set in query_sets:
                for item in query_set:
                    writer.writerow(item.get_data_dict())
        print('Written data to %s' % out_file_path)
