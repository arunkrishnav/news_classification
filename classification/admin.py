from django.contrib import admin
from .models import Article


class ArticleAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'is_cleaned']
    list_filter = ['category', 'status', 'iteration', 'is_cleaned']
    ordering = ['category']
    actions = ["mark_is_cleaned", "set_sports", "set_automobile", "set_technology", "set_entertainment", "set_weather",
               "set_health", "set_politics", "set_business"]
    search_fields = ['category']

    def mark_is_cleaned(self, request, queryset):
        queryset.update(is_cleaned=True)

    def set_sports(self, request, queryset):
        queryset.update(is_cleaned=True, category="sports")

    def set_automobile(self, request, queryset):
        queryset.update(is_cleaned=True, category="automobile")

    def set_technology(self, request, queryset):
        queryset.update(is_cleaned=True, category="technology")

    def set_entertainment(self, request, queryset):
        queryset.update(is_cleaned=True, category="entertainment")

    def set_weather(self, request, queryset):
        queryset.update(is_cleaned=True, category="weather")

    def set_health(self, request, queryset):
        queryset.update(is_cleaned=True, category="health")

    def set_politics(self, request, queryset):
        queryset.update(is_cleaned=True, category="politics")

    def set_business(self, request, queryset):
        queryset.update(is_cleaned=True, category="business")


admin.site.register(Article, ArticleAdmin)
