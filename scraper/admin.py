from django.contrib import admin
from .models import Article, Category
# Register your models here.


class ArticleAdmin(admin.ModelAdmin):
    list_display = ['title', 'category', 'is_cleaned', 'source', 'is_mapped']
    list_filter = ['category', 'is_cleaned', 'source', 'is_mapped']
    ordering = ['category']
    actions = ["mark_is_cleaned", "mark_is_mapped"]
    search_fields = ['url']

    def mark_is_cleaned(self, request, queryset):
        queryset.update(is_cleaned=True)

    def mark_is_mapped(self, request, queryset):
        queryset.update(is_mapped=True)


admin.site.register(Article, ArticleAdmin)
admin.site.register(Category)
