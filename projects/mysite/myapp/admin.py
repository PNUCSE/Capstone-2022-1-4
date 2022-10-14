from django.contrib import admin
from .models import Querytext, Answertext

# Register your models here.

class QuerytextAdmin(admin.ModelAdmin):
    search_fields = ['content']

class AnswertextAdmin(admin.ModelAdmin):
    search_fields = ['content']

admin.site.register(Querytext, QuerytextAdmin)
admin.site.register(Answertext, AnswertextAdmin)