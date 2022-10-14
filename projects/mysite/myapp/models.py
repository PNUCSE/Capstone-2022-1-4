from django.db import models

# Create your models here.

class Querytext(models.Model):
    content = models.TextField()

    def __str__(self):
        return self.content


class Answertext(models.Model):
    querytext = models.ForeignKey(Querytext, on_delete=models.CASCADE)
    content = models.TextField()

    def __str__(self):
        return self.content