from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):

    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    gender= models.CharField(max_length=30)
    address= models.CharField(max_length=30)


class predict_friendship_inference(models.Model):

    Fid= models.CharField(max_length=300)
    Name= models.CharField(max_length=300)
    Gender= models.CharField(max_length=300)
    DOB= models.CharField(max_length=300)
    Interests= models.CharField(max_length=300)
    City= models.CharField(max_length=300)
    Country= models.CharField(max_length=300)
    FName= models.CharField(max_length=300)
    FGender= models.CharField(max_length=300)
    FDOB= models.CharField(max_length=300)
    FCity= models.CharField(max_length=300)
    FCountry= models.CharField(max_length=300)
    Prediction= models.CharField(max_length=300)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



