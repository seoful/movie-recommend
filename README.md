# Movie Reccomendation System
>Agafonov Alexander B21-DS-01

>a.agafonov@innopolis.university

## Info

This is the movie recommendation system based on [LightFM](https://making.lyst.com/lightfm/docs/home.html) model. It incorporates both rating history of the users as well as the users and movies metadata.

## Usage



To run the benchmark use the following command:

```python benchmark/evaluate.py -k <K for Precision@k and Recall@k metrics>```

To show recommendations for a user by ID run the following:

```python src/model_predict.py <USER_ID> -k <Number of recommendations>```