# Movie recommender

Movie recommender using MovieLens dataset, NMF algorithm and Cosine similarity.

**Dataset**

<!-- Link: https://files.grouplens.org/datasets/movielens/ml-32m.zip -->

```powershell
Invoke-WebRequest -Uri "https://files.grouplens.org/datasets/movielens/ml-32m.zip" -OutFile "ml-32m.zip"
```

```powershell
Expand-Archive -Path ".\ml-32m.zip" -DestinationPath .
```
