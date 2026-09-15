import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sklearn.preprocessing import normalize

app = FastAPI(title="MovieLens Recommender")

# 1. Rutas de archivos
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "nmf_H_n50_iter500.pkl")
TITLES_PATH = os.path.join(BASE_DIR, "data", "movie_titles.csv")

# 2. Carga inicial de datos
print("Cargando la matriz H...")
H = joblib.load(MODEL_PATH)

# Transponer y normalizar vectores para cálculo rápido
movie_features = H.T
movie_features_norm = normalize(movie_features, axis=1)

print("Cargando catálogo de películas...")
titles_df = pd.read_csv(TITLES_PATH)
titles_list = titles_df["titles"].dropna().astype(str).tolist()

# Mapeos de búsqueda rápida
title_to_idx = {title: i for i, title in enumerate(titles_list)}
idx_to_title = {i: title for i, title in enumerate(titles_list)}

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# FIX 1: Desactivar la caché de plantillas de Jinja2 para evitar incompatibilidades en Python 3.14
templates.env.cache = None


# 3. Ruta principal (HTML)
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # FIX 2: Usar la nueva firma de TemplateResponse
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"available_titles": titles_list[:2000]},
    )


# 4. Endpoint de la API de recomendaciones
@app.get("/api/recommend")
def recommend(
    title: str = Query(..., description="Título de la película"), top_n: int = 5
):
    target_title = None

    # Búsqueda exacta primero
    if title in title_to_idx:
        target_title = title
    else:
        # Búsqueda parcial (insensible a mayúsculas)
        matches = [t for t in titles_list if title.lower() in t.lower()]
        if matches:
            target_title = matches[0]
        else:
            raise HTTPException(
                status_code=404,
                detail=f"No se encontró la película '{title}' en el catálogo filtrado.",
            )

    idx = title_to_idx[target_title]
    target_vector = movie_features_norm[idx]

    # Similitud de coseno mediante producto escalar
    similarities = movie_features_norm.dot(target_vector)
    top_indices = np.argsort(similarities)[::-1]

    recommendations = []
    for i in top_indices:
        if i == idx:
            continue  # Omitir la propia película buscada

        score = float(similarities[i])
        recommendations.append(
            {
                "title": idx_to_title[i],
                "similarity": round(score, 4),
                "percentage": f"{round(score * 100, 1)}%",
            }
        )

        if len(recommendations) == top_n:
            break

    return {"query": target_title, "recommendations": recommendations}
