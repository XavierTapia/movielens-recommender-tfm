# MovieLens Recommender TFM

Este repositorio contiene el código fuente desarrollado para el Trabajo de Fin de Máster titulado **Machine Learning aplicado a sistemas de recomendación cinematográfica**.

## Contenido del repositorio

- `movielens_recommender_analysis.ipynb`: notebook principal de análisis exploratorio, modelado, evaluación y comparación de modelos.
- `app.py`: aplicación web interactiva local desarrollada con Streamlit.
- `modelo_svd.pkl`: modelo entrenado utilizado por la aplicación.
- `data/movies.csv`: catálogo de películas del dataset MovieLens.
- `data/ratings.csv`: histórico de valoraciones del dataset MovieLens.
- `requirements.txt`: dependencias necesarias para ejecutar el proyecto.

## Modelos implementados

El proyecto compara cinco enfoques de recomendación:

1. Modelo base basado en promedios.
2. Filtrado colaborativo item-item con kNN.
3. Filtrado basado en contenido mediante TF-IDF aplicado a géneros.
4. Factorización matricial optimizada mediante SGD.
5. Sistema híbrido ponderado.

## Ejecución de la aplicación

Instalar dependencias:

```bash
pip install -r requirements.txt