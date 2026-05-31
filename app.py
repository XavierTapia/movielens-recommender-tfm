import streamlit as st
import pandas as pd
import pickle
import numpy as np

# 1. Configuración de la página (Diseño profesional)
st.set_page_config(page_title="MovieLens Recommender", page_icon="🎬", layout="wide")

# ==========================================
# CLASE DEL MODELO Y FUNCIONES AUXILIARES
# ==========================================

def clip_rating(x, min_rating=0.5, max_rating=5.0):
    return np.clip(x, min_rating, max_rating)

def generar_estrellas(rating, max_estrellas=5):
    estrellas_llenas = int(round(rating))
    estrellas_vacias = max_estrellas - estrellas_llenas

    return "★" * estrellas_llenas + "☆" * estrellas_vacias

class MatrixFactorizationSGD:
    def __init__(self, n_factors=30, lr=0.01, reg=0.05, n_epochs=15, random_state=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.random_state = random_state
        
    def fit(self, df):
        rng = np.random.default_rng(self.random_state)
        self.global_mean_ = df["rating"].mean()
        
        self.user_ids_ = sorted(df["userId"].unique())
        self.movie_ids_ = sorted(df["movieId"].unique())
        self.user_to_idx_ = {u: i for i, u in enumerate(self.user_ids_)}
        self.movie_to_idx_ = {m: i for i, m in enumerate(self.movie_ids_)}
        
        n_users = len(self.user_ids_)
        n_movies = len(self.movie_ids_)
        
        self.P_ = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.Q_ = rng.normal(0, 0.1, size=(n_movies, self.n_factors))
        self.bu_ = np.zeros(n_users)
        self.bi_ = np.zeros(n_movies)
        
        train = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        for epoch in range(self.n_epochs):
            squared_error = 0
            for row in train.itertuples(index=False):
                u = self.user_to_idx_[row.userId]
                i = self.movie_to_idx_[row.movieId]
                r = row.rating
                
                pred = self.global_mean_ + self.bu_[u] + self.bi_[i] + np.dot(self.P_[u], self.Q_[i])
                err = r - pred
                squared_error += err ** 2
                
                p_old = self.P_[u].copy()
                q_old = self.Q_[i].copy()
                
                self.bu_[u] += self.lr * (err - self.reg * self.bu_[u])
                self.bi_[i] += self.lr * (err - self.reg * self.bi_[i])
                self.P_[u] += self.lr * (err * q_old - self.reg * p_old)
                self.Q_[i] += self.lr * (err * p_old - self.reg * q_old)

        return self

    def predict_one(self, user_id, movie_id):
        if user_id not in self.user_to_idx_ or movie_id not in self.movie_to_idx_:
            return self.global_mean_
        u = self.user_to_idx_[user_id]
        i = self.movie_to_idx_[movie_id]
        pred = self.global_mean_ + self.bu_[u] + self.bi_[i] + np.dot(self.P_[u], self.Q_[i])
        return clip_rating(pred)

    def predict(self, df):
        return np.array([self.predict_one(row.userId, row.movieId) for row in df.itertuples()])

# ==========================================

# 2. Funciones de carga en Caché (Para que la web sea ultra rápida)
@st.cache_data
def cargar_datos():
    # Asegúrate de que la ruta a tus CSV sea la correcta
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

@st.cache_resource
def cargar_modelo():
    with open('modelo_svd.pkl', 'rb') as archivo:
        modelo = pickle.load(archivo)
    return modelo

# 3. Interfaz Visual (Front-End)
st.title("🎬 Sistema de Recomendación Cinematográfica")
st.markdown("### Desarrollado con Factorización Matricial (SVD / SGD)")
st.markdown("""
<style>
.perfil-card {
    border: 1px solid #30363d;
    border-radius: 6px;
    background-color: #111827;
    margin-top: 20px;
    margin-bottom: 25px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
}

.perfil-header {
    padding: 12px 18px;
    border-bottom: 1px solid #30363d;
    font-weight: 700;
    font-size: 16px;
    color: #f9fafb;
}

.perfil-body {
    display: flex;
    padding: 18px;
}

.perfil-item {
    flex: 1;
    padding: 0 20px;
    border-right: 1px solid #30363d;
}

.perfil-item:last-child {
    border-right: none;
}

.perfil-label {
    font-size: 12px;
    color: #9ca3af;
    margin-bottom: 8px;
}

.perfil-value {
    font-size: 24px;
    font-weight: 700;
    color: #f9fafb;
}

.perfil-rating {
    display: flex;
    align-items: center;
    gap: 10px;
}

.estrellas {
    color: #ff4b4b;
    font-size: 24px;
    letter-spacing: 2px;
}

.rating-numero {
    font-size: 22px;
    font-weight: 700;
    color: #f9fafb;
}

.rating-max {
    font-size: 15px;
    color: #9ca3af;
}
</style>
""", unsafe_allow_html=True)

# Cargar datos y modelo
try:
    movies, ratings = cargar_datos()
    modelo_mf = cargar_modelo()
except Exception as e:
    st.error(f"Error cargando los archivos: {e}")
    st.stop()

# --- CÁLCULO DE ESTADÍSTICAS DE USUARIOS ---
user_stats = ratings.groupby('userId').agg(
    num_ratings=('rating', 'count'),
    avg_rating=('rating', 'mean')
).reset_index()


# ==========================================
# VALORES POR DEFECTO PARA LOS FILTROS
# ==========================================
default_min_rat = int(user_stats['num_ratings'].min())
default_max_rat = int(user_stats['num_ratings'].max())

default_min_avg = float(user_stats['avg_rating'].min())
default_max_avg = float(user_stats['avg_rating'].max())

# Inicializar estados de filtros
if "num_ratings_slider" not in st.session_state:
    st.session_state.num_ratings_slider = (default_min_rat, default_max_rat)

if "avg_rating_slider" not in st.session_state:
    st.session_state.avg_rating_slider = (default_min_avg, default_max_avg)

if "generos_multiselect" not in st.session_state:
    st.session_state.generos_multiselect = []

if "peliculas_multiselect" not in st.session_state:
    st.session_state.peliculas_multiselect = []

if "cold_start_checkbox" not in st.session_state:
    st.session_state.cold_start_checkbox = False

if "usuario_selectbox" not in st.session_state:
    st.session_state.usuario_selectbox = None


# Menú lateral con Filtros
st.sidebar.header("Filtros de Usuario")

if st.sidebar.button("🧹 Limpiar filtros"):
    st.session_state.num_ratings_slider = (default_min_rat, default_max_rat)
    st.session_state.avg_rating_slider = (default_min_avg, default_max_avg)
    st.session_state.generos_multiselect = []
    st.session_state.peliculas_multiselect = []
    st.session_state.cold_start_checkbox = False
    st.session_state.usuario_selectbox = None
    st.rerun()

# Filtro 1: Experiencia (Número de valoraciones)
min_rat, max_rat = st.sidebar.slider(
    "Nº de valoraciones:",
    min_value=default_min_rat,
    max_value=default_max_rat,
    value=st.session_state.num_ratings_slider,
    key="num_ratings_slider"
)

# Filtro 2: Exigencia (Nota media otorgada)
min_avg, max_avg = st.sidebar.slider(
    "Nota media otorgada:",
    min_value=default_min_avg,
    max_value=default_max_avg,
    value=st.session_state.avg_rating_slider,
    key="avg_rating_slider"
)

# ---  GÉNEROS Y PELÍCULAS ---

# Extraer todos los géneros únicos del DataFrame de películas
todos_generos = set()
for generos in movies['genres'].dropna().str.split('|'):
    todos_generos.update(generos)
todos_generos = sorted(list(todos_generos - {'(no genres listed)'}))

# Filtro 3: Multiselección de Géneros
generos_seleccionados = st.sidebar.multiselect(
    "Que haya visto películas de los géneros:",
    todos_generos,
    key="generos_multiselect"
)

# Filtro 4: Multiselección de Películas Vistas
lista_peliculas = sorted(movies['title'].dropna().unique())
peliculas_seleccionadas = st.sidebar.multiselect(
    "Que haya visto las siguientes películas:",
    lista_peliculas,
    key="peliculas_multiselect"
)

# Aplicar todos los filtros
# 1. Filtros numéricos
usuarios_filtrados = set(user_stats[
    (user_stats['num_ratings'] >= min_rat) & (user_stats['num_ratings'] <= max_rat) &
    (user_stats['avg_rating'] >= min_avg) & (user_stats['avg_rating'] <= max_avg)
]['userId'])

# 2. Filtros de contenido (cruzando ratings y movies)
if generos_seleccionados or peliculas_seleccionadas:
    ratings_movies = ratings.merge(movies, on='movieId')
    
    # Filtrar por géneros (debe haber visto al menos una peli de CADA género seleccionado)
    for g in generos_seleccionados:
        users_with_g = set(ratings_movies[ratings_movies['genres'].str.contains(g, regex=False)]['userId'])
        usuarios_filtrados = usuarios_filtrados.intersection(users_with_g)
        
    # Filtrar por películas (debe haber visto TODAS las películas seleccionadas)
    for p in peliculas_seleccionadas:
        users_with_p = set(ratings_movies[ratings_movies['title'] == p]['userId'])
        usuarios_filtrados = usuarios_filtrados.intersection(users_with_p)

usuarios_filtrados = sorted(list(usuarios_filtrados))

#st.sidebar.divider()

# Resultados de la búsqueda
if not usuarios_filtrados:
    st.sidebar.error("⚠️ Ningún usuario cumple con TODOS los filtros a la vez. Reduce las restricciones.")
else:
    st.sidebar.success(f"✅ {len(usuarios_filtrados)} usuarios encontrados.")
    #usuario_seleccionado = st.sidebar.selectbox("Elige un User ID filtrado:", usuarios_filtrados)

    if (
        st.session_state.usuario_selectbox is None
        or st.session_state.usuario_selectbox not in usuarios_filtrados
    ):
        st.session_state.usuario_selectbox = usuarios_filtrados[0]

    usuario_seleccionado = st.sidebar.selectbox(
        "Elige un User ID filtrado:",
        usuarios_filtrados,
        index=usuarios_filtrados.index(st.session_state.usuario_selectbox),
        key="usuario_selectbox"
    )


    # ==========================================
    #       EXPLORACIÓN DE COLD-START
    # ==========================================
    st.sidebar.divider()
    st.sidebar.write("Exploración de Cold-Start:")
    ver_no_calificadas = st.sidebar.checkbox(
        "Mostrar películas SIN calificaciones",
        key="cold_start_checkbox"
    )

    if ver_no_calificadas:
        # Lógica de filtrado
        peliculas_valoradas = ratings['movieId'].unique()
        peliculas_no_valoradas = movies[~movies['movieId'].isin(peliculas_valoradas)]
        
        # Interfaz visual en la pantalla principal
        st.subheader("❄️ Películas con Arranque en Frío (0 valoraciones)")
        st.warning("El algoritmo de Factorización Matricial (SVD/SGD) ignora estas películas,  el Filtrado Basado en Contenido (TF-IDF) es capaz de evaluarlas por sus géneros.")
        st.dataframe(peliculas_no_valoradas[['title', 'genres']])



    # --- MOSTRAR PERFIL E HISTORIAL DEL USUARIO ---
    u_data = user_stats[user_stats['userId'] == usuario_seleccionado].squeeze()

    num_valoraciones = int(u_data['num_ratings'])
    nota_media = float(u_data['avg_rating'])
    estrellas_usuario = generar_estrellas(nota_media)

    st.markdown(f"""
    <div class="perfil-card">
        <div class="perfil-header">👤 Perfil del Usuario</div>
        <div class="perfil-body">
            <div class="perfil-item">
                <div class="perfil-label">ID Usuario</div>
                <div class="perfil-value">{usuario_seleccionado}</div>
            </div>
            <div class="perfil-item">
                <div class="perfil-label">Nº de valoraciones realizadas</div>
                <div class="perfil-value">{num_valoraciones}</div>
            </div>
            <div class="perfil-item">
                <div class="perfil-label">Nota media histórica otorgada</div>
                <div class="perfil-rating">
                    <span class="estrellas">{estrellas_usuario}</span>
                    <span class="rating-numero">{nota_media:.2f}</span>
                    <span class="rating-max">/ 5</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # Historial de calificaciones interactivo
    historial = ratings[ratings['userId'] == usuario_seleccionado].merge(movies, on='movieId')
    historial = historial[['title', 'genres', 'rating']].sort_values(by='rating', ascending=False)
    
    with st.expander("Ver historial de calificaciones de este usuario"):
        st.dataframe(historial, use_container_width=True, hide_index=True)

    st.divider()
    st.write("Genera recomendaciones basadas en sus factores latentes descubiertos:")

    # 4. Motor de Recomendación
    if st.button("🚀 Generar Top-10 Personalizado", type="primary"):
        with st.spinner('Analizando factores latentes...'):
            
            # Filtrar películas que el usuario ya ha visto
            peliculas_vistas = set(ratings[ratings["userId"] == usuario_seleccionado]["movieId"])
            todas_peliculas = movies["movieId"].unique()
            peliculas_no_vistas = [m for m in todas_peliculas if m not in peliculas_vistas]
            
            # Predecir nota para las películas no vistas
            predicciones = []
            for m in peliculas_no_vistas:
                try:
                    nota_predicha = modelo_mf.predict_one(usuario_seleccionado, m)
                    predicciones.append((m, nota_predicha))
                except:
                    continue
                    
            # Función auxiliar segura para ordenar
            def extraer_nota(tupla):
                id_peli, calificacion = tupla
                return calificacion
                
            # Ordenar de mayor a menor usando la función auxiliar
            predicciones.sort(key=extraer_nota, reverse=True)
            
            # Extraer el Top-10 conservando también la predicción estimada
            top_10_predicciones = predicciones[:10]
            top_10_ids = [id_peli for id_peli, calificacion in top_10_predicciones]

            # Crear diccionario: movieId -> calificación predicha
            predicciones_dict = {
                id_peli: round(float(calificacion), 2)
                for id_peli, calificacion in top_10_predicciones
            }

            # Cruzar con el DataFrame para títulos y ordenar según el ranking
            top_10_peliculas = movies[movies['movieId'].isin(top_10_ids)].copy()
            top_10_peliculas['ID_Categorico'] = pd.Categorical(
                top_10_peliculas['movieId'],
                categories=top_10_ids,
                ordered=True
            )
            top_10_peliculas = top_10_peliculas.sort_values('ID_Categorico')

            # Agregar columna con la predicción de calificación
            top_10_peliculas['prediccion_calificacion'] = top_10_peliculas['movieId'].map(predicciones_dict)
            top_10_peliculas['estrellas_prediccion'] = top_10_peliculas['prediccion_calificacion'].apply(generar_estrellas)

            # Preparar tabla final
            tabla_top10 = top_10_peliculas[
                ['movieId', 'title', 'genres', 'prediccion_calificacion', 'estrellas_prediccion']
            ].rename(columns={
                'movieId': 'ID película',
                'title': 'Título',
                'genres': 'Géneros',
                'prediccion_calificacion': 'Predicción calificación',
                'estrellas_prediccion': 'Estrellas'
            })


            # Mostrar resultado final
            st.success(f"¡Top-10 generado con éxito para el Usuario {usuario_seleccionado}!")

            st.dataframe(
                tabla_top10,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Predicción calificación": st.column_config.NumberColumn(
                        "Predicción calificación",
                        format="%.2f"
                    )
                }
            )