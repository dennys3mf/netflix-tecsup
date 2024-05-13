import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import bs4 as bs
import urllib.request
import pickle
import requests
import redis

# Cargue el modelo NLP y el vectorizador TFIDF desde el disco
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

# Crear una conexión a Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def create_similarity():
    # Verificar si los datos y la similitud ya están en la caché
    data = r.get('data')
    similarity = r.get('similarity')
    if data and similarity:
        # Si están en la caché, cargarlos
        data = pickle.loads(data)
        similarity = pickle.loads(similarity)
    else:
        # Si no están en la caché, calcularlos y almacenarlos en la caché
        data = pd.read_csv('main_data.csv')
        cv = CountVectorizer()
        count_matrix = cv.fit_transform(data['comb'])
        similarity = cosine_similarity(count_matrix)
        r.set('data', pickle.dumps(data))
        r.set('similarity', pickle.dumps(similarity))
    return data, similarity

def rcmd(m):
    m = m.lower()
    # Buscar la película en Redis
    recommendations = r.get(m)
    if recommendations:
        # Si la película se encuentra en Redis, devolver el resultado
        return pickle.loads(recommendations)
    else:
        # Si la película no se encuentra en Redis, calcular la similitud del coseno
        try:
            data.head()
            similarity.shape
        except:
            data, similarity = create_similarity()
        if m not in data['movie_title'].unique():
            return('¡No se encontro! Pruebe con otro nombre de película')
        else:
            i = data.loc[data['movie_title']==m].index[0]
            lst = list(enumerate(similarity[i]))
            lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
            lst = lst[1:11] # excluyendo el primer elemento ya que es la película solicitada en sí misma
            l = []
            for i in range(len(lst)):
                a = lst[i][0]
                l.append(data['movie_title'][a])
            # Guardar el resultado de la similitud del coseno en Redis
            r.set(m, pickle.dumps(l))
            return l
    
# Conversión de una lista de cadenas a una lista (eg. "["abc","def"]" a ["abc","def"])

def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list


# para obtener sugerencias de películas
def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

# Flask API

app = Flask(__name__)

@app.route("/")
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # obtención de datos de la solicitud AJAX
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # Obtener sugerencias de películas para Autocompletar
    suggestions = get_suggestions()

    # Llame a la función convert_to_list para cada cadena que deba convertirse en lista
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # Convertir cadena en lista (p. ej. "[1,2,3]" a [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # Representación de la cadena en una cadena de Python
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # Combinar varias listas como un diccionario que se puede pasar al archivo HTML para que se pueda procesar fácilmente y se conserve el orden de la información
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}

    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping para obtener reseñas de usuarios del sitio IMDB
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # Lista de reseñas
    reviews_status = [] # Lista de comentarios (buenos o malos)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # Pasando la revisión a nuestro modelo
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # Combinar reseñas y comentarios en un diccionario
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     

    # Pasar todos los datos al archivo HTML
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(debug=True)