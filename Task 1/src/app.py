from flask import Flask, request, render_template
import pandas as pd

app = Flask(__name__)

# Load your dataset and similarity matrix
new = pd.read_csv('../Input_files/tmdb_6000_movies.csv')
similarity = pd.read_pickle('../Output_files/similarity.pkl')
movie_list = new['title'].values


def recommend(movie):
    index = new[new['title'].str.lower() == movie.lower()].index
    if index.empty:
        return []

    distances = sorted(list(enumerate(similarity[index[0]])), reverse=True, key=lambda x: x[1])
    recommended_movies = [new.iloc[i[0]].title for i in distances[1:6]]
    return recommended_movies


@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = None
    selected_movie = None
    if request.method == 'POST':
        selected_movie = request.form['movie']
        recommendations = recommend(selected_movie)
    return render_template('index.html', movie_list=movie_list, recommendations=recommendations,
                           selected_movie=selected_movie)


if __name__ == '__main__':
    app.run(debug=True)
