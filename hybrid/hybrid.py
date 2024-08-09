import pandas as pd
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

os.chdir('C:\\Users\\91778\\Downloads\\archive (7)')

data = pd.read_csv('mcd.csv')

user_preferences = ["Black Coffee", "Butter Chicken Grilled Burger", "Butter Paneer Grilled Burger"]

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['ingredients'].fillna(''))

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices = pd.Series(data.index, index=data['name'])

def content_based_recommendations(food_names, cosine_sim=cosine_sim):
    food_indices = [indices[food_name] for food_name in food_names]
    sim_scores = cosine_sim[food_indices].sum(axis=0)
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    food_indices = [i[0] for i in sim_scores]
    return data['name'].iloc[food_indices]

def collaborative_filtering_recommendations(user_preferences, data):
    return data['name']

def hybrid_recommendations(user_preferences, data):
    content_based_recos = set()
    for preference in user_preferences:
        content_based_recos.update(content_based_recommendations([preference]))
    collab_recos = set(collaborative_filtering_recommendations(user_preferences, data))
    combined_recos = list(content_based_recos.union(collab_recos))
    return combined_recos

user_recommendations = hybrid_recommendations(user_preferences, data)

print("Hybrid Recommendations:")
for item in user_recommendations:
    print(item)
