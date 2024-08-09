import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.chdir('C:\\Users\\91778\\Downloads\\archive (4)')

data = pd.read_csv('swiggy_chennai_data.csv')

data.fillna({'index': 'Unknown', 'city': 'Chennai', 'subcity': 'Ponamallee', 'rating': 4.1,  'rating count': '100+ ratings', 'cost': 320, 'cuisine': 'Chicken Big Bucket Biriyani 9 Members', 'menu': 'Unknown', 'item': 'Unknown', 'price': 0, 'veg_or_non_veg': 'Unknown'}, inplace=True)

data['text_features'] = data['index'].astype(str) + '  ' + data['city'] + '  ' + data['subcity'] + '  ' + data['rating'].astype(str) + '  ' + data['rating count'] + '  ' + data['cuisine'] + '  ' + data['menu'] + '  ' + data['item'] + data['price'].astype(str) + '  ' + data['veg_or_non_veg']

train, test = train_test_split(data, test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(train['text_features'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

pivot_table = train.pivot_table(index='restaurant', columns='subcity', values='rating count')  # Adjust as needed
pivot_table.fillna(0, inplace=True)
collab_sim = cosine_similarity(pivot_table.T, pivot_table.T)

hybrid_recommendations = pd.DataFrame(index=train['restaurant'], columns=['content_score', 'collab_score'])

for i, restaurant in enumerate(train['restaurant']):
    content_scores = content_sim[i]
    collab_scores = collab_sim[i]
    hybrid_recommendations.loc[restaurant, 'content_score'] = content_scores.mean()
    hybrid_recommendations.loc[restaurant, 'collab_score'] = collab_scores.mean()

train['veg_or_non_veg'] = (train['veg_or_non_veg'].str.strip() == 'Non-veg').astype(int)

clf = RandomForestClassifier()
clf.fit(hybrid_recommendations, train['veg_or_non_veg'])

test_recommendations = pd.DataFrame(index=test['restaurant'], columns=['content_score', 'collab_score'])

for i, restaurant in enumerate(test['restaurant']):
    content_scores = content_sim[i]
    collab_scores = collab_sim[i]
    test_recommendations.loc[restaurant, 'content_score'] = content_scores.mean()
    test_recommendations.loc[restaurant, 'collab_score'] = collab_scores.mean()

predictions = clf.predict(test_recommendations)

accuracy = accuracy_score(test['veg_or_non_veg'], predictions)
print("Accuracy:", accuracy)
