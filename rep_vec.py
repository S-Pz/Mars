import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import LatentDirichletAllocation

#nltk para tfidf 
# usar as funcionlidades da lib
with open("game_awards_filtered.json", "r") as file:
    data = json.load(file)

# Combine "title" and "comments" for vectorization
texts = [f"{entry['title']} {entry['comments']}" for entry in data]

# Vectorize the data using TF-IDF
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(texts)

# Convert to DataFrame for better visualization
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()
)

# Lista de palavras negativas relacionadas a opiniões sobre o Game Awards 2024
negative_words = ["unfair", "rigged", "doesn't make sense", "bad game", "terrible", "nonsense", "cheated", "wrong"]

# Atualizar a matriz TF-IDF com peso 1 para palavras negativas
def update_negative_weights(tfidf_df, texts, negative_words):
    for i, text in enumerate(texts):
        for word in negative_words:
            if word in text.lower():
                if word in tfidf_df.columns:
                    tfidf_df.loc[i, word] = 1  # Definir peso 1 para palavras negativas encontradas
                else:
                    # Caso a palavra não exista como coluna, adicioná-la
                    tfidf_df[word] = 0
                    tfidf_df.loc[i, word] = 1
    return tfidf_df

# Atualizar a matriz
tfidf_negative_updated = update_negative_weights(tfidf_df, texts, negative_words)

# Verificar resultado
# tfidf_negative_updated.head()

tfidf_negative_updated.to_csv("tfidf_negative_updated.csv", index=False)

# Perform LDA on TF-IDF matrix to identify topics
lda = LatentDirichletAllocation(n_components=3, random_state=42)  # 3 topics as example
lda.fit(tfidf_matrix)

# Extract topics and their representative words
def get_topics(lda_model, feature_names, n_top_words=5):
    topics = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        topics[f"Topic {topic_idx+1}"] = top_features
    return topics

# Get feature names from the TF-IDF vectorizer
tfidf_feature_names = vectorizer.get_feature_names_out()

# Get the topics
topics = get_topics(lda, tfidf_feature_names)

# Return the extracted topics
print(topics)
