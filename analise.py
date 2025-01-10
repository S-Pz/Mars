import re
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import silhouette_score

# === 1. Leitura dos Dados ===
with open("posts_data2.json", "r") as infile:
    dataset = json.load(infile)

print(f"Total de registros carregados: {len(dataset)}")

# === 2. Definição de palavras-chave ===
keywords = [
    "game awards", "nominee", "winner", "goty", 
    "best game", "awards ceremony", "TGA", 
    "best game award", "best game awards", "best game award ceremony",
    "astro bot", "Nicolas Doucet", "Black Myth Wukong", "Elden Ring"
]

# === 3. Filtro dos dados ===
def contains_keywords(text, keywords):
    pattern = "|".join(keywords)
    return bool(re.search(pattern, text, re.IGNORECASE))

filtered_data = []
for entry in dataset:
    if contains_keywords(entry.get("title", ""), keywords) or contains_keywords(entry.get("comments", ""), keywords):
        filtered_data.append(entry)

print(f"Total de registros filtrados: {len(filtered_data)}")

# Salvando os dados filtrados
with open("game_awards_filtered.json", "w") as outfile:
    json.dump(filtered_data, outfile, indent=4)

print("Dados filtrados salvos em 'game_awards_filtered.json'")

# === 4. Conversão para DataFrame ===
filtered_df = pd.DataFrame(filtered_data)
filtered_df.fillna("", inplace=True)

def word_count(text):
    return len(text.split()) if isinstance(text, str) else 0

filtered_df["post_word_count"] = filtered_df["title"].apply(word_count)
filtered_df["comment_word_count"] = filtered_df["comments"].apply(word_count)

# Estatísticas básicas
print("\n=== Estatísticas Básicas ===")
print(filtered_df.describe(include="all"))

# Visualização: Histogramas
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(filtered_df["post_word_count"], bins=20, color='blue', alpha=0.7)
plt.title("Distribuição de Palavras nos Títulos")
plt.xlabel("Número de Palavras")
plt.ylabel("Frequência")

plt.subplot(1, 2, 2)
plt.hist(filtered_df["comment_word_count"], bins=20, color='green', alpha=0.7)
plt.title("Distribuição de Palavras nos Comentários")
plt.xlabel("Número de Palavras")
plt.ylabel("Frequência")

plt.tight_layout()
plt.show()

# === 5. Clusterização com TF-IDF ===
filtered_df["combined_text"] = filtered_df["title"] + " " + filtered_df["comments"]

vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
tfidf_matrix = vectorizer.fit_transform(filtered_df["combined_text"])

# Reduzindo dimensionalidade (PCA)
pca = PCA(n_components=2, whiten=True)
reduced_data = pca.fit_transform(tfidf_matrix.toarray())

# Encontrar o melhor número de clusters
print("\n=== Avaliação de Clusters ===")
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(reduced_data)
    score = silhouette_score(reduced_data, clusters)
    print(f"Clusters: {k}, Silhouette Score: {score}")

# Escolher o número ótimo de clusters
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(reduced_data)
filtered_df["cluster"] = clusters

# Visualização dos clusters
plt.figure(figsize=(8, 6))
for cluster_id in range(num_clusters):
    cluster_points = reduced_data[clusters == cluster_id]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {cluster_id}")

plt.legend()
plt.title("Clusterização de Postagens")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.show()

# === 6. Exploração de Palavras por Cluster ===
additional_stop_words = {"a", "the", "is", "to", "in", "and", "of", "for", "on", "at", "with"}
cluster_texts = filtered_df.groupby("cluster")["combined_text"].apply(lambda texts: " ".join(texts))

for cluster_id, texts in cluster_texts.items():
    words = [word for word in re.findall(r"\b\w+\b", texts.lower()) if word not in additional_stop_words]
    word_counts = Counter(words)
    print(f"\nPalavras mais frequentes no Cluster {cluster_id}:")
    print(word_counts.most_common(10))

# === 7. Correlação de Contagem de Palavras ===
correlation = filtered_df[["post_word_count", "comment_word_count"]].corr()
print("\n=== Correlação entre Atributos ===")
print(correlation)

# Visualização: Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(
    filtered_df["post_word_count"], 
    filtered_df["comment_word_count"], 
    alpha=0.5, 
    color='purple'
)
plt.title("Relação entre Palavras no Título e Comentário")
plt.xlabel("Número de Palavras no Título")
plt.ylabel("Número de Palavras no Comentário")
plt.show()