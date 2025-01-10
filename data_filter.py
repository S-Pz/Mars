import re
import json

import pandas as pd
import matplotlib.pyplot as plt

# === 1. Leitura dos Dados ===
# Supondo que os dados brutos estejam em um arquivo JSON "reddit_data.json"
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
    pattern = "|".join(keywords)  # Cria um padrão como "game awards|nominee|winner|..."
    return bool(re.search(pattern, text, re.IGNORECASE))

filtered_data = []

for entry in dataset:
    if contains_keywords(entry["title"], keywords) or contains_keywords(entry["comments"], keywords):
        filtered_data.append(entry)

print(f"Total de registros filtrados: {len(filtered_data)}")

# Salvando os dados filtrados
with open("game_awards_filtered.json", "w") as outfile:
    json.dump(filtered_data, outfile, indent=4)

print("Dados filtrados salvos em 'game_awards_filtered.json'")

# === 4. Análise Inicial ===
# Convertendo para DataFrame para facilitar a análise
filtered_df = pd.DataFrame(filtered_data)

# Estatísticas básicas
print("\n=== Estatísticas Básicas ===")
print(filtered_df.describe(include="all"))

# Contagem de palavras no título e comentário
def word_count(text):
    return len(text.split()) if isinstance(text, str) else 0

filtered_df["post_word_count"] = filtered_df["title"].apply(word_count)
filtered_df["comment_word_count"] = filtered_df["comments"].apply(word_count)

print("\n=== Estatísticas de Contagem de Palavras ===")
print(filtered_df[["post_word_count", "comment_word_count"]].describe())

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

# Visualização: Box-Plots
plt.figure(figsize=(10, 5))
plt.boxplot([filtered_df["post_word_count"], filtered_df["comment_word_count"]], labels=["Títulos", "Comentários"])
plt.title("Variabilidade de Contagem de Palavras")
plt.ylabel("Número de Palavras")
plt.show()

# === 5. Exploração de Correlações ===
correlation = filtered_df[["post_word_count", "comment_word_count"]].corr()
print("\n=== Correlação entre Atributos ===")
print(correlation)

# Visualização: Scatter Plot
plt.figure(figsize=(8, 6))
plt.scatter(filtered_df["post_word_count"], filtered_df["comment_word_count"], alpha=0.5, color='purple')
plt.title("Relação entre Palavras no Título e Comentário")
plt.xlabel("Número de Palavras no Título")
plt.ylabel("Número de Palavras no Comentário")
plt.show()
