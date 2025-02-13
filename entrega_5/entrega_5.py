import json
import networkx as nx
import nltk
import matplotlib.pyplot as plt
import community  # Pacote python-louvain
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import defaultdict
from itertools import combinations

# Baixar recursos do NLTK
nltk.download('vader_lexicon')

# Inicializar analisador de sentimento
analyzer = SentimentIntensityAnalyzer()

# Carregar os dados do JSON
with open("posts_data.json", "r") as file:
    dataset = json.load(file)

# Criar o grafo
G = nx.Graph()

# Dicionário para armazenar sentimentos de cada usuário
user_sentiments = defaultdict(list)

# Processar os comentários e coletar sentimentos
for entry in dataset:
    author = entry.get("comments_auth", "Unknown")  # Autor do comentário
    comment = entry.get("comments", "")

    # Análise de sentimento do comentário inteiro
    sentiment_score = analyzer.polarity_scores(comment)['compound']

    # Normalizar o peso para evitar valores negativos
    normalized_score = sentiment_score + 1  # Agora está no intervalo [0,2]

    # Salvar sentimento do usuário
    user_sentiments[author].append(normalized_score)

# Criar nós para cada usuário
for user in user_sentiments:
    G.add_node(user)

# Criar arestas baseadas na similaridade de sentimento
for user1, user2 in combinations(user_sentiments.keys(), 2):
    scores1 = user_sentiments[user1]
    scores2 = user_sentiments[user2]

    # Calcular média dos sentimentos dos usuários
    avg_sentiment1 = sum(scores1) / len(scores1)
    avg_sentiment2 = sum(scores2) / len(scores2)

    # Criar peso baseado na proximidade de sentimento
    weight = 1 - abs(avg_sentiment1 - avg_sentiment2)  # Quanto mais próximo de 1, mais similar

    if weight > 0:  # Apenas adiciona se houver alguma similaridade
        G.add_edge(user1, user2, weight=weight)

# Detectar comunidades usando Louvain
partition = community.best_partition(G)  # Retorna um dicionário {nó: comunidade}

# Exibir as comunidades detectadas
communities = defaultdict(list)
for node, comm_id in partition.items():
    communities[comm_id].append(node)

for comm_id, members in communities.items():
    print(f"🔹 Comunidade {comm_id + 1}: {members}")

# Desenhar o grafo colorindo as comunidades
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G)
colors = [partition[node] for node in G.nodes()]

nx.draw(G, pos, with_labels=True, node_color=colors, cmap=plt.cm.Set3, edge_color="gray", font_size=8)
plt.title("Grafo de Usuários Baseado em Sentimento")
plt.show()
