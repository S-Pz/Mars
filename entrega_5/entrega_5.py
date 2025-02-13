import nltk, json, re
import networkx as nx
import community as community_louvain  
import matplotlib.pyplot as plt

from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
from collections import defaultdict

###################################Leitura de arquivo#####################################
with open("posts_data.json", "r") as file:
    dataset = json.load(file)

################################Definição palavras chaves e palavras negativas############
key_words = [ 'canada', 'facebook', 'instagram','eua', 'usa','Elon Musk', 'joebiden','Kamala Harris',
              'Lindsey Graham','greenland','island','acquire','purchase','buy','military','control',
              'markzuckerberg','zuckerberg','trump','Obama',"corrupt","biased", "lies","racist","fake"
            ]

negative_words = [
    "corrupt", "unfair", "biased", "lies", "dishonest", "scam", "fraud","hate", "terrible", "awful",
    "disgusting", "nonsense", "rigged","manipulated", "fake", "wrong", "unacceptable", "shameful", 
    "problematic","disgrace", "toxic", "incompetent", "hypocrite", "crooked", "failing","horrible",
    "evil","selfish", "ruined", "destroyed", "overrated", "controversial", "dangerous", "threat",
    "misleading", "cheated","broken", "backward", "divisive", "outdated", "pathetic", "ignorant", 
    "arrogant", "oppressive", "exploitative", "despicable", "racist","crazy",'fat','ugly','stupid',
    'dumb','idiot','fool','foolish','dull','dullard'
]
####################################Pré-processamento#####################################################
nltk.download('punkt_tab')
nltk.download('vader_lexicon')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

author_topic_map = {}
filtred_sentences = []

for entry in dataset:
    author = entry['comments_auth']

    words_tokens_comments = word_tokenize(entry['comments'])
    
    filtered_words = [
        w.lower() for w in words_tokens_comments 
        if w.lower() not in stop_words and re.match(r'^[a-zA-Z]+$', w)
    ]
    
    related_keywords = [word for word in filtered_words if word in key_words]

    if filtered_words: 
        filtred_sentences.append(filtered_words)
    
    if related_keywords:
        if author not in author_topic_map:
            author_topic_map[author] = set()
        author_topic_map[author].update(related_keywords)

print(author_topic_map)

#################################### Word2Vec e análise de sentimento #####################################################
analyzer = SentimentIntensityAnalyzer()
all_similar_words = [] 

for i, filtred_list in enumerate(filtred_sentences):

    model = Word2Vec([filtred_list], vector_size=100, window=10, min_count=1, workers=4, sg=1)  # Treina um modelo para cada conjunto de frases
    valid_key_words = [word for word in key_words if word in model.wv]
    valid_negative_words = [word for word in negative_words if word in model.wv]

    if valid_key_words:
        try:
            similar_words = model.wv.most_similar(positive=valid_key_words, negative=valid_negative_words)
            relation = [(item[0], round(item[1], 2)) for item in similar_words]
            all_similar_words.extend(relation)  # Adiciona os resultados à lista geral

            print(f"\n🔹Lista {i + 1} - Palavras mais associadas:")
            for word, score in relation:
                print(f"{word}: {score}")

        except KeyError:
            print(f"\n ⚠️ Lista {i + 1}: Não há palavras suficientes para calcular similaridade.")
            continue  # Caso não tenha palavras suficientes, ignora e segue para a próxima lista

# Remove palavras duplicadas
word_scores = defaultdict(list)
for word, score in all_similar_words:
    word_scores[word].append(score)

# Calcula a média da pontuação de similaridade para cada palavra
aggregated_scores = [(word, round(sum(scores) / len(scores), 2)) for word, scores in word_scores.items()]
aggregated_scores.sort(key=lambda x: x[1], reverse=True)  # Ordena do mais similar para o menos similar

##################################### Exibe os resultados finais agregados#################################
print("\n🔹 Palavras mais associadas ao conjunto completo:")
for word, score in aggregated_scores:
    print(f"{word}: {score}")

    related_sentences = [entry['comments'] for entry in dataset if word in entry['comments']]
    combined_text = " ".join(related_sentences)
    sentiment_score = analyzer.polarity_scores(combined_text)['compound']
    sentiment = "Positivo" if sentiment_score > 0 else "Negativo" if sentiment_score < 0 else "Neutro"

    print(f"Palavra: {word} - Score: {sentiment_score} ({sentiment})")

#################Dicionário a partir dos comentários filtrados e geração de tópicos#################
dictionary = Dictionary(filtred_sentences)
corpus = [dictionary.doc2bow(text) for text in filtred_sentences]

# Treinando o modelo LDA para encontrar 5 tópicos
lda = LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)

# Mostrando os tópicos identificados no terminal
topics = lda.print_topics(num_words=5)
for topic in topics:
    print(topic)

###################################Grafo de autores e tópicos########################################
G = nx.Graph()

for author in author_topic_map.keys():
    G.add_node(author)

authors = list(author_topic_map.keys())

for i in range(len(authors)):
    for j in range(i + 1, len(authors)):
        common_topics = author_topic_map[authors[i]].intersection(author_topic_map[authors[j]])
        if common_topics:  # Se há interseção de tópicos, cria conexão
            G.add_edge(authors[i], authors[j], weight=len(common_topics))
            
# Detecção das comunidades no grafo
partition = community_louvain.best_partition(G)

# Identifica a maior comunidade
community_sizes = {}

for node, comm_id in partition.items():
    community_sizes[comm_id] = community_sizes.get(comm_id, 0) + 1

# Obtém a maior comunidade
largest_community_id = max(community_sizes, key=community_sizes.get)
largest_community = [node for node, comm_id in partition.items() if comm_id == largest_community_id]

# Ajuste do tamanho dos nós com base na comunidade
node_sizes = [800 if node in largest_community else 400 for node in G.nodes()]

# Layout para melhor visualização
pos = nx.spring_layout(G, k=0.5)

# Colorindo os nós por comunidade
node_colors = [partition[node] for node in G.nodes()]

plt.figure(figsize=(12, 8))

# Arestas e cores com base no peso
edges = G.edges(data=True)
edge_colors = [data['weight'] for _, _, data in edges]

nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors, 
        width=2, node_size=node_sizes, font_size=10, cmap=plt.cm.rainbow, edge_cmap=plt.cm.Blues)

# Exibe os pesos das conexões
edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

# Título
plt.title("🔗 Rede de Autores Baseada em Tópicos com Comunidades")
plt.show()

# Plotando comunidades individualmente
for comm_id in set(partition.values()):
    community_nodes = [node for node, comm in partition.items() if comm == comm_id]
    subgraph = G.subgraph(community_nodes)
    plt.figure(figsize=(8, 6))
    pos_sub = nx.spring_layout(subgraph, k=0.5)
    nx.draw(subgraph, pos_sub, with_labels=True, node_size=600, font_size=8, node_color='lightblue', edge_color='grey')
    plt.title(f"Comunicade {comm_id}")
    plt.show()
