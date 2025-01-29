import nltk, json, re
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

####################### Pré-processamento ##########################

nltk.download('stopwords')
nltk.download('punkt_tab')

with open("posts_data.json", "r") as file:
    dataset = json.load(file)

stop_words = set(stopwords.words('english'))

key_words = ['DonaldTrump', 'canada', 'facebook', 'instagram',
              'eua', 'sa','Elon Musk', 'joebiden','Kamala Harris',
              'Lindsey Graham','greenland','island','acquire',
              'purchase','buy','military','control','markzuckerberg',
              'zuckerberg','trump','Obama'
            ]

negative_words = [
    "corrupt", "unfair", "biased", "lies", "dishonest", "scam", "fraud", 
    "hate", "terrible", "awful", "disgusting", "nonsense", "rigged", 
    "manipulated", "fake", "wrong", "unacceptable", "shameful", "problematic", 
    "disgrace", "toxic", "incompetent", "hypocrite", "crooked", "failing", 
    "horrible", "evil", "selfish", "ruined", "destroyed", "overrated", 
    "controversial", "dangerous", "threat", "misleading", "cheated", 
    "broken", "backward", "divisive", "outdated", "pathetic", "ignorant", 
    "arrogant", "oppressive", "exploitative", "despicable", "racist","crazy",
    'fat','ugly','stupid','dumb','idiot','fool','foolish','dull','dullard',
]

def contains_keywords(text, keywords):
    pattern = "|".join(keywords)  # Cria um padrão como "game awards|nominee|winner|..."
    return bool(re.search(pattern, text, re.IGNORECASE))

filtred_words = []
filtred_words_2 = []

for entry in dataset:
    
    words_tokens_comments = word_tokenize(entry['comments'])

    for w in words_tokens_comments:
        if w.lower() not in stop_words and re.match(r'^[a-zA-Z]+$', w):
            #clean_text = ''.join(w)
            filtred_words.append(w)
    
for word in filtred_words:
    if contains_keywords(word.lower(), key_words):
       filtred_words_2.append(word)

print(filtred_words_2)

filtred_words = []

for word in filtred_words_2:
    if word.lower() in negative_words:
        filtred_words.append(word)

print("lenght",len(filtred_words))


####################### TF-IDF ##########################
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(filtred_words)

# tfidf_df = pd.DataFrame(
#     tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out()
# )