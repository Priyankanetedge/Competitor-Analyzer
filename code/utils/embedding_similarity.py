from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords(text, top_n=5):
    vec = CountVectorizer(stop_words='english').fit([text])
    bag = vec.transform([text])
    sum_words = bag.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    keywords = [w for w, _ in words_freq[:top_n]]
    return ", ".join(keywords)

def rank_companies(user_description, user_keywords, scraped_data, top_n=5):
    user_text = user_description + " " + " ".join(user_keywords)
    user_embedding = model.encode([user_text])
    scores = []
    for item in scraped_data:
        company_text = item["services"] + " " + item.get("keywords", "")
        company_embedding = model.encode([company_text])
        score = cosine_similarity(user_embedding, company_embedding)[0][0]
        scores.append({
            "company_name": item["company_name"],
            "url": item["url"],
            "services": item["services"],
            "keywords": item["keywords"],
            "score": score
        })

    return sorted(scores, key=lambda x: x["score"], reverse=True)[:top_n]