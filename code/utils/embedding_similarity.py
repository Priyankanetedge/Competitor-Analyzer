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
    #for comp in sorted(scores, key=lambda x: x["score"], reverse=True):
        #print(f"{comp['company_name']} → {comp['score']:.4f}")

    return sorted(scores, key=lambda x: x["score"], reverse=True)[:top_n]

'''from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_keywords(text, top_n=5):
    try:
        vec = CountVectorizer(stop_words='english').fit([text])
        bag = vec.transform([text])
        sum_words = bag.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        keywords = [w for w, f in words_freq[:top_n]]
        return ", ".join(keywords)
    except:
        return ""
    
# Deduplicate based on domain name (normalized)
def get_domain(url):
    from urllib.parse import urlparse
    return urlparse(url).netloc.replace("www.", "").split('/')[0].lower()


def rank_companies(user_description, user_keywords, scraped_data, top_n=5):
    user_input_text = user_description + " " + " ".join(user_keywords)
    user_embedding = model.encode([user_input_text])

    company_scores = []

    for item in scraped_data:
        services_text = item['services']
        keywords_text = item['keywords']

        if not keywords_text:
            keywords_text = extract_keywords(services_text)

        company_text = services_text + " " + keywords_text
        company_embedding = model.encode([company_text])
        score = cosine_similarity(user_embedding, company_embedding)[0][0]

        company_scores.append({
            "company_name": item["company_name"],
            "url": item["url"],
            "services": services_text,
            "keywords": keywords_text,
            "score": score
        })

    ranked = sorted(company_scores, key=lambda x: x['score'], reverse=True)
    return ranked[:top_n]
''''''
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from urllib.parse import urlparse

model = SentenceTransformer('all-MiniLM-L6-v2')


def extract_keywords(text, top_n=5):
    try:
        vec = CountVectorizer(stop_words='english').fit([text])
        bag = vec.transform([text])
        sum_words = bag.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        keywords = [w for w, f in words_freq[:top_n]]
        return ", ".join(keywords)
    except:
        return ""


def get_domain(url):
    parsed = urlparse(url)
    return parsed.netloc.replace("www.", "").lower()


def rank_companies(user_description, user_keywords, scraped_data, top_n=5):
    user_input_text = user_description + " " + " ".join(user_keywords)
    user_embedding = model.encode([user_input_text])

    company_scores = []

    for item in scraped_data:
        services_text = item.get('services', '')
        keywords_text = item.get('keywords', '')

        if not keywords_text:
            keywords_text = extract_keywords(services_text)

        company_text = services_text + " " + keywords_text
        company_embedding = model.encode([company_text])
        score = cosine_similarity(user_embedding, company_embedding)[0][0]

        company_scores.append({
            "company_name": item.get("company_name", ""),
            "url": item.get("url", ""),
            "services": services_text,
            "keywords": keywords_text,
            "score": score,
            "domain": get_domain(item.get("url", ""))
        })

    # Sort by similarity
    ranked = sorted(company_scores, key=lambda x: x['score'], reverse=True)

    # ✅ Deduplicate by domain
    unique_results = []
    seen_domains = set()
    for item in ranked:
        if item['domain'] not in seen_domains:
            unique_results.append(item)
            seen_domains.add(item['domain'])
        if len(unique_results) >= top_n:
            break

    return unique_results
'''