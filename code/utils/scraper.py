import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_domain(url):
    return urlparse(url).netloc.replace("www.", "").strip().lower()

def is_company_homepage(result):
    from urllib.parse import urlparse
    import re

    url = result["url"]
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    path = parsed.path.strip("/").lower()
    title = result["title"].lower()
    snippet = result["snippet"].lower()

    # Reject only if it's clearly a non-company or aggregator domain
    block_domains = [
        "medium.com",  "techtarget.com", "reddit.com", "quora.com",
        "community.openai.com","hubspot.com", "capterra.com", "forbes.com", "springer.com", "wordpress.com",
        "dev.to","hrvisionevent.com","cxtoday.com","techdogs.com", "analyticsindiamag.com","psqh.com",
        "rupahealth.com","appliedradiology.com","k12dive.com","fastercapital.com","influencermarketinghub.com",
        "news.ycombinator.com","profiletree.com","mediaincanada.com","edtechinnovationhub.com","aithority.com",
        "theaijournal.substack.com","scribd.com","pmarketresearch.com","iecc.libguides.com",
        "districtadministration.com","mypossibilit.com","biospace.com","nature.com","theguardian.com","scopus.com",
        "sciencedirect.com","arxiv.org","ieee.org","ncbi.nlm.nih.gov","huggingface.co/blog","stackexchange.com", 
        "stackoverflow.com","discord.com","venturebeat.com", "wired.com", "theverge.com", "nytimes.com","cnn.com", 
        "bbc.com", "businessinsider.com","hashnode.com", "substack.com", "towardsdatascience.com","forem.com",
        "inc42.com","erpublications.com","journals.lww.com","ijrpr.com","bestdigitaltoolsmentor.com",
        "globenewswire.com","martech360.com","lpsonline.sas.upenn.edu","nb-data.com","monday.com","andrewchen.com"
        #"linkedin.com","youtube.com","facebook.com"
    ]
    if any(bad in domain for bad in block_domains):
        return False

    # Accept if it's a subpage of a .com and domain is not in blocked list
    if parsed.netloc.endswith((".com",".co",".app",".io","ai")) and parsed.scheme.startswith("http"):
        #if path.count("/") <= 2 and not any(kw in path for kw in ["blog", "article", "resources", "pulse", "news","best","top", "post"]):
            return True

    # Allow fallback: /product /services /ai /solutions in URL path
    allow_keywords = ["product", "service", "solutions", "platform", "ai"]
    if any(kw in path for kw in allow_keywords):
        return True

    return False

def clean_company_name(url):
    domain = extract_domain(url)
    return domain.split(".")[0].capitalize()

def extract_keywords(text, top_n=10):
    try:
        text = re.sub(r"[^A-Za-z0-9\s]", " ", text)
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform([text])
        scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)

        raw_keywords = [word.strip().lower() for word, _ in sorted_scores if len(word) > 2 and not word.isdigit()]
        final_keywords = []
        for kw in raw_keywords:
            if not any(kw in existing or existing in kw for existing in final_keywords):
                final_keywords.append(kw)
            if len(final_keywords) == top_n:
                break

        return ", ".join(final_keywords)
    except Exception as e:
        print(f"⚠️ Error in keyword extraction: {e}")
        return ""

def scrape_website(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        body_text = " ".join([p.get_text().strip() for p in paragraphs[:10]])

        meta_desc = ""
        meta_tag = soup.find("meta", attrs={"name": "description"})
        if meta_tag and meta_tag.get("content"):
            meta_desc = meta_tag["content"].strip()

        content = meta_desc or body_text
        keywords = extract_keywords(content)

        return {
            "url": url,
            "company_name": clean_company_name(url),
            "services": content,
            "keywords": keywords
        }

    except Exception as e:
        print(f"⚠️ Error scraping {url}: {e}")
        return None