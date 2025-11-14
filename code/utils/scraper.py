import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_domain(url):
    return urlparse(url).netloc.replace("www.", "").strip().lower()

from urllib.parse import urlparse

def is_company_homepage(result, threshold=60):
    url = result["url"]
    parsed = urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.strip("/").lower()
    title = result.get("title", "").lower()
    snippet = result.get("snippet", "").lower()

    # 🧱 Full block list (yours)
    block_domains = {
        "medium.com", "techtarget.com", "reddit.com", "quora.com","thegoodtrade.com","sustainablejungle.com",
        "community.openai.com", "hubspot.com", "capterra.com", "forbes.com", "springer.com", "wordpress.com",
        "dev.to", "hrvisionevent.com", "cxtoday.com", "techdogs.com", "analyticsindiamag.com", "psqh.com",
        "rupahealth.com", "appliedradiology.com", "k12dive.com", "fastercapital.com", "influencermarketinghub.com",
        "news.ycombinator.com", "profiletree.com", "mediaincanada.com", "edtechinnovationhub.com", "aithority.com",
        "theaijournal.substack.com", "scribd.com", "pmarketresearch.com", "iecc.libguides.com",
        "districtadministration.com", "mypossibilit.com", "biospace.com", "nature.com", "theguardian.com",
        "scopus.com","sciencedirect.com", "arxiv.org", "ieee.org", "ncbi.nlm.nih.gov", "huggingface.co", 
        "stackexchange.com", "stackoverflow.com", "discord.com", "venturebeat.com", "wired.com", "theverge.com", 
        "nytimes.com", "cnn.com", "bbc.com", "businessinsider.com", "hashnode.com", "substack.com", 
        "towardsdatascience.com", "forem.com","inc42.com", "erpublications.com", "journals.lww.com", "ijrpr.com", 
        "bestdigitaltoolsmentor.com","xperiencify.com","globenewswire.com", "martech360.com", 
        "lpsonline.sas.upenn.edu", "nb-data.com", "monday.com", "andrewchen.com","verywellmind.com",
        "digitalmarketinginstitute.com"
    }

    # 🚫 Direct domain match or subdomain match
    if any(domain == d or domain.endswith("." + d) for d in block_domains):
        return False

    # 🚫 Block by TLD or patterns
    if domain.endswith((".blog", ".substack.com", ".notion.site", ".github.io")):
        return False

    # ✅ Scoring logic (customizable)
    score = 100

    if any(kw in path for kw in ["blog", "news", "post", "article", "press", "review", "comparison"]):
        score -= 30
    if any(kw in title for kw in ["top", "best", "guide", "how to", "vs", "comparison", "review"]):
        score -= 15
    if any(kw in snippet for kw in ["read more", "subscribe", "published", "author", "updated"]):
        score -= 15

    if any(kw in path for kw in ["about", "contact", "company", "services", "product"]):
        score += 10
    if any(kw in snippet for kw in ["our product", "our service", "solution", "platform", "pricing"]):
        score += 10

    return score >= threshold

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