import streamlit as st
from utils.web_search import search_competitor_urls
from utils.scraper import scrape_website,is_company_homepage
from utils.embedding_similarity import rank_companies
import pandas as pd
import time

def get_enough_companies(description, keywords, min_required=5, max_attempts=10):
    from utils.web_search import search_competitor_urls
    from utils.scraper import is_company_homepage
    import time

    seen_urls = set()
    valid_results = []

    for attempt in range(max_attempts):
        results = search_competitor_urls(description, keywords, max_results=10)
        for r in results:
            url = r["url"]
            if url not in seen_urls and is_company_homepage(r):
                valid_results.append(r)
                seen_urls.add(url)

        if len(valid_results) >= min_required:
            return valid_results[:min_required]

        time.sleep(1)

    return []  # NOT valid_results[:min_required]


st.set_page_config(page_title="Competitor Analyzer", layout="wide")

st.title("🧠 AI-Powered Competitor Analyzer")

st.markdown("Enter your **service description** and relevant **keywords** to discover and compare top 5 similar businesses.")

with st.form("user_input_form"):
    description = st.text_area("📝 Description of Your Services", height=100)
    keywords_input = st.text_input("🔑 Keywords (comma-separated)")
    submitted = st.form_submit_button("🔍 Analyze Competitors")

if submitted:
    if not description.strip():
        st.warning("Please enter a service description.")
    elif not keywords_input.strip():
        st.warning("Please enter at least one keyword.")
    else:
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]

        st.info("🔎 Searching competitors...")
        #results = search_competitor_urls(description=description, keywords=keywords, max_results=10)
        results = get_enough_companies(description, keywords, min_required=5)

        scraped_data = []
        st.info("🧽 Scraping competitor websites...")
        progress = st.progress(0)

        for i, result in enumerate(results):
            data = scrape_website(result["url"])
            if data:
                scraped_data.append(data)
            progress.progress((i + 1) / len(results))

        if not scraped_data:
            st.error("❌ No valid competitors found.")
        else:
            st.success("✅ Found valid competitors! Ranking now...")

            ranked = rank_companies(description, keywords, scraped_data, top_n=5)
                    
            # Display Top 5
            st.markdown("## 🏆 Top 5 Competitors")
            rows = []
            for idx, comp in enumerate(ranked, 1):
                with st.container():
                    st.markdown(f"### {idx}. **{comp['company_name']}**")
                    st.markdown(f"🔗 [Visit Website]({comp['url']})")
                    st.markdown(f"**Services:** {comp['services']}")
                    st.markdown(f"**Keywords:** {comp['keywords']}")
                    st.markdown(f"🔢 **Similarity Score:** {comp['score']:.2f}")
                    st.markdown("---")
                    # 👇 Prepare row for CSV
                    rows.append({
                        "S. No": idx,
                        "Company Name": comp["company_name"],
                        "Services": comp["services"],
                        "Keywords": comp["keywords"]
                    })

            # 👇 CSV Export Button
            if rows:
                df = pd.DataFrame(rows)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Results as CSV",
                    data=csv,
                    file_name="top_competitors.csv",
                    mime="text/csv"
                )