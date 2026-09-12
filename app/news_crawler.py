import feedparser
from bs4 import BeautifulSoup
import json
import os
from datetime import datetime

# RSS源列表
RSS_FEEDS = [
    {
        "source": "BBC",
        "url": "http://feeds.bbci.co.uk/news/rss.xml"
    },
    {
        "source": "Reuters",
        "url": "https://feeds.reuters.com/reuters/topNews"
    },
    {
        "source": "TechCrunch",
        "url": "https://techcrunch.com/feed/"
    }
]

def clean_html(raw_text: str) -> str:
    """使用BeautifulSoup清除HTML标签，返回纯文本"""
    if not raw_text:
        return ""
    soup = BeautifulSoup(raw_text, "html.parser")
    return soup.get_text(strip=True)

def fetch_rss_feed(feed_info: dict) -> list[dict]:
    """抓取单个RSS源，提取清洗后的新闻条目"""
    feed = feedparser.parse(feed_info["url"])
    news_items = []
    for entry in feed.entries:
        item = {
            "source": feed_info["source"],
            "title": entry.get("title", ""),
            "link": entry.get("link", ""),
            "published": entry.get("published", ""),
            "summary": clean_html(entry.get("summary", ""))
        }
        news_items.append(item)
    return news_items

def main():
    # 创建输出目录
    output_dir = "data/raw_news"
    os.makedirs(output_dir, exist_ok=True)

    all_news = []
    for feed in RSS_FEEDS:
        print(f"Fetching {feed['source']} ...")
        items = fetch_rss_feed(feed)
        all_news.extend(items)
        print(f"Got {len(items)} news from {feed['source']}")

    # 文件名带当前日期
    date_str = datetime.now().strftime("%Y%m%d")
    output_path = os.path.join(output_dir, f"news_{date_str}.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_news, f, ensure_ascii=False, indent=2)

    print(f"\nSaved total {len(all_news)} news to {output_path}")

if __name__ == "__main__":
    main()
