import requests
from bs4 import BeautifulSoup
import os

# Years to scrape
years = [2025,2024, 2023, 2022, 2021, 2020,2019,2018,2017,2016]

base_url = "https://www.nirfindia.org/Rankings/{}/EngineeringRanking.html"

headers = {
    "User-Agent": "Mozilla/5.0"
}

for year in years:
    print(f"\n📅 Processing Year: {year}")

    url = base_url.format(year)
    res = requests.get(url, headers=headers)

    if res.status_code != 200:
        print(f"❌ Failed to fetch page for {year}")
        continue

    soup = BeautifulSoup(res.text, "lxml")

    pdf_links = []

    # Extract PDF links
    for link in soup.find_all("a"):
        href = link.get("href")
        if href and ".pdf" in href:
            if not href.startswith("http"):
                href = "https://www.nirfindia.org" + href
            pdf_links.append(href)

    # Create year folder
    year_folder = f"data/pdfs/{year}"
    os.makedirs(year_folder, exist_ok=True)

    print(f"🔗 Found {len(pdf_links)} PDFs")

    # Download PDFs
    for i, pdf_url in enumerate(pdf_links):
        try:
            response = requests.get(pdf_url, headers=headers)

            file_path = os.path.join(year_folder, f"{year}_{i}.pdf")

            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"✅ Downloaded: {file_path}")

        except Exception as e:
            print(f"⚠️ Error downloading {pdf_url}: {e}")

print("\n🎉 Multi-year scraping completed!")