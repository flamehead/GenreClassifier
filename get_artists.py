import requests
from bs4 import BeautifulSoup
import csv

urls = [
    "https://en.wikipedia.org/wiki/List_of_Canadian_musicians",
    "https://en.wikipedia.org/wiki/List_of_bands_from_Canada"
]

headers = {
    "User-Agent": "Mozilla/5.0"
}

names = set()

for url in urls:
    print(f"Fetching: {url}")
    response = requests.get(url, headers=headers)

    response.encoding = "utf-8"

    soup = BeautifulSoup(response.text, "html.parser")

    content = soup.find("div", class_="mw-parser-output")
    if not content:
        continue

    for li in content.find_all("li"):
        link = li.find("a")
        if link and link.get("title"):
            name = link.get_text(strip=True)

            if len(name) > 1 and not name.lower().startswith("list of"):
                names.add(name)

print(f"Collected {len(names)} names")

sorted_names = sorted(names)

with open("canadian_musicians_and_bands.csv", "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Name"])
    for name in sorted_names:
        writer.writerow([name])

print("Saved CSV successfully")