from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests
import re
from urllib.parse import urljoin
load_dotenv()

headers = {
    "User-Agent": "UGA-CS-RAG-Assistant/0.1 (student project; contact: https://github.com/Suhail-Jones)"
}

#Gets the site and parses it into beautifulsoup html
r = requests.get("https://csci.franklin.uga.edu/courses/all", headers=headers)
soup = BeautifulSoup(r.text, 'html.parser')

#Finds every individual cs course link, constructs the full url, and adds it to the sources list
#if any 4-digit course number in the slug falls in the 1000-4999 undergrad range.
sources = set()
baseURL = "https://csci.franklin.uga.edu"
links = soup.find_all(href=re.compile('content/csci'))
for link in links:
    courseNumbers = re.findall(r'\d{4}', link['href'])
    if any(1000 <= int(num) <= 4999 for num in courseNumbers):
        fullURL = urljoin(baseURL, link['href'])
        sources.add(fullURL)

with open('sources.txt', 'w') as file:
    for url in sorted(sources):
        file.write(f"{url}\n")
print(len(sources))

