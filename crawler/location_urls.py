import requests
import urllib.parse
import csv
from bs4 import BeautifulSoup

earthcam_url = "https://www.earthcam.com/network/"
csv_headers = ["Location", "URL"]

# Get html content of "network" page
resp = requests.get(url=earthcam_url)
data = resp.content.decode("utf-8")

# Init soup object from bs4 lib
soup = BeautifulSoup(data, "lxml")

# Get elements from sideList which contain link
side_list = soup.find("div", {"class": "sideList"})
elements = side_list.find_all("p", {"class": "location"})

# Extract link from every element and write it to csv with location name
with open("locations.csv", "w", encoding="utf-8") as f:
    # Create csv file
    writer = csv.writer(f, lineterminator="\r")
    writer.writerow(csv_headers)

    # Iterate over elements, write formed url and location to csv
    for el in elements:
        location_link = el.find("span")
        location = location_link.text

        path = (location_link.find("a").get("href"))[len("index.php?"):]
        url = earthcam_url + "?" + path

        writer.writerow([location, url])
