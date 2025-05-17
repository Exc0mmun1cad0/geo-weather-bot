import requests
import csv
from bs4 import BeautifulSoup


EARTHCAM_URL = "https://www.earthcam.com/network/"
CSV_HEADERS = ["Location", "URL"]


response = requests.get(EARTHCAM_URL)
soup = BeautifulSoup(response.text, "lxml")

with open("locations.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(CSV_HEADERS)

    location_elements = soup.find_all('a', class_=['locationLink', 'highlightOn'])
    for location_element in location_elements:
        url_path = location_element.get("href")[len("index.php?"):]
        location = location_element.text
        if location == "": continue
        writer.writerow(
            [location, EARTHCAM_URL + "?" + url_path],
        )
