import requests
import urllib.parse
import csv
from bs4 import BeautifulSoup


def get_location_urls() -> list[str]:
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

    camera_urls = []

    for el in elements:
        location_link = el.find("span")
        location = location_link.text

        path = (location_link.find("a").get("href"))[len("index.php?"):]
        url = earthcam_url + "?" + path

        camera_urls.append(url)

    return camera_urls
