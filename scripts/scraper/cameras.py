# TODO: fix issue with New York
import time
import csv
import lxml
import re
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs
from playwright.sync_api import sync_playwright, Page


CSV_HEADERS = ["Country", "City", "CameraName", "CameraURL"]


def press_consent_button(page: Page) -> None:
    earthcam_url = "https://www.earthcam.com/network"
    button_selector = ".fc-button.fc-cta-consent.fc-primary-button"

    page.goto(earthcam_url)
    page.wait_for_selector(button_selector)
    time.sleep(2)
    page.locator(button_selector).click()


def get_url_params(url):
    params = parse_qs(urlparse(url).query)
    return params["country"][0], params["page"][0]


def write_location_links(page: Page, writer: csv.writer, location_url: str):
    page.goto(location_url)
    page.wait_for_load_state()
    time.sleep(1.5)

    # container contains all camera items
    container = page.locator('div.listContainer.row')
    soup = BeautifulSoup(container.inner_html(), "lxml")

    country, page = get_url_params(location_url)

    camera_items = soup.find_all(class_=re.compile("camera"))
    for camera_item in camera_items:
        camera_name = camera_item.find("span", class_="featuredTitle").text
        city = camera_item.find("div", class_="featuredCity").text
        camera_url = camera_item.find("a", class_="featuredTitleLink").get("href")
        if page != "world": city = page + ", " + city
        print(camera_name, city, camera_url)
        writer.writerow([country, city, camera_name, camera_url])


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        press_consent_button(page)

        with open("locations.csv", "r") as f_in, open("cameras.csv", "w") as f_out:
            reader, writer = csv.reader(f_in), csv.writer(f_out)
            next(reader)
            writer.writerow(CSV_HEADERS)

            for row in reader:
                location_url = row[1]
                write_location_links(page, writer, location_url)

        browser.close()


if __name__ == "__main__":
    main()
