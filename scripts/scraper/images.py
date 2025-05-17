import csv
import os
import time
import requests
from playwright.sync_api import sync_playwright, Page


CSV_HEADERS = ["image", "temperature"]
IMAGE_DIR = "images"


def press_consent_button(page: Page) -> None:
    earthcam_url = "https://www.earthcam.com/network"
    button_selector = ".fc-button.fc-cta-consent.fc-primary-button"

    page.goto(earthcam_url)
    page.wait_for_selector(button_selector)
    time.sleep(2)
    page.locator(button_selector).click()


def download_image(image_url: str) -> str:
    resp = requests.get(image_url)
    image_name = str(int(time.time())) + ".jpg"
    image_path = os.path.join(IMAGE_DIR, image_name)
    with open(image_path, "wb") as f:
        f.write(resp.content)
    return image_name


def get_image_url_and_t(page: Page, camera_url: str) -> tuple[str, str]:
    page.goto(camera_url)
    page.wait_for_load_state()
    img_url = page.locator('img[loading="lazy"]').get_attribute('src')
    t = page.locator("div.myec_celsius").text_content()
    return img_url, t


def main() -> None:
    try:
        os.mkdir(IMAGE_DIR)
    except FileExistsError:
        pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        press_consent_button(page)

        with open("cameras.csv", "r") as f_in, open(os.path.join(IMAGE_DIR, "images.csv"), "w") as f_out:
            reader = csv.reader(f_in)
            next(reader)
            writer = csv.writer(f_out)
            writer.writerow(CSV_HEADERS)

            skipped = 0
            for row in reader:
                camera_url = row[3]
                try:
                    image_url, t = get_image_url_and_t(page, camera_url)
                    image_name = download_image(image_url)
                    writer.writerow([image_name, t[:t.find(" ")]])
                except:
                    skipped += 1
                    print(skipped)
                    continue


if __name__ == "__main__":
    main()
