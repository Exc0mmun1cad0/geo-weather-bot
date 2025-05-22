import aiohttp

from urllib.parse import urlencode


class TgClient:
    tg_bot_host = "api.telegram.org"
    send_message_method = "sendMessage"

    def __init__(self, bot_token: str) -> None:
        self.bot_url = f"https://{self.tg_bot_host}/bot{bot_token}"
        self.session = aiohttp.ClientSession()

    async def send_message(self, chat_id: str, text: str) -> str:
        url_params = {
            "chat_id": chat_id,
            "text": text,
        }
        url = f"{self.bot_url}/{self.send_message_method}?{urlencode(url_params)}"
        async with self.session.get(url) as response:
            return await response.json()

    async def close(self):
        await self.session.close()

######## Example usage ########
# import asyncio

# from http_client.http_client import TelegramAPI

# async def main() -> None:
#     bot_token = "8150771800:AAE4XXBXMn6oaI2ZkzkU3k3jpotUfu-0OrI"
#     chat_id = "477972853"
#     text = "Hello"

#     tg_api = TelegramAPI(bot_token)
#     response = await tg_api.send_message(chat_id, text)
#     await tg_api.close()

# if __name__ == "__main__":
#     asyncio.run(main())
