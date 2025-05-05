import os
import logging
import asyncio

from aiogram import Bot, Dispatcher


def NewBot() -> Bot:
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise ValueError("bot tOKEN is missing. Set it thorugh env var BOT_TOKEN")
    
    bot = Bot(token=token)
    return bot


dp = Dispatcher()
bot = NewBot()


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
