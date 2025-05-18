from typing import Awaitable, Any, Callable, Dict
from aiogram import BaseMiddleware
from aiogram.fsm.storage.redis import RedisStorage
from aiogram.types import Message, TelegramObject


from tg_bot.messages.messages import TOO_MANY_REQUESTS_MSG


class ThrottlingMiddleware(BaseMiddleware):
    def __init__(
            self,
            storage: RedisStorage,
            limit: int = 5,
            window: int = 15,
    ) -> None:
        self.storage = storage
        self.window = window
        self.limit = limit

    async def __call__(
            self,
            handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
            event: Message,
            data: Dict[str, Any]
    ) -> Any:
        user_key = f"user:{event.from_user.id}"

        count = await self.storage.redis.incr(user_key)
        if count == 1:
            await self.storage.redis.expire(user_key, self.window)

        if count >= self.limit:
            await event.answer(TOO_MANY_REQUESTS_MSG)
            return

        return await handler(event, data)
