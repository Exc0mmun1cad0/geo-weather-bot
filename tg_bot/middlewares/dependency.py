from typing import Awaitable, Any, Callable, Dict
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject


from common.rabbitmq.producer import RabbitMQProducer
from common.s3.writer import S3Writer


class DependencyMiddleware(BaseMiddleware):
    def __init__(
            self,
            rabbit_producer: RabbitMQProducer,
            s3_writer: S3Writer,
    ) -> None:
        self.rabbit_producer = rabbit_producer
        self.s3_writer = s3_writer

    async def __call__(
        self,
        handler: Callable[[TelegramObject, Dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: Dict[str, Any]
    ) -> Any:
        data["rabbit_producer"] = self.rabbit_producer
        data["s3_writer"] = self.s3_writer
        return await handler(event, data)
