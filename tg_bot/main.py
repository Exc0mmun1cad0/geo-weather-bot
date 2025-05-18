import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.redis import RedisStorage


from config import settings as config
from handlers.handlers import router
from middlewares.throttling import ThrottlingMiddleware
from middlewares.dependency import DependencyMiddleware
from common.rabbitmq.producer import RabbitMQProducer
from common.s3.writer import S3Writer


async def main() -> None:
    # Connect to redis
    redis_storage = RedisStorage.from_url(config.REDIS_URL)

    # Connect to RabbitMQ
    rabbit_producer = RabbitMQProducer(config.RABBITMQ_TOPIC)
    await rabbit_producer.connect(config.RABBITMQ_URL)

    # Connect to S3 storage
    s3_writer = S3Writer(
        access_key=config.S3_ACCESS_KEY_ID,
        secret_key=config.S3_SECRET_ACCESS_KEY,
        endpoint_url=config.S3_ENDPOINT_URL,
        bucket_name=config.S3_BUCKET,
    )

    # Regsiter middlewares
    router.message.middleware.register(ThrottlingMiddleware(redis_storage))
    router.message.middleware.register(DependencyMiddleware(rabbit_producer, s3_writer))

    bot = Bot(token=config.TG_BOT_TOKEN)
    dp = Dispatcher()

    dp.include_router(router)
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
