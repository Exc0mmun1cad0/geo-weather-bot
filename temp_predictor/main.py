import asyncio

from config import settings as config
from common.s3.reader import S3Reader
from common.rabbitmq.consumer import RabbitMQConsumer
from service.service import PredictionService
from predictor.predictor import TemperaturePredictor
from tg_client.tg_client import TgClient


async def main() -> None:
    # Connect to RabbitMQ
    rabbit_consumer = RabbitMQConsumer(config.RABBITMQ_TOPIC)
    await rabbit_consumer.connect(config.RABBITMQ_URL)

    # Connect to S3 storage
    s3_reader = S3Reader(
        access_key=config.S3_ACCESS_KEY_ID,
        secret_key=config.S3_SECRET_ACCESS_KEY,
        endpoint_url=config.S3_ENDPOINT_URL,
        bucket_name=config.S3_BUCKET,
    )

    # Create instance of telegram client (async http client under the hood)
    tg_client = TgClient(config.TG_BOT_TOKEN)

    # Create predictor with trained models
    temp_predictor = TemperaturePredictor("temp_predictor/models")


    app = PredictionService(rabbit_consumer, s3_reader, temp_predictor, tg_client)
    await app.run()



if __name__ == "__main__":
    asyncio.run(main())
