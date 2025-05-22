import asyncio
import json
from aio_pika.abc import AbstractIncomingMessage

from common.s3.reader import S3Reader
from common.rabbitmq.consumer import RabbitMQConsumer
from temp_predictor.predictor.predictor import TemperaturePredictor
from temp_predictor.tg_client.tg_client import TgClient

class PredictionService:
    def __init__(self,
                 rabbit_consumer: RabbitMQConsumer,
                 s3_reader: S3Reader,
                 predictor: TemperaturePredictor,
                 tg_client: TgClient) -> None:
        self.rabbit_consumer = rabbit_consumer
        self.s3_reader = s3_reader
        self.predictor = predictor
        self.tg_client = tg_client

    async def process_message(self, message: AbstractIncomingMessage) -> None:
        async with message.process():
            print(f"Got message frm queue: {message.body}")

            decoded_msg = json.loads(message.body)
            chat_id, image = decoded_msg["chat_id"], decoded_msg["image"]

            print(f"Downloading image {image} from S3...")
            image_bytes = await self.s3_reader.download_image(image)

            # do inference and send result back to the (future) user
            result = await self.predictor.predict_from_bytes(image_bytes)
            await self.tg_client.send_message(chat_id, result)


    async def run(self):
        await self.rabbit_consumer.start_consuming(self.process_message)
