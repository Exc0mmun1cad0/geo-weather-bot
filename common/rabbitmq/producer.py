from .base import BaseRabbitMQ

from aio_pika import Message

class RabbitMQProducer(BaseRabbitMQ):
    async def publish_message(self, message_body: str):
        await self.channel.default_exchange.publish(
            Message(body=message_body.encode()),
            routing_key=self.routing_key,
        )


##### Example ######
# import asyncio

# from rabbitmq.producer import RabbitMQProducer


# async def main():
#     producer = RabbitMQProducer("messages")
#     await producer.connect("amqp://guest:guest@localhost:5672")

#     await producer.publish_message("Whats uuuup")
#     await producer.close()
    


# if __name__ == "__main__":
#     asyncio.run(main())
