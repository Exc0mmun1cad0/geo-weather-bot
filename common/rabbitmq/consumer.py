from .base import BaseRabbitMQ

import asyncio

from aio_pika.abc import AbstractIncomingMessage
from typing import Callable, Awaitable

class RabbitMQConsumer(BaseRabbitMQ):
    async def start_consuming(
        self, 
        callback: Callable[[AbstractIncomingMessage], Awaitable[None]],
    ) -> None:
        # Will take no more than 10 messages in advance
        await self.channel.set_qos(prefetch_count=100)
        # Start consuming itself
        await self.queue.consume(callback)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            print("Cancelled")
        finally:
            await self.close()

###### Example ###### 
# from rabbitmq.consumer import RabbitMQConsumer

# import asyncio
# from aio_pika.abc import AbstractIncomingMessage


# async def process_message(
#     message: AbstractIncomingMessage,
# ) -> None:
#     async with message.process():
#         print(message.body)
#         await asyncio.sleep(1)

# async def main():
#     producer = RabbitMQConsumer("messages")
#     await producer.connect("amqp://guest:guest@localhost:5672")

#     await producer.start_consuming(process_message)


# if __name__ == "__main__":
#     asyncio.run(main())
