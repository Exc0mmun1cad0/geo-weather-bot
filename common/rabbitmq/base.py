from abc import ABC
from aio_pika import connect_robust
from aio_pika.exceptions import ChannelClosed


class BaseRabbitMQ(ABC):
    def __init__(self, queue_name: str, exchange_name: str = "", routing_key: str = None) -> None:
        self.queue_name = queue_name
        self.exchange_name = exchange_name
        self.routing_key = routing_key or queue_name
        self.connection = None
        self.channel = None
        self.queue = None

    async def connect(self, url: str) -> None:
        self.connection = await connect_robust(url)
        self.channel = await self.connection.channel()

        queue_params = {"name": self.queue_name, "durable": True}
        # try: TODO: refactor this bullshit
        #     self.queue = await self.channel.declare_queue(**queue_params, passive=True)
        # except ChannelClosed:
        self.queue = await self.channel.declare_queue(**queue_params)

        # Bind to exchange in case it's not default
        # TODO: maybe i need it but idk

    async def close(self) -> None:
        # Close channel
        if self.channel and not self.channel.is_closed:
            await self.channel.close()

        # Close connection
        if self.connection and not self.connection.is_closed:
            await self.connection.close()
