import json

from io import BytesIO
from uuid import uuid1
from aiogram.filters import CommandStart, Command
from aiogram.types import Message
from aiogram import F, Router


from common.rabbitmq.producer import RabbitMQProducer
from common.s3.writer import S3Writer
from tg_bot.messages.messages import *


router = Router()


@router.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(START_MSG)


@router.message(Command("info"))
async def command_info_handler(message: Message) -> None:
    await message.answer(INFO_MSG)


# Handle images of PNG/JPEG format or attached PNG/JPEG files
@router.message(F.photo | F.document.mime_type.in_({"image/png", "image/jpeg"}))
async def images_handler(message: Message, rabbit_producer: RabbitMQProducer, s3_writer: S3Writer) -> None:
    await message.answer(IMAGE_HANDLER_MSG)

    # Prepare Key(object_name) and bytes(body) for S3 and RabbitMQ
    key = str(uuid1()) # object name
    # object body
    file_id = ""
    if message.document is None:
        file_id = message.photo[-1].file_id
    elif message.photo is None:
        file_id = message.document.file_id
    file = await message.bot.get_file(file_id)
    file_path = file.file_path
    file_bytes = BytesIO()
    await message.bot.download_file(file_path=file_path, destination=file_bytes)
    file_bytes.seek(0)

    # Upload file to S3
    await s3_writer.upload_image(key, file_bytes)
    # Notify about that through RabbitMQ
    msg = {"chat_id": message.chat.id, "image": key}
    await rabbit_producer.publish_message(json.dumps(msg))


@router.message()
async def unknown_command_handler(message: Message) -> None:
    await message.answer(UNKNOWN_CMD_MSG)
