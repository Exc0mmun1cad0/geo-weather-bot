from io import BytesIO

from .base import BaseS3


class S3Writer(BaseS3):
    async def upload_image(
            self,
            key: str,
            file_bytes: BytesIO,
    ) -> None:
        async with self.get_client() as client:
            await client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=file_bytes.getvalue(),
            )
