from .base import BaseS3


class S3Reader(BaseS3):
    async def download_image(
            self,
            object_name: str,
    ):
        async with self.get_client() as client:
            response = await client.get_object(
                Bucket=self.bucket_name,
                Key=object_name,
            )
            async with response["Body"] as stream:
                content = await stream.read()

            with open("downloaded.jpg", "wb") as f:
                f.write(content)
