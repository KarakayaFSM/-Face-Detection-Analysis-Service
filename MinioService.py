from minio import Minio
from minio.error import S3Error
import os
import requests

MINIO_ACCESS_NAME = 'MINIO_ACCESS_NAME'
MINIO_ACCESS_SECRET = 'MINIO_ACCESS_SECRET'
MINIO_URL = 'MINIO_URL'
DEFAULT_BUCKET_NAME = 'my-images'
DEFAULT_FILE_PATH = '/'
DEFAULT_URL_PREFIX = f'{os.environ[MINIO_URL]}/{os.environ[MINIO_ACCESS_NAME]}/{DEFAULT_BUCKET_NAME}'


def initialize_minio_client():
    return Minio(
        os.environ[MINIO_URL].removeprefix('https://'),
        os.environ[MINIO_ACCESS_NAME],
        os.environ[MINIO_ACCESS_SECRET]
    )


client = initialize_minio_client()

def download_file(object_name):
    response = None
    try:
        response = client.get_object(
            DEFAULT_BUCKET_NAME,
            f'{DEFAULT_FILE_PATH}/{object_name}'
        )
    except S3Error as err:
        print(err)

    return response
