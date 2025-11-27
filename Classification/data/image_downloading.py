from google.cloud import storage
import os, zipfile

BUCKET_NAME = "dx-scin-public-data"
PREFIX = "dataset/images/"
OUTPUT_DIR = "images"
def download_images():
    client = storage.Client.create_anonymous_client()
    bucket = client.bucket(BUCKET_NAME)
    blobs = bucket.list_blobs(prefix=PREFIX)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for blob in blobs:
        if blob.name.endswith("/"):
            continue
        filename = blob.name.split("/")[-1]
        blob.download_to_filename(os.path.join(OUTPUT_DIR, filename))
        print("Downloaded", filename)

def zip_images():
    with zipfile.ZipFile("images.zip", "w") as z:
        for file in os.listdir(OUTPUT_DIR):
            z.write(os.path.join(OUTPUT_DIR, file))

if __name__ == "__main__":
    download_images()
    zip_images()
