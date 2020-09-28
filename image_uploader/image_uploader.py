import sys
import requests
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

# image_url = sys.argv[1] #we pass the url as an argument

cred = credentials.Certificate('./superb-binder-287603-firebase-adminsdk-7ri7u-a573c155d9.json')
firebase_admin.initialize_app(cred, {
    'storageBucket': 'superb-binder-287603.appspot.com'
})
bucket = storage.bucket()


def upload(path):
    image_data = path
    uploadPath = "test/" + path
    blob = bucket.blob(uploadPath)
    blob.upload_from_string(
            image_data,
            content_type='image/png'
        )
    return blob.public_url