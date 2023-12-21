from fastapi import FastAPI, Response
import cv2
import numpy as np
import base64
import json

from spoofing import is_spoofing
from starlette.requests import Request
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace

from pymilvus import connections, db, Collection, utility, CollectionSchema, FieldSchema, DataType
from pydantic import BaseModel
from dotenv import load_dotenv

import os

load_dotenv('.env')
db_name = os.environ.get("db_name")
collection_name = os.environ.get("collection_name") 
user = os.environ.get("user")
password = os.environ.get("password")
host = os.environ.get("host")
port = os.environ.get("port")

class Item(BaseModel):
    identity: str
    image_base64: str

conn = connections.connect(
    alias="default",
    user=user,
    password=password,
    host=host,
    port=port,
    db_name=db_name
)

# Get an existing collection.
collection = Collection(collection_name)      
collection.load(replica_number=1)

# Check the loading progress and loading status
utility.load_state(collection_name)
# Output: <LoadState: Loaded>

utility.loading_progress(collection_name)
# Output: {'loading_progress': 100%}

search_params = {
    "metric_type": "COSINE", 
    "offset": 0, 
    "ignore_growing": False, 
    "params": {"nprobe": 1}
}

detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

# image -> bgr_image
def crop_image(image, scale = 1):
    scale = 1
    imageWidth = int(image.shape[1] * scale)
    imageHeight = int(image.shape[0] * scale)
    resizedImage = cv2.resize(image, (imageWidth, imageHeight))

    detector.setInputSize((imageWidth, imageHeight))
    faces1 = detector.detect(resizedImage)
    face = faces1[1][0] if faces1[1] is not None else None
    if (face is None):
        return None

    coords = face[:-1].astype(np.int32)
    detected_image = image[coords[1]:(coords[1] + coords[3]), coords[0]:(coords[0] + coords[2])]
    return detected_image

def get_face_embedding(bgr_image):
    return DeepFace.represent(bgr_image, enforce_detection=False, model_name="Facenet512")[0]['embedding']

def predict(identity, bgr_detected_image):
    image_embedding = get_face_embedding(bgr_detected_image)
    results = collection.search(
    data=[image_embedding], 
    anns_field="embedding", 
    # the sum of `offset` in `param` and `limit` 
    # should be less than 16384.
    param=search_params,
    limit=1,
    expr=f"identity == '{identity}'",
    # set the names of the fields you want to 
    # retrieve from the search result.
    output_fields=['identity', 'embedding'],
    consistency_level="Strong"
    )

    return {
        'label' : results[0][0].entity.identity,
        # 'distance' : results[0][0].distance
        'probability' : results[0][0].distance
        
    }

def base64ToImage(image_base64):
    image_bytes = base64.b64decode(image_base64)
    # Konversi byte ke numpy array
    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
    # Membaca gambar menggunakan OpenCV
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    image = cv2.flip(image, 1)
    image = cv2.resize(image, (480, 640))
    return image


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Ganti dengan daftar domain yang diizinkan jika perlu
    allow_credentials=True,
    allow_methods=["*"],  # Atur metode permintaan yang diizinkan jika perlu
    allow_headers=["*"],  # Atur header permintaan yang diizinkan jika perlu
)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/dataset/count/{identity}")
async def countDataset(request: Request, identity: str):
    res = collection.query(
            expr= f"identity == '{identity}'", 
            output_fields = ["count(*)"],
    )
    count = res[0]['count(*)']
    return {
        identity: identity,
        'count': count
    }

@app.post("/dataset")
async def insertDataset(request: Request):
    data = await request.form()

    detected_image = base64ToImage(data['image_base64'])
    # cv2.imwrite('gambar_asli.jpg', image)
    # detected_image = crop_image(image)
    
    # if detected_image is None:
    #     print('gagal')
    #     return {
    #         'message': 'wajah tidak terdeteksi'
    #     }
    detected_image = cv2.resize(detected_image, (480, 640))
    flip_detected_image = cv2.flip(detected_image, 1)
    # print('berhasil')

    cv2.imwrite('image.jpg', detected_image)
    cv2.imwrite('flipped_image.jpg', flip_detected_image)

    identity = data['identity']
    embedding = get_face_embedding(detected_image)
    flip_embedding = get_face_embedding(flip_detected_image)

    try:
        result = collection.insert([
            [identity, identity], [embedding, flip_embedding]
        ])
        return True
    except Exception as e:
        print(e)
    

@app.delete("/dataset/{identity}")
async def deleteDataset(request: Request, identity: str):
    expr = f"identity == '{identity}'"
    collection.delete(expr)
    return True

@app.post("/face-recognition")
async def faceRecognition(request: Request):
    # data = await request.form()
    data = await request.form()
    identity = data['identity']
    detected_image = base64ToImage(data['image_base64'])
    # detected_image = crop_image(image)

    # if detected_image is None:
    #     return {
    #         'message': 'wajah tidak terdeteksi'
    #     }
    
    # detected_image = cv2.resize(detected_image, (480, 640))
    # cv2.imwrite('test.jpg', detected_image)
    try:
        prediction = predict(identity, detected_image)
    except Exception as e:
        prediction = {
            'identity': None
        }
    
    return prediction
    

@app.post("/spoofing")
async def detectSpoofing(request: Request):
    try:
        # data = await request.form()
        data = await request.form()
        image_base64 = data['image_base64']
        image_bytes = base64.b64decode(image_base64)
        # Konversi byte ke numpy array
        image_np = np.frombuffer(image_bytes, dtype=np.uint8)
        # Membaca gambar menggunakan OpenCV
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
        # image = cv2.flip(image, 1)
        image = cv2.resize(image, (480, 640))
        cv2.imwrite('test.jpg', image)
        # return image_base64
        return is_spoofing(image)
    
    except Exception as e:
        return {
            'type': None,
            'score' : None
        }
