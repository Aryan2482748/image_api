from fastapi import FastAPI, File, UploadFile, Response
from fastapi.responses import StreamingResponse
from typing import List
import io
import cv2
import numpy as np
from PIL import Image

app = FastAPI()

@app.post("/process_image/stream")
async def process_image_stream(image: UploadFile = File(...)):
    # Read the image file
    contents = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Detect edges
    edges = cv2.Canny(blur_image, 10, 70)

    # Convert the processed image to bytes for streaming
    res, im_png = cv2.imencode(".png", edges)
    image_bytes = io.BytesIO(im_png.tobytes())

    return StreamingResponse(image_bytes, media_type="image/png")
