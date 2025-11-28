from fastapi import FastAPI, UploadFile, File
from ia_service import predict_from_bytes
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = predict_from_bytes(image_bytes)
    return result

@app.get("/")
async def read_index():
    return FileResponse("index.html")