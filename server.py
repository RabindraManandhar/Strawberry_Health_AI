from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, StreamingResponse # FileResponse for single file, StreamResponse for zipfile
from datetime import datetime
import os
import zipfile
from io import BytesIO


app = FastAPI()

# Directory to save uploaded images
upload_dir =  "uploaded_images"
os.makedirs(upload_dir, exist_ok=True)


@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Generate a unique filename using the current timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{timestamp}_{file.filename}"

        # Save the uploaded file
        file_path = os.path.join(upload_dir, filename)

        with open(file_path, "wb") as buffer:
            buffer.write(await file.read())

        return {"filename": {filename}, "status": "Image received successfully!"}
    except Exception as e:
        return {"error": str(e)}

        
@app.get("/list_images")
def list_images():
    files = os.listdir(upload_dir)
    return {f"files: {files}"}


@app.get("/get_image/{filename}")
def get_image(filename: str):
    file_path = os.path.join(upload_dir, filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Image not found!")
    
    return FileResponse(file_path)


@app.get("/get_all_images")
def get_all_images():
   try:
        # Check if thre are any files in the uploaded images directory
        files = os.listdir(upload_dir)
        if not files:
            raise HTTPException(status_code=404, detail="No images found")

        # Create a BytesIO object to hold the zip data in memory
        zip_buffer = BytesIO()    

        # Create a zip file in memory
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Add all files in the upload directory to the zip file
            for file in files:
                file_path = os.path.join(upload_dir, file)
                if os.path.isfile(file_path):
                    zip_file.write(file_path, file)


        # Seek to the beginning of the BytesIO buffer
        zip_buffer.seek(0)

        # Return the zip file as a response
        return StreamingResponse(
            zip_buffer, media_type="application/x-zip-compressed", headers={"content-Disposition": "attachment; filename=images.zip"}
        )
   
   except Exception as e:
       # log the error for debugging
       print(f"Error while zipping images: {str(e)}") 
       raise HTTPException(status_code=500, detail="Internal Server Error")
