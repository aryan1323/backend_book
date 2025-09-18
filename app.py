from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil, os
from tempfile import NamedTemporaryFile
from book_ocr_matcher import BookOCRMatcher

app = FastAPI()
matcher = BookOCRMatcher("books.csv")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["*"] only if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/match-books")
async def match_books(
    image: UploadFile = File(...),
    genres: str = Form(None)
):
    if not image:
        raise HTTPException(status_code=400, detail="No image file provided")

    genre_list = [g.strip().lower() for g in genres.split(",")] if genres else []

    try:
        with NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1]) as tmp:
            shutil.copyfileobj(image.file, tmp)
            tmp_path = tmp.name

        results = matcher.process_bookshelf_image(tmp_path, min_score=50)
        print(genre_list)
        if genre_list:
            results = [
                r for r in results
                if any(genre in r["categories"].lower() for genre in genre_list)
            ]
        

        os.remove(tmp_path)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error processing image")
