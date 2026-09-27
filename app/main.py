from pathlib import Path
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from .analyzer import analyze

app = FastAPI(title="GenreDNA", version="0.1.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    return Path("static/index.html").read_text(encoding="utf-8")

@app.post("/api/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    allowed = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(400, "Supported formats: MP3, WAV, FLAC, OGG, M4A")
    data = await file.read()
    if len(data) > 100 * 1024 * 1024:
        raise HTTPException(413, "Maximum file size is 100 MB.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        return analyze(path)
    except Exception as exc:
        raise HTTPException(422, f"Could not analyze audio: {exc}") from exc
    finally:
        Path(path).unlink(missing_ok=True)
