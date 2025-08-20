# Secure Bharat — Fraud Call Detection (API + Frontend + Android snippets)

A tiny, free‑tier‑friendly fraud call detector. It learns patterns from phone numbers and optionally boosts risk when transcripts contain suspicious phrases (KYC/OTP/etc).

## 0) Prerequisites
- Python 3.10+ installed
- Git + GitHub account
- (Optional) Android Studio if integrating in an app

## 1) Project Files
```
app.py
feature_extractor.py
train_model.py
fraud_numbers.csv
requirements.txt
render.yaml
Procfile
web/index.html
README.md
```

## 2) Create & activate virtual env (Windows)
```powershell
python -m venv .venv
.\.venv\Scriptsctivate
```
Mac/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3) Install requirements
```bash
pip install -r requirements.txt
```

## 4) Train the model
```bash
python train_model.py
```

## 5) Run locally
```bash
uvicorn app:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs and test.

## 6) Make a GitHub repo
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<you>/secure-bharat-fraud-call-api.git
git push -u origin main
```

## 7) Deploy to Render
- New → Web Service → Connect GitHub repo
- Build: `pip install -r requirements.txt && python train_model.py`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`

## 8) API
- `GET /health`
- `GET /predict?number=+911234567890`
- `POST /predict_batch` { "numbers": ["+91...", "+1..."] }
- `POST /predict_call` { "caller": "...", "transcript": "..." }

## 9) Android (Kotlin) OkHttp
```kotlin
implementation("com.squareup.okhttp3:okhttp:4.12.0")
val client = OkHttpClient()
val JSON = "application/json; charset=utf-8".toMediaType()
fun predictCall(baseUrl: String, caller: String, transcript: String?, onDone: (String)->Unit) {
    val body = JSONObject().put("caller", caller).put("transcript", transcript).toString().toRequestBody(JSON)
    val req = Request.Builder().url("$baseUrl/predict_call").post(body).build()
    client.newCall(req).enqueue(object: Callback {
        override fun onFailure(call: Call, e: IOException) { onDone("Error: ${e.message}") }
        override fun onResponse(call: Call, response: Response) { onDone(response.body?.string() ?: "{}") }
    })
}
```

## 10) Static Web UI
Open `web/index.html`, set API base, test.
