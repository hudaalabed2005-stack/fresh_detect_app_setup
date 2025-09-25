import os, io, csv
from datetime import datetime
from collections import deque
from typing import Optional

import threading
import numpy as np
from PIL import Image
import requests
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

import torchvision
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode

# basic config
torch.set_num_threads(1)
_INFER_LOCK = threading.Lock()

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH  = os.getenv("MODEL_PATH", "/app/spoilage_model.pth")
MAX_POINTS  = int(os.getenv("MAX_POINTS", "240"))

FRESHNESS_NAMES = ["Fresh", "Spoiled"]

# preprocessing
IMG_TX = T.Compose([
    T.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor()
])

# FastAPI
app = FastAPI(title="Fruit Freshness & Gas Detector")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

LAST = {"vision": None, "vision_updated": None, "gas": None, "gas_updated": None}
HISTORY = deque(maxlen=MAX_POINTS)
EXPO_TOKENS = set()
LAST_DECISION = None
#notifications
def send_expo_push(title: str, body: str, data: dict | None = None):
    if not EXPO_TOKENS:
        return {"ok": False, "detail": "no tokens"}
    payload = [
        {"to": t, "title": title, "body": body, "sound": "default", "data": data or {}}
        for t in EXPO_TOKENS
    ]
    try:
        r = requests.post(
            "https://exp.host/--/api/v2/push/send",
            json=payload,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            timeout=10,
        )
        return {"ok": r.ok, "status": r.status_code, "resp": r.text[:200]}
    except Exception as e:
        return {"ok": False, "detail": str(e)}
# notifier hook when decision changes
def maybe_notify_on_spoilage():
    global LAST_DECISION
    try:
        s = _summarize(LAST)  # you already have _summarize
        decision = s.get("decision")
        if decision and decision != LAST_DECISION:
            if decision == "SPOILED":
                vis = s.get("vision") or {}
                lbl = vis.get("label") or "Unknown"
                conf = vis.get("confidence") or 0
                g = s.get("gas_ppm") or {}
                title = "Spoilage Alert"
                body = f"{lbl} ({conf}%) • CO₂:{g.get('co2')} NH₃:{g.get('nh3')}"
                send_expo_push(title, body, {"summary": s})
            LAST_DECISION = decision
    except Exception:
        pass

# registeration
class ExpoToken(BaseModel):
    token: str

@app.post("/register_expo")
def register_expo(t: ExpoToken):
    if not t.token.startswith("ExponentPushToken"):
        return JSONResponse({"error": "invalid_token"}, status_code=400)
    EXPO_TOKENS.add(t.token.strip())
    return {"ok": True, "count": len(EXPO_TOKENS)}

@app.post("/unregister_expo")
def unregister_expo(t: ExpoToken):
    EXPO_TOKENS.discard(t.token.strip())
    return {"ok": True, "count": len(EXPO_TOKENS)}

# model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = 0.7
        
        try:
            self.base = torchvision.models.resnet18(weights=None)
        except TypeError:
            self.base = torchvision.models.resnet18(pretrained=False)
        for m in self.base.modules():
            if hasattr(m, "inplace"):
                m.inplace = False
        for p in list(self.base.parameters())[:-15]:
            p.requires_grad = False
        self.base.fc = nn.Sequential()

        self.block1 = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),
        )
        self.block2 = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 9)
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.block1(x)
        y1 = self.block2(x)
        y2 = self.block3(x)
        return y1, y2

_model = None

def load_model():
    global _model
    if _model is not None:
        return _model

    # TorchScript
    try:
        m = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        m.eval().to(DEVICE)
        _model = m
        return _model
    except Exception:
        pass

    # full module or state_dict
    obj = torch.load(MODEL_PATH, map_location=DEVICE)
    if isinstance(obj, nn.Module):
        _model = obj.eval().to(DEVICE)
        return _model
    if isinstance(obj, dict):
        m = Model().to(DEVICE)
        m.load_state_dict(obj, strict=True)
        m.eval()
        _model = m
        return _model

    raise RuntimeError("Unsupported checkpoint format at MODEL_PATH")

def predict_pil(pil: Image.Image):
    model = load_model()
    x = IMG_TX(pil).unsqueeze(0).to(DEVICE)

    with _INFER_LOCK, torch.inference_mode():
        out = model(x)
        if isinstance(out, (tuple, list)) and len(out) >= 2:
            y2 = out[1]  # freshness head
        else:
            y2 = out
        probs_t = torch.softmax(y2, dim=1)[0].tolist()

    idx = int(np.argmax(probs_t))
    label = FRESHNESS_NAMES[idx]
    conf = float(probs_t[idx]) * 100.0
    raw = {FRESHNESS_NAMES[i]: float(p) for i, p in enumerate(probs_t)}
    return {"label": label, "confidence": round(conf, 1), "raw": {"probs": raw}}

# Vision
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        data = await image.read()
        pil = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": "invalid_image", "detail": f"Could not read image ({e})"}, status_code=400)

    try:
        out = predict_pil(pil)
        LAST["vision"] = out
        LAST["vision_updated"] = datetime.utcnow().isoformat()
        maybe_notify_on_spoilage()
        return JSONResponse(out)
    
    except Exception as e:
        return JSONResponse({"error": "inference_failed", "detail": str(e)}, status_code=500)

# Gas 
class GasReading(BaseModel):
    vrl: Optional[float] = None
    adc: Optional[int]   = None
    adc_max: Optional[int] = 4095
    vref: Optional[float]  = 3.3
    rl:   Optional[float]  = 10000.0
    rs:   Optional[float]  = None
    r0:   Optional[float]  = None

def _ppm_from_ratio(ratio: float, a: float, b: float) -> float:
    if ratio is None or ratio <= 0:
        return 0.0
    return max(0.0, a * (ratio ** b))

@app.post("/gas")
def gas(g: GasReading):
    VREF = float(g.vref or 3.3)
    RL   = float(g.rl or 10000.0)
    used_adc = None
    adc_max  = int(g.adc_max or 4095)

    if g.vrl is None and g.adc is not None:
        used_adc = int(g.adc)
        g.vrl = (used_adc / adc_max) * VREF

    if g.vrl is None and g.rs is None:
        return JSONResponse({"error": "need vrl, adc, or rs"}, status_code=400)

    rs = float(g.rs) if g.rs is not None else ((VREF - g.vrl) * RL) / max(0.001, g.vrl)
    r0 = float(g.r0) if g.r0 is not None else rs
    ratio = rs / max(1e-6, r0)

    data = {
        "vrl": round(g.vrl, 3),
        "rs": round(rs, 1),
        "r0": round(r0, 1),
        "ratio": round(ratio, 3),
        "ppm": {
            "co2":     round(_ppm_from_ratio(ratio, 116.6021, -2.7690), 1),
            "nh3":     round(_ppm_from_ratio(ratio, 102.6940, -2.4880), 1),
            "benzene": round(_ppm_from_ratio(ratio, 76.63,   -2.1680), 1),
            "alcohol": round(_ppm_from_ratio(ratio, 77.255,  -3.18),   1),
        },
        "raw": {"adc": used_adc, "adc_max": adc_max, "vref": VREF, "rl": RL, "r0": r0}
    }

    LAST["gas"] = data
    LAST["gas_updated"] = datetime.utcnow().isoformat()
    maybe_notify_on_spoilage()

    # Compute combined decision (vision + gas thresholds)
    summary = _summarize(LAST)
    decision = summary["decision"]
    
    # Save in history (ppm + decision)
    HISTORY.append({
        "time": datetime.utcnow().isoformat(),
        "ppm": data["ppm"],
        "decision": decision
    })
    
    return {"ok": True, "data": data, "decision": decision}

@app.get("/history")
def history():
    return {"history": list(HISTORY)}

@app.get("/export.csv")
def export_csv():
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["timestamp_utc", "co2_ppm", "nh3_ppm", "benzene_ppm", "alcohol_eq", "decision"])
    for r in HISTORY:
        ppm = r["ppm"]
        w.writerow([
            r["time"],
            ppm.get("co2"),
            ppm.get("nh3"),
            ppm.get("benzene"),
            ppm.get("alcohol"),
            r.get("decision") 
        ])
    return HTMLResponse(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="gas_history.csv"'}
    )

# Summary / Health 
def _summarize(last: dict) -> dict:
    """
    Combine the latest vision prediction and gas ppm into a single, simple decision.
    - Vision only votes 'rotten' if label says spoiled/rotten AND confidence >= VISION_MIN_CONF.
    - Any high gas flag can mark the sample as spoiled.
    """
    # thresholds
    VISION_MIN_CONF = 60.0   # %
    CO2_HI  = 2000.0         # ppm
    NH3_HI  = 15.0           # ppm
    BENZ_HI = 5.0            # ppm
    ALC_HI  = 10.0           # eq

    pred = last.get("vision") or {}
    gas  = (last.get("gas") or {}).get("ppm", {}) or {}

    co2   = gas.get("co2")
    nh3   = gas.get("nh3")
    benz  = gas.get("benzene")
    alco  = gas.get("alcohol")

    # gas flags
    co2_hi = (co2 is not None)  and (co2  >= CO2_HI)
    nh3_hi = (nh3 is not None)  and (nh3  >= NH3_HI)
    voc_hi = ((benz or 0) >= BENZ_HI) or ((alco or 0) >= ALC_HI)

    # vision vote (label + confidence)
    label = str(pred.get("label") or "")
    conf  = float(pred.get("confidence") or 0.0)
    looks_rotten = ("spoiled" in label.lower() or "rotten" in label.lower()) and (conf >= VISION_MIN_CONF)

    spoiled = bool(looks_rotten or co2_hi or nh3_hi or voc_hi)

    return {
        "vision": pred,
        "gas_ppm": {"co2": co2, "nh3": nh3, "benzene": benz, "alcohol": alco},
        "gas_flags": {"co2_high": co2_hi, "nh3_high": nh3_hi, "voc_high": voc_hi},
        "decision": "SPOILED" if spoiled else "FRESH",
        "meta": {
            "max_points": MAX_POINTS,
            "thresholds": {
                "vision_min_conf": VISION_MIN_CONF,
                "co2_hi": CO2_HI, "nh3_hi": NH3_HI, "benz_hi": BENZ_HI, "alcohol_hi": ALC_HI
            }
        }
    }


@app.get("/summary")
def summary():
    return _summarize(LAST)

@app.get("/healthz")
def healthz():
    return {"ok": True, "time": datetime.utcnow().isoformat()}
    
# UI
@app.get("/", response_class=HTMLResponse)
def welcome():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Welcome • Smart Freshness Checker</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Raleway:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root{ --green:#22c55e; --green-dark:#16a34a; --glass: rgba(255,255,255,.9); --border: rgba(0,0,0,.06); --shadow: 0 10px 30px rgba(0,0,0,.18) }
  body{
    margin:0; font-family:'Raleway',system-ui,Arial,Helvetica,sans-serif; color:#1f2937;
    background:url('https://i.pinimg.com/originals/30/ab/43/30ab43926be6852d3b03572459ab847d.gif') center/cover no-repeat fixed;
    min-height:100vh; display:grid; place-items:center; padding:24px;
  }
  body::before{
    content:""; position:fixed; inset:0;
    background:rgba(255,255,255,.4); backdrop-filter: blur(2px);
  }
  .wrap{
    position:relative; z-index:1;
    width:min(880px,92%); background:var(--glass); border:1px solid var(--border);
    border-radius:20px; box-shadow:var(--shadow); padding:38px; text-align:center;
    backdrop-filter: blur(6px);
  }
  h1{ margin:8px 0 16px; font-size:clamp(2rem,3.5vw,2.6rem); color:#065f46; font-weight:800; letter-spacing:.6px }
  p{ margin:14px auto; max-width:64ch; line-height:1.7; font-weight:500; font-size:1.05rem }
  .cta{ display:flex; gap:14px; flex-wrap:wrap; justify-content:center; margin-top:22px }
  a.btn{
    text-decoration:none; background:var(--green); color:#fff; padding:14px 22px; border-radius:12px;
    font-weight:700; letter-spacing:.4px; font-size:1rem;
    box-shadow:0 6px 20px rgba(34,197,94,.35); transition:background .2s, transform .15s;
  }
  a.btn:hover{ background:var(--green-dark); transform:translateY(-2px) }
  a.btn.secondary{ background:#e7fff1; color:#064e3b; border:1px solid #b9f3d2; box-shadow:none }
  .pill{ display:inline-block; padding:8px 14px; border-radius:999px; background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0; font-weight:700; font-size:1rem }
  footer{ margin-top:24px; font-size:1rem; opacity:.9; line-height:1.5; font-weight:500 }
</style>
</head>
<body>
  <div class="wrap">
    <span class="pill">Smart Freshness Checker</span>
    <h1>Welcome 👋</h1>
    <p>
      This simple tool helps you check whether your fruit and veggies are still fresh.  
      Take a quick photo and add air-reading values from a small plug-in sensor.  
      The app looks for early signs of spoilage and gives you a clear “Fresh” or “Spoiled” result.
    </p>
    <p>
      <b>How it works:</b><br>
      1) Open the app and upload a photo, or use the camera. <br>
      2) If you have a gas sensor, send readings to improve the result. <br>
      3) See the final prediction and a simple chart of recent readings. <br>
    </p>
    <div class="cta">
      <a class="btn" href="/home">🚀 Launch App</a>
      <a class="btn secondary" href="/export.csv">⬇️ Download Data CSV</a>
    </div>
    <footer>
      Tip: On phones, allow camera permissions and use <b>🔄 Flip</b> to switch to the back camera for sharper photos.
    </footer>
  </div>
</body>
</html>
"""


@app.get("/home", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fruit Freshness & Gas Detector</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<style>
  :root{
    --green:#22c55e; --green-dark:#16a34a; --blue:#0ea5e9; --amber:#f59e0b;
    --red:#ef4444; --slate:#1f2937; --glass-bg: rgba(255,255,255,.86);
    --glass-border: rgba(0,0,0,.06); --shadow: 0 10px 30px rgba(0,0,0,.18);
  }
  body{
    margin:0; font-family:Inter,system-ui,Arial,Helvetica,sans-serif; color:var(--slate);
    background:url('https://i.pinimg.com/originals/3d/91/51/3d9151870044e69f2d93a9d0311275dd.gif') center/cover no-repeat fixed; 
    min-height:100vh
  }
  body::before{content:""; position:fixed; inset:0; background:linear-gradient(180deg,rgba(0,0,0,.32),rgba(0,0,0,.22)); pointer-events:none; z-index:-1}
  header{
    background:linear-gradient(90deg, rgba(34,197,94,.97), rgba(22,163,74,.97));
    padding:18px 16px; text-align:center; color:#fff; box-shadow:var(--shadow);
    position:sticky; top:0; z-index:10; backdrop-filter: blur(4px);
  }
  header h1{ margin:0; font-size:clamp(1.3rem,2.8vw,1.9rem); letter-spacing:.3px }
  header p{ margin:6px 0 0 0; opacity:.95 }
  .container{ width:92%; max-width:1100px; margin:24px auto; display:grid; gap:18px }
  .card{
    background:var(--glass-bg); border-radius:16px; padding:18px; box-shadow:var(--shadow);
    border:1px solid var(--glass-border); backdrop-filter: blur(6px);
    transition: transform .15s ease, box-shadow .2s ease;
  }
  .card:hover{ transform:translateY(-1px); box-shadow: 0 12px 34px rgba(0,0,0,.22); }
  .card h2{ margin:0 0 12px; color:var(--green-dark); font-size:1.15rem; border-left:4px solid var(--green-dark); padding-left:10px }
  button{
    background:var(--green); color:#fff; padding:10px 14px; border:none; border-radius:10px;
    cursor:pointer; font-weight:800; letter-spacing:.2px; box-shadow: 0 5px 18px rgba(34,197,94,.35);
    display:inline-flex; align-items:center; gap:8px; transition: background .15s ease, transform .1s ease;
  }
  button:hover{ background:var(--green-dark) }
  button:active{ transform:translateY(1px) }
  button.secondary{ background:#e7fff1; color:#0b3d2e; border:1px solid #b9f3d2; box-shadow:none; }
  button.gray{ background:#f3f4f6; color:#111827; border:1px solid #e5e7eb; box-shadow:none; }
  input[type=file], input[type=number], select, input[type=time]{
    padding:10px 12px; border:1px solid #d1d5db; border-radius:10px; background:#fff; font-weight:600;
    outline:none; transition:border-color .15s ease, box-shadow .15s ease;
  }
  input[type=file]:focus, input[type=number]:focus, select:focus, input[type=time]:focus{ border-color: var(--green); box-shadow: 0 0 0 3px rgba(34,197,94,.18); }
  .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:center }
  .pill{ padding:6px 10px; border-radius:999px; font-weight:700; font-size:.9rem; border:1px solid #d1d5db; background:#fff }
  .ok{ background:#ecfdf5; color:#065f46; border:1px solid #a7f3d0 }
  .bad{ background:#fef2f2; color:#991b1b; border:1px solid #fecaca }
  .warn{ background:#fffbeb; color:#92400e; border:1px solid #fde68a }
  .big{ font-size:22px; font-weight:900; margin-top:10px }
  img,video,canvas{ max-width:100%; border-radius:12px; margin-top:10px }
  #preview{ display:none; }
  pre{
    white-space:pre-wrap; background:#0b1220; color:#e5e7eb; border-radius:12px; padding:12px; max-height:320px; overflow:auto;
    border:1px solid rgba(255,255,255,.05);
  }
  .chart-wrap{
    position:relative; height:280px; width:100%; overflow:hidden; border-radius:12px; background:#ffffffe6; border:1px solid rgba(0,0,0,.06)
  }
  .chart-empty{
    position:absolute; inset:0; display:flex; align-items:center; justify-content:center;
    color:#6b7280; font-size:.95rem; pointer-events:none; font-weight:700;
  }
  .tiny{ font-size:.85rem; opacity:.8 }
</style>
</head>
<body>
<header>
  <h1>🍎 Fruit Freshness & Gas Detector</h1>
  <p>Upload, predict, and view gas-based decision</p>
</header>
<div class="container">
  <div class="card">
    <h2>1) Upload or Capture Fruit Image <span id="visionStatus" class="pill">idle</span></h2>

    <!-- Upload + Predict -->
    <div class="row">
      <input id="file" type="file" accept="image/*" />
      <button type="button" onclick="predictFile()">🔮 Predict</button>
      <button type="button" class="secondary" onclick="startCam()">📷 Use Webcam</button>
      <button type="button" class="gray" onclick="snap()">📸 Snapshot (Predict Now)</button>
      <button type="button" class="gray" onclick="stopCam()">⏹ Stop Camera</button>
      <button type="button" class="gray" onclick="clearVision()">🧹 Clear Image</button>
    </div>

    <!-- NEW: Camera chooser / front-back flip / auto daily -->
    <div class="row">
      <select id="camSelect" title="Select camera"></select>
      <button type="button" class="gray" onclick="flipCam()">🔄 Flip (front/back)</button>
      <label class="tiny">
        <input id="autoDaily" type="checkbox" onchange="toggleAutoDaily()" />
        Auto daily webcam prediction
      </label>
      <label class="tiny">Time <input id="autoTime" type="time" value="09:00" onchange="saveAutoSettings()" /></label>
      <span class="tiny" id="autoInfo"></span>
    </div>

    <video id="video" autoplay playsinline width="320" height="240" style="display:none;background:#000"></video>
    <canvas id="canvas" width="320" height="240" style="display:none"></canvas>
    <img id="preview" alt="preview" />
    <div id="visionTop" class="big"></div>
    <span id="visionBadge" class="pill" style="display:none"></span>
  </div>

  <div class="card">
    <h2>2) Gas Sensor Reading <span id="gasStatus" class="pill">idle</span></h2>
    <div class="row" style="margin-bottom:8px">
      ADC <input id="adc" type="number" value="1800" />
      Vref <input id="vref" type="number" value="3.3" step="0.1" />
      RL(Ω) <input id="rl" type="number" value="10000" />
      R0(Ω) <input id="r0" type="number" value="10000" />
      <button type="button" onclick="sendGas()">📤 Send</button>
      <button type="button" class="gray" onclick="preset('fresh')">🍏 Fresh</button>
      <button type="button" class="gray" onclick="preset('spoiled')">🍌 Spoiled</button>
      <button type="button" class="gray" onclick="resetGas()">🔁 Reset</button>
    </div>
    <div id="gasBadges" style="margin-top:6px"></div>
  </div>

  <div class="card">
    <h2>3) Final Decision</h2>
    <div id="decision" class="big"></div>
    <pre id="raw"></pre>
  </div>

  <div class="card">
    <h2>4) Gas Chart (last points)</h2>
    <div class="chart-wrap">
      <canvas id="gasChart"></canvas>
      <div id="chartEmpty" class="chart-empty">No readings yet</div>
    </div>
  </div>
</div>

<script>
"use strict";
const $ = (id)=>document.getElementById(id);
const el = {
  file:$('file'), preview:$('preview'), video:$('video'), canvas:$('canvas'),
  visionBadge:$('visionBadge'), visionTop:$('visionTop'),
  gasBadges:$('gasBadges'), decision:$('decision'), raw:$('raw'),
  visionStatus:$('visionStatus'), gasStatus:$('gasStatus'),
  chartEmpty:$('chartEmpty'), gasChart:$('gasChart'),
  adc:$('adc'), vref:$('vref'), rl:$('rl'), r0:$('r0'),
  camSelect:$('camSelect'), autoDaily:$('autoDaily'), autoTime:$('autoTime'), autoInfo:$('autoInfo')
};

/* ====== helpers ====== */
function clearVision(){
  el.preview.src=''; el.preview.style.display='none';
  el.video.style.display='none'; el.canvas.style.display='none';
  el.visionBadge.style.display='none'; el.visionTop.textContent='';
}
function setStatus(target, text){ (target==='vision'?el.visionStatus:el.gasStatus).textContent = text; }
function updateVision(j){
  const lbl = String(j.label||'?'); const conf = Number(j.confidence||0).toFixed(1);
  el.visionBadge.style.display='inline-block';
  const bad = /spoiled|rotten/i.test(lbl);
  el.visionBadge.className='pill '+(bad?'bad':'ok');
  el.visionBadge.textContent = lbl+' • '+conf+'%';
  el.visionTop.textContent = lbl.toUpperCase();
}

/* ====== upload predict ====== */
async function predictFile(){
  const f = el.file.files[0];
  if(!f){ alert("Choose an image"); return; }
  el.preview.src = URL.createObjectURL(f); el.preview.style.display='block';
  const fd = new FormData(); fd.append('image', f, f.name);
  setStatus('vision', 'working…');
  try{
    const r = await fetch('/predict', {method:'POST', body:fd});
    const j = await r.json();
    if(!r.ok || j.error){ alert('Predict failed: '+(j.error||r.statusText)); return; }
    updateVision(j);
    await refresh();
  } finally { setStatus('vision', 'idle'); }
}

/* ====== webcam (multi-camera + flip) ====== */
let stream = null;
let currentFacing = 'environment'; // 'user' | 'environment'
let currentDeviceId = null;

function isSecure() {
  return location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
}

async function listCams(){
  try { 
    await navigator.mediaDevices.getUserMedia({video:true, audio:false}); 
  } catch(_){}

  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === 'videoinput');
  
  const camSel = document.getElementById('camSelect');
  if (!camSel) return;  // guard in case element missing

  // dropdown
  camSel.innerHTML = cams.map((c,i)=>
    `<option value="${c.deviceId}">${c.label || ('Camera '+(i+1))}</option>`
  ).join('');

  // Attach change listener once
  if (!camSel.dataset.bound) {
    camSel.addEventListener('change', async (e)=>{
      currentDeviceId = e.target.value || null;
      await startCam(currentDeviceId);
    });
    camSel.dataset.bound = "1"; // mark so we don’t double bind
  }

  // Pick preferred camera
  if (cams.length && !currentDeviceId) {
    const back = cams.find(c => /back|rear|environment|wide/i.test(c.label));
    currentDeviceId = (back || cams[0]).deviceId;
    camSel.value = currentDeviceId;
  }
}

function stopCam(){
  const vid = document.getElementById('video');
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  vid.srcObject = null;
  vid.style.display='none';
}

async function startCam(deviceId){
  if (!isSecure()) {
    alert('Camera requires HTTPS (or localhost). Please open this page over https://');
    return;
  }
  stopCam();

  const tryConstraints = async (constraints) => {
    return await navigator.mediaDevices.getUserMedia(constraints);
  };

  let constraintsList = [];

  if (deviceId) {
    constraintsList.push({ video: { deviceId: { exact: deviceId } }, audio: false });
  } else {
    // 1) strict back camera
    constraintsList.push({ video: { facingMode: { exact: 'environment' } , width:{ideal:1280}, height:{ideal:720}}, audio: false });
    // 2) ideal back (fallback)
    constraintsList.push({ video: { facingMode: { ideal: 'environment' } , width:{ideal:1280}, height:{ideal:720}}, audio: false });
    // 3) whatever default
    constraintsList.push({ video: true, audio: false });
  }

  const vid = document.getElementById('video');
  let ok = null, lastErr = null;
  for (const c of constraintsList) {
    try { ok = await tryConstraints(c); break; } catch(e){ lastErr = e; }
  }
  if (!ok) { alert('Camera error: '+ lastErr); return; }

  stream = ok;
  vid.srcObject = stream;
  vid.style.display='block';

  // Wait for dimensions, then size canvas correctly
  await vid.play().catch(()=>{});
  if (vid.readyState >= 2) {
    sizeCanvasToVideo();
  } else {
    vid.onloadedmetadata = () => sizeCanvasToVideo();
  }

  // Track selected device
  const track = stream.getVideoTracks()[0];
  const settings = track.getSettings?.() || {};
  currentDeviceId = settings.deviceId || deviceId || currentDeviceId;

  await listCams();
  if (currentDeviceId) document.getElementById('camSelect').value = currentDeviceId;

  // keep your auto-daily scheduler alive if you had one
  if (typeof initAutoDaily === 'function') initAutoDaily();
}

function sizeCanvasToVideo(){
  const vid = document.getElementById('video');
  const cvs = document.getElementById('canvas');
  // Use actual stream size to avoid black snaps
  const w = vid.videoWidth || 320;
  const h = vid.videoHeight || 240;
  cvs.width = w;
  cvs.height = h;
}

document.getElementById('camSelect').addEventListener('change', async (e)=>{
  currentDeviceId = e.target.value || null;
  await startCam(currentDeviceId);
});

async function flipCam(){
  // toggle desired facing and try strict facingMode first
  currentFacing = (currentFacing === 'environment') ? 'user' : 'environment';
  try {
    await startCam(null); // will attempt facingMode path
  } catch {
    // fallback: cycle to another device in the list
    const sel = document.getElementById('camSelect');
    const opts = Array.from(sel.options);
    if(opts.length > 1){
      const idx = opts.findIndex(o => o.value === currentDeviceId);
      const next = opts[(idx+1)%opts.length].value;
      await startCam(next);
    }
  }
}

// Snapshot -> predict
async function snap(){
  if(!stream){ alert('Start the webcam first'); return; }
  const vid = document.getElementById('video');
  const cvs = document.getElementById('canvas');
  const ctx = cvs.getContext('2d');

  // Ensure canvas matches live stream size
  if (cvs.width !== vid.videoWidth || cvs.height !== vid.videoHeight) {
    sizeCanvasToVideo();
  }

  ctx.drawImage(vid, 0, 0, cvs.width, cvs.height);
  cvs.toBlob(async b=>{
    const fd = new FormData(); fd.append('image', b, 'snap.jpg');
    document.getElementById('visionStatus').textContent='working…';
    try{
      const r = await fetch('/predict', {method:'POST', body:fd});
      const j = await r.json();
      if(!r.ok || j.error){ alert('Predict failed: '+(j.error||r.statusText)); return; }
      if (typeof updateVision === 'function') updateVision(j);
      if (typeof refresh === 'function') await refresh();
      if (typeof rememberAutoRun === 'function') rememberAutoRun();
    } finally {
      document.getElementById('visionStatus').textContent='idle';
    }
  }, 'image/jpeg', 0.92);
}

/* ====== Auto daily prediction (client-side) ====== */
// settings persistence
const LS_KEY_ON   = 'autoDaily_on';
const LS_KEY_TIME = 'autoDaily_time';
const LS_KEY_LAST = 'autoDaily_last';

function loadAutoSettings(){
  el.autoDaily.checked = localStorage.getItem(LS_KEY_ON) === '1';
  const t = localStorage.getItem(LS_KEY_TIME) || '09:00';
  el.autoTime.value = t;
  paintAutoInfo();
}
function saveAutoSettings(){
  localStorage.setItem(LS_KEY_ON, el.autoDaily.checked ? '1' : '0');
  localStorage.setItem(LS_KEY_TIME, el.autoTime.value || '09:00');
  paintAutoInfo();
}
function toggleAutoDaily(){ saveAutoSettings(); initAutoDaily(); }

function paintAutoInfo(){
  const last = localStorage.getItem(LS_KEY_LAST);
  const t = el.autoTime.value || '09:00';
  el.autoInfo.textContent = el.autoDaily.checked
    ? `Scheduled daily at ${t}${last?` • last run: ${new Date(last).toLocaleString()}`:''}`
    : `Auto daily is off`;
}
function rememberAutoRun(){
  localStorage.setItem(LS_KEY_LAST, new Date().toISOString());
  paintAutoInfo();
}

let autoTimer = null;

function initAutoDaily(){
  if(autoTimer){ clearInterval(autoTimer); autoTimer = null; }
  if(!el.autoDaily.checked) return;
  // check every 30s whether it's time and camera is active
  autoTimer = setInterval(()=>{
    if(!stream) return; // only run when webcam is active
    const target = (el.autoTime.value || '09:00').split(':');
    const hh = parseInt(target[0]||'9',10), mm = parseInt(target[1]||'0',10);
    const now = new Date();
    const lastISO = localStorage.getItem(LS_KEY_LAST);
    const last = lastISO ? new Date(lastISO) : null;

    // compute today's scheduled time
    const sched = new Date(now.getFullYear(), now.getMonth(), now.getDate(), hh, mm, 0, 0);

    // If it's after scheduled time and we haven't run today, run once
    const notRunToday = !last || last.toDateString() !== now.toDateString();
    if(now >= sched && notRunToday){
      snap(); // will mark last-run after success
    }
  }, 30000);
}

/* ====== GAS ====== */
async function sendGas(){
  const body = { adc:parseInt(el.adc.value||'0'), vref:parseFloat(el.vref.value||'3.3'),
                 rl:parseInt(el.rl.value||'10000'), r0:parseInt(el.r0.value||'10000'),
                 adc_max:4095 };
  setStatus('gas','working…');
  try{
    const r = await fetch('/gas',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    if(!r.ok){ alert('Gas send failed'); return; }
    await refresh(); await loadChart(true);
  } finally { setStatus('gas','idle'); }
}
function resetGas(){ el.adc.value="1800"; el.vref.value="3.3"; el.rl.value="10000"; el.r0.value="10000"; }
function preset(t){ if(t==='fresh'){ el.adc.value="700"; el.r0.value="12000"; } else { el.adc.value="2500"; el.r0.value="8000"; } }

/* ====== Summary / Polling ====== */
async function refresh(){
  const r = await fetch('/summary',{cache:'no-store'}); const s = await r.json();
  if(s.vision && s.vision.label){ updateVision(s.vision); }
  const g  = s.gas_ppm || {}, gf = s.gas_flags || {};
  el.gasBadges.innerHTML = [
    badge('CO₂ '+(g.co2??'—')+' ppm', gf.co2_high?'bad':'ok'),
    badge('NH₃ '+(g.nh3??'—')+' ppm', gf.nh3_high?'bad':'ok'),
    badge('VOC '+(g.alcohol??'—')+' eq', gf.voc_high?'warn':'ok')
  ].join(' ');
  el.decision.className = 'big '+(s.decision==='SPOILED'?'bad':'ok');
  el.decision.textContent = s.decision || '';
  el.raw.textContent = JSON.stringify(s, null, 2);
}
const badge = (t,c)=>'<span class="pill '+c+'">'+t+'</span>';
refresh(); setInterval(refresh, 2000);

/* ====== Chart ====== */
let gasChart=null;
function buildGradient(ctx, color){
  const g=ctx.createLinearGradient(0,0,0,ctx.canvas.height);
  g.addColorStop(0,  color + 'AA'); g.addColorStop(1, color + '00'); return g;
}
async function loadChart(forceFetch=false){
  try{
    const r = await fetch('/history',{cache:'no-store'}); const j = await r.json();
    const rows = Array.isArray(j.history)? j.history : [];
    el.chartEmpty.style.display = rows.length ? 'none':'flex';
    const labels = rows.map(h=> new Date(h.time).toLocaleTimeString());
    const co2=rows.map(h=>h.ppm?.co2??null), nh3=rows.map(h=>h.ppm?.nh3??null), benz=rows.map(h=>h.ppm?.benzene??null);
    const canvas = el.gasChart; const ctx = canvas.getContext('2d');
    const ds = [
      {label:'CO₂ (ppm)', data:co2, tension:.35, borderColor:'#22c55e', pointRadius:0, hitRadius:12, fill:true, backgroundColor:buildGradient(ctx,'#22c55e')},
      {label:'NH₃ (ppm)', data:nh3, tension:.35, borderColor:'#0ea5e9', pointRadius:0, hitRadius:12, fill:true, backgroundColor:buildGradient(ctx,'#0ea5e9')},
      {label:'Benzene (ppm)', data:benz, tension:.35, borderColor:'#f59e0b', pointRadius:0, hitRadius:12, fill:true, backgroundColor:buildGradient(ctx,'#f59e0b')}
    ];
    const options = { responsive:true, maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{position:'bottom',labels:{boxWidth:12,font:{weight:700}}}},
      scales:{x:{grid:{display:false}}, y:{beginAtZero:true, grid:{color:'rgba(0,0,0,.06)'}}},
      animation:{duration:350}
    };
    if(!gasChart){ canvas.style.height='280px'; gasChart = new Chart(ctx, {type:'line', data:{labels, datasets:ds}, options}); }
    else{ gasChart.data.labels=labels; gasChart.data.datasets[0].data=co2; gasChart.data.datasets[1].data=nh3; gasChart.data.datasets[2].data=benz; gasChart.update(); }
  }catch(_){}
}
loadChart(true); setInterval(()=>loadChart(true), 10000);

/* ====== boot ====== */
loadAutoSettings();
if(navigator.mediaDevices?.getUserMedia){
  listCams().catch(()=>{ /* ignore */ });
}
</script>
</body>
</html>
"""
