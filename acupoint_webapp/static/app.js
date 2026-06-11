// 통합 웹앱 프런트 (1b): 브라우저 온디바이스 CV + 증상 패널
// - MediaPipe HandLandmarker(WASM) 로 실시간 21 랜드마크
// - M2(StandardScaler+MLP) 를 순수 JS forward 로 포팅 -> 11혈 좌표 (core.py 와 동일 기하)
// - ① /recommend 결과의 has_cv_model 혈자리를 손 위에 강조

import { FilesetResolver, HandLandmarker }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const MP_VER = "0.10.14";
const WASM = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`;
const MODEL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// MediaPipe 손 토폴로지 (core.py 인덱스와 동일: WRIST0, INDEX_MCP5, MIDDLE_MCP9, PINKY_MCP17)
const WRIST = 0, INDEX_MCP = 5, MIDDLE_MCP = 9, PINKY_MCP = 17;
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],        [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],   [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],[0,17],
];

let M2 = null;            // 가중치
let handLandmarker = null;
let running = false;
let highlightCodes = new Set();   // ① 추천 중 손 위 강조할 경혈코드

const video = document.getElementById("cam");
const canvas = document.getElementById("overlay");
const ctx = canvas.getContext("2d");
const camMsg = document.getElementById("cam-msg");
const camStatus = document.getElementById("cam-status");

// ---------- M2 순수 JS forward ----------
function relu(v){ for(let i=0;i<v.length;i++) if(v[i]<0) v[i]=0; return v; }
function dense(x, W, b, act){
  const out = new Float64Array(b.length);
  for(let j=0;j<b.length;j++){
    let s = b[j];
    for(let i=0;i<x.length;i++) s += x[i]*W[i][j];
    out[j]=s;
  }
  return act==="relu" ? relu(out) : out;
}
// 53차원 입력(canon 42 + onehot 11) -> (cx,cy)
function m2forward(feat53){
  const sc = M2.scaler;
  const x = new Float64Array(feat53.length);
  for(let i=0;i<x.length;i++) x[i] = (feat53[i]-sc.mean[i])/sc.scale[i];
  let h = x;
  for(const L of M2.layers) h = dense(h, L.W, L.b, L.act);
  return h;  // length 2
}

// ---------- 기하 (core.py 와 동일) ----------
function computeAcupoints(lmPx){
  // lmPx: [{x,y}*21] 픽셀좌표
  const o = lmPx[WRIST];
  const r = {x: lmPx[MIDDLE_MCP].x-o.x, y: lmPx[MIDDLE_MCP].y-o.y};
  const s = Math.max(Math.hypot(r.x, r.y), 1e-6);
  const e1 = {x:r.x/s, y:r.y/s};
  let e2 = {x:-e1.y, y:e1.x};
  const vi = {x:lmPx[INDEX_MCP].x-o.x, y:lmPx[INDEX_MCP].y-o.y};
  const vp = {x:lmPx[PINKY_MCP].x-o.x, y:lmPx[PINKY_MCP].y-o.y};
  if(vi.x*vp.y - vi.y*vp.x < 0){ e2 = {x:-e2.x, y:-e2.y}; }   // chirality 정규화

  // canon 42 features
  const canon = new Float64Array(42);
  for(let i=0;i<21;i++){
    const dx = lmPx[i].x-o.x, dy = lmPx[i].y-o.y;
    canon[i*2]   = (dx*e1.x + dy*e1.y)/s;
    canon[i*2+1] = (dx*e2.x + dy*e2.y)/s;
  }
  const out = [];
  const n = M2.acups.length;
  for(let i=0;i<n;i++){
    const feat = new Float64Array(53);
    feat.set(canon, 0);
    feat[42+i] = 1;                       // onehot
    const [cx, cy] = m2forward(feat);
    const px = o.x + s*(cx*e1.x + cy*e2.x);
    const py = o.y + s*(cx*e1.y + cy*e2.y);
    const a = M2.acups[i], meta = M2.meta[a];
    out.push({name:a, name_kr:meta.name_kr, code:meta.code, x:px, y:py});
  }
  return out;
}

// ---------- 그리기 ----------
function draw(lmPx, acupoints){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  // 뼈대
  ctx.lineWidth = 2; ctx.strokeStyle = "rgba(120,170,255,.7)";
  for(const [a,b] of HAND_CONNECTIONS){
    ctx.beginPath(); ctx.moveTo(lmPx[a].x,lmPx[a].y); ctx.lineTo(lmPx[b].x,lmPx[b].y); ctx.stroke();
  }
  // 랜드마크 점
  ctx.fillStyle = "rgba(120,170,255,.9)";
  for(const p of lmPx){ ctx.beginPath(); ctx.arc(p.x,p.y,2.5,0,7); ctx.fill(); }
  // 혈자리
  for(const ap of acupoints){
    const hi = highlightCodes.has(ap.code);
    ctx.beginPath();
    ctx.arc(ap.x, ap.y, hi?9:5, 0, 7);
    ctx.fillStyle = hi ? "#33d6a6" : "rgba(255,255,255,.55)";
    ctx.fill();
    ctx.lineWidth = hi?3:1.5; ctx.strokeStyle = hi?"#0c3b2e":"rgba(0,0,0,.6)"; ctx.stroke();
    if(hi){
      ctx.font = "bold 13px system-ui"; ctx.fillStyle="#eafff6";
      ctx.strokeStyle="rgba(0,0,0,.85)"; ctx.lineWidth=3;
      const label = `${ap.name_kr} ${ap.code}`;
      ctx.strokeText(label, ap.x+11, ap.y+4);
      ctx.fillText(label, ap.x+11, ap.y+4);
    }
  }
}

// ---------- 루프 ----------
let lastTs = -1, loggedOnce = false;
function loop(){
  if(!running) return;
  if(video.readyState >= 2 && video.videoWidth > 0){
    if(canvas.width !== video.videoWidth){
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    }
    const ts = performance.now();
    if(ts !== lastTs){
      lastTs = ts;
      let res;
      try{
        res = handLandmarker.detectForVideo(video, ts);
      }catch(err){
        camStatus.textContent = "검출 오류: " + err.message;
        console.error("detectForVideo 실패:", err);
        requestAnimationFrame(loop); return;
      }
      const n = (res.landmarks && res.landmarks.length) ? 1 : 0;
      if(n){
        const lmPx = res.landmarks[0].map(p => ({x:p.x*canvas.width, y:p.y*canvas.height}));
        const acupoints = computeAcupoints(lmPx);
        draw(lmPx, acupoints);
        camMsg.style.display = "none";
        if(!loggedOnce){
          loggedOnce = true;
          console.log("✅ 손 검출됨. 캔버스", canvas.width+"x"+canvas.height,
                      "| 첫 혈자리", acupoints[0]);
        }
        camStatus.textContent = `실시간 인식 중 · 손 1 · 혈자리 ${acupoints.length}`;
      }else{
        ctx.clearRect(0,0,canvas.width,canvas.height);
        camMsg.textContent = "손이 보이지 않습니다. 손 전체가 화면에 들어오게, 밝은 곳에서 비춰주세요.";
        camMsg.style.display = "flex";
        camStatus.textContent = "실시간 인식 중 · 손 미검출";
      }
    }
  }
  requestAnimationFrame(loop);
}

// ---------- 초기화 ----------
async function initCV(){
  if(handLandmarker) return;
  camStatus.textContent = "모델 로딩 중…";
  if(!M2) M2 = await (await fetch("/m2_weights.json")).json();
  const vision = await FilesetResolver.forVisionTasks(WASM);
  const make = (delegate) => HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: "VIDEO", numHands: 1, minHandDetectionConfidence: 0.3,
  });
  try{
    handLandmarker = await make("GPU");
    console.log("HandLandmarker: GPU delegate");
  }catch(e){
    console.warn("GPU delegate 실패 → CPU 폴백:", e.message);
    handLandmarker = await make("CPU");
    console.log("HandLandmarker: CPU delegate");
  }
  camStatus.textContent = "";
}

async function startCamera(){
  const btn = document.getElementById("cam-btn");
  btn.disabled = true;
  try{
    await initCV();
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "environment", width:{ideal:960} }, audio:false });
    video.srcObject = stream;
    await video.play();
    running = true;
    btn.textContent = "카메라 중지";
    btn.disabled = false;
    camStatus.textContent = "실시간 인식 중";
    loop();
  }catch(e){
    camStatus.textContent = "카메라를 열 수 없습니다: " + e.message;
    camMsg.textContent = "카메라 권한이 필요합니다. (HTTPS 또는 localhost 에서만 동작)";
    btn.disabled = false;
  }
}

function stopCamera(){
  running = false;
  const s = video.srcObject;
  if(s) s.getTracks().forEach(t=>t.stop());
  video.srcObject = null;
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const btn = document.getElementById("cam-btn");
  btn.textContent = "카메라 시작";
  camStatus.textContent = "";
  camMsg.textContent = "‘카메라 시작’을 눌러 손을 비춰주세요.";
  camMsg.style.display = "flex";
}

document.getElementById("cam-btn").addEventListener("click", ()=>{
  running ? stopCamera() : startCamera();
});

// ---------- ① 증상 -> 혈자리 ----------
function renderResult(r){
  const el = document.getElementById("result");
  const disc = document.getElementById("disclaimer");
  disc.textContent = r.disclaimer || "";
  if(r.error){ el.innerHTML = `<div class="emergency"><h3>오류</h3><p>${r.error}</p></div>`; return; }

  if(r.red_flag){
    highlightCodes = new Set();
    el.innerHTML = `<div class="emergency"><h3>⚠️ 응급 가능성</h3>
      <p>${r.advice}</p></div>`;
    return;
  }
  const list = (r.acupoints||[]);
  highlightCodes = new Set(list.filter(a=>a.has_cv_model).map(a=>a.code));
  const cvCount = highlightCodes.size;

  let html = `<p class="advice">${r.advice}</p>`;
  if(list.length){
    html += `<div class="acup-list">` + list.map(a=>`
      <div class="acup">
        <span class="name">${a.acupoint||a.name_kr||""}</span>
        <span class="code">${a.code}</span>
        ${a.has_cv_model?'<span class="badge cv">손 위 표시</span>':''}
        <span class="for">${a.for_symptom||""}</span>
      </div>`).join("") + `</div>`;
    if(cvCount) html += `<p class="muted small" style="margin-top:10px">
      손 위 표시 가능한 혈자리 ${cvCount}개 — 오른쪽에서 카메라로 손을 비추면 강조됩니다.</p>`;
  }
  if(r.symptoms && r.symptoms.length)
    html += `<p class="muted small">인식된 증상: ${r.symptoms.join(", ")}</p>`;
  html += `<p class="provider">분석 모델: ${r.provider}${r.used_mock?" (mock·무과금)":""}</p>`;
  el.innerHTML = html;
}

async function recommend(){
  const btn = document.getElementById("recommend-btn");
  const text = document.getElementById("symptom-input").value.trim();
  if(!text){ document.getElementById("symptom-input").focus(); return; }
  const mock = document.getElementById("mock-chk").checked;
  btn.disabled = true; btn.textContent = "분석 중…";
  try{
    const res = await fetch("/recommend", {
      method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({text, mock}),
    });
    renderResult(await res.json());
  }catch(e){
    document.getElementById("result").innerHTML =
      `<div class="emergency"><h3>오류</h3><p>서버 연결 실패: ${e.message}</p></div>`;
  }finally{
    btn.disabled = false; btn.textContent = "혈자리 추천";
  }
}
document.getElementById("recommend-btn").addEventListener("click", recommend);
document.getElementById("symptom-input").addEventListener("keydown", e=>{
  if(e.key==="Enter" && (e.metaKey||e.ctrlKey)) recommend();
});

// ---------- ① 음성 입력(STT): MediaRecorder -> /transcribe(Whisper) -> 증상칸 ----------
// 브라우저별 컨테이너가 달라(크롬 webm/opus, iOS 사파리 mp4) 지원되는 첫 포맷을 고른다.
let mediaRecorder = null, audioChunks = [], recording = false;
const micBtn = document.getElementById("mic-btn");
const micStatus = document.getElementById("mic-status");
const setMicStatus = (t)=>{ micStatus.textContent = t || ""; };

function pickMime(){
  const cands = ["audio/webm;codecs=opus","audio/webm","audio/mp4","audio/ogg"];
  for(const m of cands) if(window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return "";
}
function extFor(type){
  return type.includes("mp4") ? "mp4" : type.includes("ogg") ? "ogg" : "webm";
}

async function toggleMic(){
  if(recording){ mediaRecorder && mediaRecorder.stop(); return; }      // 두 번째 클릭 = 종료
  if(!navigator.mediaDevices || !window.MediaRecorder){
    setMicStatus("이 브라우저는 음성 녹음을 지원하지 않습니다."); return;
  }
  let stream;
  try{
    stream = await navigator.mediaDevices.getUserMedia({ audio:true });
  }catch(e){
    setMicStatus("마이크를 열 수 없습니다: " + e.message + " (HTTPS/localhost 필요)"); return;
  }
  const mime = pickMime();
  mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined);
  audioChunks = [];
  mediaRecorder.ondataavailable = e=>{ if(e.data && e.data.size) audioChunks.push(e.data); };
  mediaRecorder.onstop = async ()=>{
    stream.getTracks().forEach(t=>t.stop());
    recording = false; micBtn.classList.remove("rec"); micBtn.textContent = "🎤 음성";
    const type = (mediaRecorder.mimeType || mime || "audio/webm");
    const blob = new Blob(audioChunks, { type });
    if(!blob.size){ setMicStatus("녹음된 음성이 없습니다."); return; }
    const fd = new FormData();
    fd.append("audio", blob, "voice." + extFor(type));
    micBtn.disabled = true; setMicStatus("음성 변환 중…");
    try{
      const res = await fetch("/transcribe", { method:"POST", body: fd });
      const j = await res.json();
      if(j.error){ setMicStatus(j.error); return; }
      const t = (j.text || "").trim();
      if(!t){ setMicStatus("인식된 음성이 없습니다. 다시 말씀해 주세요."); return; }
      const ta = document.getElementById("symptom-input");
      const prev = ta.value.trim();
      ta.value = prev ? (prev + " " + t) : t;     // 기존 입력 뒤에 이어붙임
      ta.focus();
      setMicStatus("");
    }catch(e){
      setMicStatus("변환 실패: " + e.message);
    }finally{
      micBtn.disabled = false;
    }
  };
  mediaRecorder.start();
  recording = true; micBtn.classList.add("rec"); micBtn.textContent = "⏹ 종료";
  setMicStatus("말씀하세요… (버튼을 다시 누르면 변환)");
}
micBtn.addEventListener("click", toggleMic);
