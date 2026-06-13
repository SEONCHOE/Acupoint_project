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
  ctx.lineWidth = 2; ctx.strokeStyle = "rgba(150,148,138,.78)";
  for(const [a,b] of HAND_CONNECTIONS){
    ctx.beginPath(); ctx.moveTo(lmPx[a].x,lmPx[a].y); ctx.lineTo(lmPx[b].x,lmPx[b].y); ctx.stroke();
  }
  // 랜드마크 점
  ctx.fillStyle = "rgba(150,148,138,.92)";
  for(const p of lmPx){ ctx.beginPath(); ctx.arc(p.x,p.y,2.5,0,7); ctx.fill(); }
  // 혈자리
  for(const ap of acupoints){
    const hi = highlightCodes.has(ap.code);
    ctx.beginPath();
    ctx.arc(ap.x, ap.y, hi?9:5, 0, 7);
    ctx.fillStyle = hi ? "#2f9d62" : "rgba(255,255,255,.6)";
    ctx.fill();
    ctx.lineWidth = hi?3:1.5; ctx.strokeStyle = hi?"#103d28":"rgba(0,0,0,.6)"; ctx.stroke();
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
// 혈자리 상세의 출처 외부 링크(KMCRIC 경혈 DB). 콘텐츠는 저작권상 임베드/복제하지 않고 링크만.
const KMCRIC_ACUPOINT_DB = "https://www.kmcric.com/database/acupoint";

// ----- 손 위치도(자체 제작): 표준 손 모형 21랜드마크에 M2로 11혈을 계산해 SVG로 그림 -----
// 외부 이미지/저작물 없이 우리 모델 출력만 렌더 → 라이선스 무관, 실시간 AR과 좌표 일치.
// MediaPipe 인덱스 순서(0 손목 … 4 엄지끝 … 20 새끼끝)의 정규화(0~1) 펼친 오른손 포즈.
const TEMPLATE_LANDMARKS = [
  [0.50,0.95],                                  // 0 wrist
  [0.36,0.86],[0.28,0.77],[0.23,0.69],[0.19,0.62], // 1-4 thumb
  [0.43,0.56],[0.41,0.43],[0.40,0.34],[0.39,0.26], // 5-8 index
  [0.52,0.54],[0.52,0.39],[0.52,0.29],[0.52,0.20], // 9-12 middle
  [0.61,0.56],[0.63,0.43],[0.64,0.34],[0.65,0.26], // 13-16 ring
  [0.70,0.60],[0.74,0.50],[0.76,0.43],[0.78,0.37], // 17-20 pinky
];

async function ensureM2(){
  if(!M2) M2 = await (await fetch("/m2_weights.json")).json();
  return M2;
}

// 혈자리별 손 면(해부학 기준): 손바닥(palmar) vs 손등(dorsal).
// 손바닥: 노궁(PC8)·어제(LU10)·태연(LU9). 나머지(합곡·중저·액문·후계, 손톱 옆 정혈)는 손등.
const ACU_SIDE = {
  LU9:"palmar", LU10:"palmar", PC8:"palmar",
  LI4:"dorsal", TE2:"dorsal", TE3:"dorsal", SI3:"dorsal",
  LI1:"dorsal", SI1:"dorsal", HT9:"dorsal", TE1:"dorsal",
};

const HM_CHAINS = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]];
const HM_NAIL_TIPS = [4,8,12,16,20];

// 닫힌 점들을 Catmull-Rom 으로 부드럽게 잇는 path d (손바닥 윤곽을 매끄럽게).
function smoothClosed(pts){
  const n = pts.length;
  let d = `M${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for(let i=0;i<n;i++){
    const p0=pts[(i-1+n)%n], p1=pts[i], p2=pts[(i+1)%n], p3=pts[(i+2)%n];
    const c1x=p1.x+(p2.x-p0.x)/6, c1y=p1.y+(p2.y-p0.y)/6;
    const c2x=p2.x-(p3.x-p1.x)/6, c2y=p2.y-(p3.y-p1.y)/6;
    d += `C${c1x.toFixed(1)} ${c1y.toFixed(1)} ${c2x.toFixed(1)} ${c2y.toFixed(1)} ${p2.x.toFixed(1)} ${p2.y.toFixed(1)}`;
  }
  return d + "Z";
}

const HM_PALM_INFLATE = 1.09;   // 손바닥을 중심에서 바깥으로 확장(도톰)
const HM_PALM_FAT = 11;         // 손바닥 실루엣을 손가락보다 더 굵게

// 손바닥 둘레점을 중심에서 바깥으로 확장 → 도톰한 손바닥.
function inflatePalm(pts){
  const cx = pts.reduce((s,p)=>s+p.x, 0) / pts.length;
  const cy = pts.reduce((s,p)=>s+p.y, 0) / pts.length;
  return pts.map(p => ({ x:cx+(p.x-cx)*HM_PALM_INFLATE, y:cy+(p.y-cy)*HM_PALM_INFLATE }));
}

// 손바닥 둘레점. 새끼관절~손목 사이에 소지구(hypothenar) 융기점 2개를 끼워
// 척측(새끼쪽) 가장자리를 부드럽게 부풀린다. 융기는 후계(SI3) 좌표 바깥으로 잡아 확실히 감싼다.
function palmOutline(lm, ulnar){
  let hiHypo, loHypo;
  if(ulnar){                                   // SI3 기준: 바깥(+x)으로 충분히 밀어 감쌈
    hiHypo = { x: ulnar.x + 12, y: ulnar.y - 14 };  // 새끼관절 쪽(위)
    loHypo = { x: ulnar.x + 15, y: ulnar.y + 12 };  // 손목 쪽(아래)
  }else{
    hiHypo = { x: lm[17].x + (lm[17].x-lm[13].x)*0.6, y: lm[17].y + (lm[0].y-lm[17].y)*0.22 };
    loHypo = { x: lm[17].x + (lm[17].x-lm[13].x)*0.72, y: lm[17].y + (lm[0].y-lm[17].y)*0.46 };
  }
  return [lm[0], lm[1], lm[5], lm[9], lm[13], lm[17], hiHypo, loHypo];
}

// 손 실루엣 한 겹(손바닥 베지에 blob + 손가락/엄지 둥근 캡슐). kind: "edge"(테두리)|"fill"(살).
// 손바닥은 손가락보다 더 굵게(+FAT) 그려 도톰하게. palm 은 미리 계산한 둘레점(inflate 포함).
function handSilhouette(palm, lm, w, kind){
  let s = `<path class="hm-${kind} hm-blob" d="${smoothClosed(palm)}" stroke-width="${w + HM_PALM_FAT}"/>`;
  for(const ch of HM_CHAINS){
    const d = "M" + ch.map(i => `${lm[i].x.toFixed(1)} ${lm[i].y.toFixed(1)}`).join(" L");
    s += `<path class="hm-${kind} hm-line" d="${d}" stroke-width="${w}"/>`;
  }
  return s;
}

// 손등용 손톱: 손가락 끝마디 축에 맞춘 작은 타원.
function handNails(lm){
  let s = "";
  for(const i of HM_NAIL_TIPS){
    const tip = lm[i], prev = lm[i-1];
    const cx = tip.x*0.62 + prev.x*0.38, cy = tip.y*0.62 + prev.y*0.38;
    const ang = Math.atan2(tip.y-prev.y, tip.x-prev.x) * 180/Math.PI + 90;
    s += `<ellipse class="hm-nail" cx="${cx.toFixed(1)}" cy="${cy.toFixed(1)}" rx="4.6" ry="6.2"`
       + ` transform="rotate(${ang.toFixed(1)} ${cx.toFixed(1)} ${cy.toFixed(1)})"/>`;
  }
  return s;
}

// code(예: "LU10") 혈자리를 강조한 손 위치도 SVG 문자열. M2 가 로드돼 있어야 함.
function buildHandMap(code){
  const W = 200, H = 210, FW = 16;     // FW: 손가락 굵기
  const lm = TEMPLATE_LANDMARKS.map(([x,y]) => ({ x:x*W, y:y*H }));
  const aps = computeAcupoints(lm);
  const side = ACU_SIDE[code] || "palmar";
  const sideKr = side === "dorsal" ? "손등" : "손바닥";
  const si3 = aps.find(a => a.code === "SI3");      // 척측 경계혈 → 소지구 융기 기준점
  const palm = inflatePalm(palmOutline(lm, si3));
  const skin = handSilhouette(palm, lm, FW + 3, "edge") + handSilhouette(palm, lm, FW, "fill");
  const nailLayer = side === "dorsal" ? handNails(lm) : "";
  let acu = "", lab = "";
  for(const ap of aps){
    if((ACU_SIDE[ap.code] || "palmar") !== side) continue;   // 같은 면의 혈자리만 표시
    const hi = ap.code === code;
    acu += `<circle class="acu${hi?' hi':''}" cx="${ap.x.toFixed(1)}" cy="${ap.y.toFixed(1)}" r="${hi?5:3}"/>`;
    if(hi){
      const right = ap.x > W/2;
      lab = `<text class="lab" x="${(right?ap.x-8:ap.x+8).toFixed(1)}" y="${(ap.y-8).toFixed(1)}"
        text-anchor="${right?'end':'start'}">${ap.name_kr} ${ap.code}</text>`;
    }
  }
  const badge = `<text class="hm-side" x="10" y="17">${sideKr} 면</text>`;
  return `<svg viewBox="0 0 ${W} ${H + 20}" class="handmap-svg" role="img" aria-label="${code} ${sideKr} 개략 위치">`
    + skin + nailLayer + acu + lab + badge + `</svg>`
    + `<p class="handmap-cap">손 <b>${sideKr}</b> 쪽 개략 위치예요(실시간 AR과 동일 모델). 정확한 취혈은 전문가 확인이 필요합니다.</p>`;
}

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

  const SEV = { low:"가벼운 정도", medium:"중간 정도", high:"심한 정도", emergency:"응급" };
  const sevLabel = SEV[r.severity] || "보통";

  let html = "";
  if(list.length){
    html += `<p class="result-intro">말씀해주신 증상은 <strong>${sevLabel}</strong>(으)로
      평가됩니다. 본 증상을 관리할 수 있는 아래 혈자리를 추천드립니다.</p>`;
  }
  html += `<p class="advice">${r.advice}</p>`;
  if(list.length){
    html += `<div class="acup-list">` + list.map((a, i)=>{
      const name = a.acupoint || a.name_kr || "";
      const treats = (a.treats || []).filter(Boolean).join(", ");
      const rows = [];
      if(a.position) rows.push(`<div class="d-row"><dt>위치</dt><dd>${a.position}</dd></div>`);
      if(a.meridian) rows.push(`<div class="d-row"><dt>경락</dt><dd>${a.meridian}</dd></div>`);
      if(treats)     rows.push(`<div class="d-row"><dt>주치</dt><dd>${treats}</dd></div>`);
      rows.push(`<div class="d-row"><dt>코드</dt><dd class="mono">${a.code}</dd></div>`);
      // 출처: KMCRIC 경혈 DB(저작권 All Rights Reserved) — 크롤링 대신 해당 DB로 외부 링크
      const mer = a.meridian ? `${a.meridian}(${a.code.replace(/[0-9]+$/,"")}) ` : "";
      rows.push(`<div class="d-row"><dt>자세히</dt><dd>
        <a class="ext" href="${KMCRIC_ACUPOINT_DB}" target="_blank" rel="noopener noreferrer"
          >KMCRIC 경혈 DB에서 ${mer}«${name}» 보기 ↗</a></dd></div>`);
      // 손 위치도는 CV 모델이 있는(11혈) 혈자리만 — 클릭 시 지연 렌더
      const mapBox = a.has_cv_model
        ? `<div class="hand-map" data-code="${a.code}" aria-label="${name} 손 위 개략 위치"></div>` : "";
      return `<div class="acup-item">
        <div class="acup" role="button" tabindex="0" aria-expanded="false" data-i="${i}">
          <span class="name">${name}</span>
          <span class="code">${a.code}</span>
          ${a.has_cv_model?'<span class="badge cv">손 위 표시</span>':''}
          <span class="for">${a.for_symptom||""}</span>
          <span class="chev" aria-hidden="true">▾</span>
        </div>
        <div class="acup-detail" hidden>${mapBox}<dl>${rows.join("")}</dl></div>
      </div>`;
    }).join("") + `</div>`;
    if(cvCount) html += `<p class="muted small" style="margin-top:10px">
      손 위 표시 가능한 혈자리 ${cvCount}개 — 카메라로 손을 비추면 강조됩니다. 각 혈자리를 누르면 상세가 펼쳐져요.</p>`;
  }
  if(r.symptoms && r.symptoms.length)
    html += `<p class="muted small">인식된 증상: ${r.symptoms.join(", ")}</p>`;
  html += `<p class="provider">분석 모델: ${r.provider}${r.used_mock?" (mock·무과금)":""}</p>`;
  el.innerHTML = html;

  // 혈자리 클릭/키보드 -> 위치·경락·주치 상세 토글 + 손 위치도 지연 렌더
  el.querySelectorAll(".acup[data-i]").forEach(row=>{
    const toggle = async ()=>{
      const item = row.closest(".acup-item");
      const det = item.querySelector(".acup-detail");
      const open = item.classList.toggle("open");
      row.setAttribute("aria-expanded", open ? "true" : "false");
      det.hidden = !open;
      if(!open) return;
      const map = det.querySelector(".hand-map[data-code]");
      if(map && !map.dataset.done){
        map.dataset.done = "1";
        map.innerHTML = `<span class="handmap-load">손 위치도 불러오는 중…</span>`;
        try{ await ensureM2(); map.innerHTML = buildHandMap(map.dataset.code); }
        catch(e){ map.innerHTML = `<span class="handmap-load">위치도를 불러오지 못했습니다.</span>`; }
      }
    };
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", e=>{
      if(e.key==="Enter" || e.key===" "){ e.preventDefault(); toggle(); }
    });
  });
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
    btn.disabled = false; btn.textContent = "혈자리 찾기";
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
