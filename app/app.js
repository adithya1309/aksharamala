const PAD = document.getElementById("pad");
const ctx = PAD.getContext("2d", { willReadFrequently: true });
const PEN = 12;
const ERASER = 34;
// training images are cropped to the ink and resized to 32x32, mean ink 21.5% of
// the frame. If a downsampled stroke lands much thinner than that it is a canvas
// artifact, not handwriting, so thicken it once before the model sees it.
const MIN_INK = 0.12;

let labels = null;
let session = null;
let target = null;
let tool = "pencil";
let covered = false;
let drew = false;
let drawing = false;

function clearPad() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, PAD.width, PAD.height);
  drew = false;
  document.getElementById("submit").disabled = true;
}

function pos(e) {
  const r = PAD.getBoundingClientRect();
  return [(e.clientX - r.left) * PAD.width / r.width, (e.clientY - r.top) * PAD.height / r.height];
}

PAD.addEventListener("pointerdown", (e) => {
  PAD.setPointerCapture(e.pointerId);
  drawing = true;
  ctx.strokeStyle = tool === "pencil" ? "#000" : "#fff";
  ctx.lineWidth = tool === "pencil" ? PEN : ERASER;
  ctx.lineCap = ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(...pos(e));
});

PAD.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  ctx.lineTo(...pos(e));
  ctx.stroke();
  drew = true;
  document.getElementById("submit").disabled = !session;
});

for (const ev of ["pointerup", "pointercancel", "pointerleave"]) {
  PAD.addEventListener(ev, () => { drawing = false; });
}

// train/serve parity: crop to the ink bbox, resize to 32x32, binarize, ink = 1
function preprocess() {
  const { data, width, height } = ctx.getImageData(0, 0, PAD.width, PAD.height);
  let x0 = width, y0 = height, x1 = -1, y1 = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4] < 128) {
        if (x < x0) x0 = x;
        if (x > x1) x1 = x;
        if (y < y0) y0 = y;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) return null;

  const small = document.createElement("canvas");
  small.width = small.height = 32;
  const sctx = small.getContext("2d", { willReadFrequently: true });
  sctx.fillStyle = "#fff";
  sctx.fillRect(0, 0, 32, 32);
  // the dataset stretches the segmented character to 32x32 without keeping aspect
  sctx.drawImage(PAD, x0, y0, x1 - x0 + 1, y1 - y0 + 1, 0, 0, 32, 32);

  const px = sctx.getImageData(0, 0, 32, 32).data;
  let ink = new Float32Array(1024);
  for (let i = 0; i < 1024; i++) ink[i] = px[i * 4] < 128 ? 1 : 0;

  let coverage = ink.reduce((a, b) => a + b, 0) / 1024;
  if (coverage < MIN_INK) ink = dilate(ink);
  return ink;
}

function dilate(src) {
  const out = new Float32Array(1024);
  for (let y = 0; y < 32; y++) {
    for (let x = 0; x < 32; x++) {
      let on = 0;
      for (let dy = -1; dy <= 1 && !on; dy++) {
        for (let dx = -1; dx <= 1 && !on; dx++) {
          const ny = y + dy, nx = x + dx;
          if (ny >= 0 && ny < 32 && nx >= 0 && nx < 32 && src[ny * 32 + nx]) on = 1;
        }
      }
      out[y * 32 + x] = on;
    }
  }
  return out;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exp = logits.map((v) => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((v) => v / sum);
}

async function predict() {
  const ink = preprocess();
  if (!ink) return null;
  const out = await session.run({ input: new ort.Tensor("float32", ink, [1, 1, 32, 32]) });
  const probs = softmax(Array.from(out.logits.data));
  return probs
    .map((p, i) => ({ index: i, p }))
    .sort((a, b) => b.p - a.p)
    .slice(0, 3);
}

// look-cover-write-check: the glyph is hidden while writing, then revealed with
// the verdict so the student compares their stroke against the model answer
function setCovered(on) {
  covered = on;
  const el = document.getElementById("target");
  el.textContent = on ? "?" : target.display;
  el.classList.toggle("covered", on);
  document.getElementById("cover").textContent = on ? "Reveal" : "Cover";
}

function newQuestion() {
  target = labels[Math.floor(Math.random() * labels.length)];
  document.getElementById("translit").textContent = target.transliteration;
  setCovered(document.getElementById("autocover").checked);
  document.getElementById("verdict").hidden = true;
  document.getElementById("next").hidden = true;
  document.getElementById("submit").hidden = false;
  document.getElementById("skip").hidden = false;
  clearPad();
}

function showVerdict(top) {
  setCovered(false);  // checking is the point of the exercise
  const right = top[0].index === target.index;
  const headline = document.getElementById("headline");
  headline.className = right ? "right" : "wrong";
  headline.textContent = right
    ? `Correct - ${target.display} (${(top[0].p * 100).toFixed(1)}% confident)`
    : `Not quite. That looks like ${labels[top[0].index].display} ${labels[top[0].index].transliteration} (${(top[0].p * 100).toFixed(1)}%)`;
  document.getElementById("alternatives").innerHTML = top
    .slice(1)
    .map((t) => `<li>${labels[t.index].display} ${labels[t.index].transliteration} - ${(t.p * 100).toFixed(1)}%</li>`)
    .join("");
  document.getElementById("verdict").hidden = false;
  document.getElementById("next").hidden = false;
  document.getElementById("submit").hidden = true;
  document.getElementById("skip").hidden = true;
}

document.getElementById("cover").onclick = () => setCovered(!covered);
document.getElementById("autocover").onchange = (e) => {
  localStorage.setItem("autocover", e.target.checked ? "1" : "");
  setCovered(e.target.checked);
};
document.getElementById("pencil").onclick = () => setTool("pencil");
document.getElementById("eraser").onclick = () => setTool("eraser");
document.getElementById("reset").onclick = clearPad;
document.getElementById("skip").onclick = newQuestion;
document.getElementById("next").onclick = newQuestion;
document.getElementById("submit").onclick = async () => {
  if (!drew) return;
  const top = await predict();
  if (top) showVerdict(top);
};

function setTool(name) {
  tool = name;
  document.getElementById("pencil").classList.toggle("active", name === "pencil");
  document.getElementById("eraser").classList.toggle("active", name === "eraser");
}

async function boot() {
  clearPad();
  try {
    document.getElementById("autocover").checked = localStorage.getItem("autocover") === "1";
  } catch {}  // private windows can throw on storage access
  try {
    const res = await fetch("labels.json");
    if (!res.ok) throw new Error(`labels.json: ${res.status}`);
    labels = await res.json();
    if (labels.length !== 85) throw new Error(`expected 85 labels, got ${labels.length}`);
    session = await ort.InferenceSession.create("model.onnx");
  } catch (err) {
    document.getElementById("translit").textContent = `failed to load: ${err.message}`;
    throw err;
  }
  newQuestion();
}

boot();
