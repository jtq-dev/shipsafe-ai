async function postJson(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body)
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const fr = new FileReader();
    fr.onload = () => resolve(fr.result);
    fr.onerror = reject;
    fr.readAsDataURL(file);
  });
}

document.getElementById("predictBtn").onclick = async () => {
  const f = document.getElementById("img").files?.[0];
  const out = document.getElementById("predOut");
  if (!f) return (out.textContent = "Pick an image first.");
  out.textContent = "Predicting...";
  try {
    const b64 = await fileToDataUrl(f);
    const res = await postJson("/predict", { image_base64: b64 });
    out.innerHTML = `<div class="font-semibold">Prediction: ${res.prediction}</div>
                     <div class="text-zinc-600 text-xs mt-1">Probabilities: ${res.probabilities.map(x=>x.toFixed(3)).join(", ")}</div>`;
  } catch (e) {
    out.textContent = "Error: " + e.message;
  }
};

document.getElementById("qaBtn").onclick = async () => {
  const q = document.getElementById("q").value.trim();
  const useLLM = document.getElementById("useLLM").checked;
  const out = document.getElementById("qaOut");
  if (!q) return (out.textContent = "Ask a question first.");
  out.textContent = "Thinking...";
  try {
    const res = await postJson("/qa", { question: q, top_k: 4, use_llm: useLLM });
    const chunks = res.chunks.map(c =>
      `<div class="mt-2 p-3 bg-zinc-50 rounded-xl border">
         <div class="text-xs text-zinc-500">${c.source} • score=${c.score.toFixed(3)}</div>
         <div class="text-sm mt-1 whitespace-pre-wrap">${c.text}</div>
       </div>`).join("");
    out.innerHTML = `<div class="font-semibold">Answer</div>
                     <div class="mt-1 whitespace-pre-wrap">${res.answer}</div>
                     <div class="font-semibold mt-4">Sources</div>${chunks}`;
  } catch (e) {
    out.textContent = "Error: " + e.message;
  }
};
