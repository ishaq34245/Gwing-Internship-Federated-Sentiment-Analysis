# app.py
from flask import Flask, request, jsonify, render_template_string
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import numpy as np

# Map numeric labels to human-readable names
LABEL_MAP = {
    0: "Very Negative",
    1: "Negative",
    2: "Positive",
    3: "Very Positive",
}

# Prefer the saved final_model folder in checkpoints
MODEL_DIR = os.path.join("checkpoints", "final_model")
from src.utils import MODEL_NAME

# For serving, CPU is fine (and avoids CUDA issues)
device = torch.device("cpu")
print(f"[APP] Using device: {device}")

app = Flask(__name__)

# -----------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------
if os.path.isdir(MODEL_DIR):
    load_dir = MODEL_DIR
else:
    print(f"[APP] checkpoints/final_model not found, loading base model '{MODEL_NAME}'")
    load_dir = MODEL_NAME

print(f"[APP] Loading from: {load_dir}")

# Load model
model = AutoModelForSequenceClassification.from_pretrained(load_dir)
model.to(device)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(load_dir)


# -----------------------------
# COMPLETELY NEW WEB UI
# -----------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Sentiment Lab · Text Inspector</title>
  <style>
    :root {
      --bg: #f3f4f6;
      --panel: #ffffff;
      --accent: #2563eb;
      --accent-soft: #dbeafe;
      --border: #e5e7eb;
      --text-main: #111827;
      --text-muted: #6b7280;
      --danger: #b91c1c;
      --negative: #f97373;
      --positive: #22c55e;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: radial-gradient(circle at top left, #e0f2fe 0, #f3f4f6 45%, #eef2ff 100%);
      color: var(--text-main);
      display: flex;
      align-items: stretch;
      justify-content: center;
    }

    .shell {
      width: 100%;
      max-width: 1120px;
      margin: 32px auto;
      display: grid;
      grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr);
      gap: 24px;
      padding: 0 16px;
    }

    @media (max-width: 900px) {
      .shell {
        grid-template-columns: 1fr;
        margin-top: 16px;
      }
    }

    .panel {
      background: var(--panel);
      border-radius: 18px;
      border: 1px solid var(--border);
      box-shadow:
        0 18px 40px rgba(15, 23, 42, 0.08),
        0 0 0 1px rgba(148, 163, 184, 0.15);
      padding: 20px 22px;
    }

    .panel-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 14px;
    }

    .title-block h1 {
      margin: 0;
      font-size: 1.45rem;
      letter-spacing: 0.02em;
    }

    .title-block p {
      margin: 4px 0 0 0;
      font-size: 0.85rem;
      color: var(--text-muted);
    }

    .badge-version {
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      border: 1px solid #bfdbfe;
    }

    .input-label {
      font-size: 0.8rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--text-muted);
      margin-bottom: 6px;
    }

    textarea {
      width: 100%;
      min-height: 120px;
      border-radius: 12px;
      border: 1px solid var(--border);
      background: #f9fafb;
      color: var(--text-main);
      padding: 10px 12px;
      font-size: 0.95rem;
      resize: vertical;
      transition: border-color 0.15s ease, box-shadow 0.15s ease, background 0.15s;
    }

    textarea::placeholder {
      color: #9ca3af;
    }

    textarea:focus {
      outline: none;
      border-color: var(--accent);
      background: #ffffff;
      box-shadow: 0 0 0 1px var(--accent-soft);
    }

    .controls-row {
      margin-top: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      flex-wrap: wrap;
    }

    .btn-primary {
      padding: 9px 18px;
      border-radius: 999px;
      border: none;
      background: var(--accent);
      color: white;
      font-size: 0.92rem;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 12px 24px rgba(37, 99, 235, 0.35);
      transition: background 0.15s ease, transform 0.1s ease, box-shadow 0.15s;
    }

    .btn-primary span.icon {
      font-size: 1.05rem;
    }

    .btn-primary:hover {
      background: #1d4ed8;
      transform: translateY(-1px);
      box-shadow: 0 18px 30px rgba(37, 99, 235, 0.4);
    }

    .btn-primary:active {
      transform: translateY(0);
      box-shadow: 0 10px 18px rgba(37, 99, 235, 0.3);
    }

    .btn-primary:disabled {
      cursor: wait;
      opacity: 0.7;
      transform: none;
      box-shadow: 0 0 0 rgba(0,0,0,0);
    }

    .hint-text {
      font-size: 0.8rem;
      color: var(--text-muted);
    }

    .samples {
      margin-top: 14px;
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }

    .sample-pill {
      font-size: 0.8rem;
      padding: 5px 10px;
      border-radius: 999px;
      border: 1px dashed #cbd5f5;
      background: #eff6ff;
      color: #1d4ed8;
      cursor: pointer;
      transition: background 0.15s ease, border-color 0.15s ease;
    }

    .sample-pill:hover {
      background: #e0ecff;
      border-color: #93c5fd;
    }

    /* Right side: results panel */

    .result-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 6px;
    }

    .result-title {
      font-size: 0.95rem;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--text-muted);
    }

    .pill-main {
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.8rem;
      border: 1px solid var(--border);
      background: #f9fafb;
      color: var(--text-muted);
    }

    .pill-main.ok {
      border-color: #bbf7d0;
      background: #ecfdf5;
      color: #166534;
    }

    .pill-main.empty {
      border-style: dashed;
    }

    #result {
      margin-top: 10px;
      border-radius: 14px;
      border: 1px dashed var(--border);
      padding: 10px 12px;
      background: #f9fafb;
      font-size: 0.9rem;
      display: none;
    }

    .result-section + .result-section {
      margin-top: 8px;
      padding-top: 8px;
      border-top: 1px solid #e5e7eb;
    }

    .label-chip {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 10px;
      border-radius: 999px;
      font-size: 0.85rem;
      border: 1px solid #e5e7eb;
    }

    .label-chip-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #9ca3af;
    }

    .label-chip.very-negative .label-chip-dot {
      background: #ef4444;
    }
    .label-chip.negative .label-chip-dot {
      background: #f97316;
    }
    .label-chip.positive .label-chip-dot {
      background: #22c55e;
    }
    .label-chip.very-positive .label-chip-dot {
      background: #0ea5e9;
    }

    .confidence-value {
      font-weight: 600;
    }

    .prob-list {
      margin: 4px 0 0 0;
      padding: 0;
      list-style: none;
    }

    .prob-item {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      padding: 1px 0;
    }

    .prob-item span.label {
      color: var(--text-muted);
    }

    .prob-item span.value {
      font-variant-numeric: tabular-nums;
    }

    .side-footer {
      margin-top: 16px;
      font-size: 0.78rem;
      color: var(--text-muted);
      display: flex;
      flex-direction: column;
      gap: 2px;
    }

    code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      font-size: 0.8rem;
      padding: 2px 4px;
      border-radius: 4px;
      background: #e5e7eb;
    }
  </style>
</head>
<body>
  <div class="shell">
    <!-- Left: Input panel -->
    <section class="panel">
      <div class="panel-header">
        <div class="title-block">
          <h1>Sentiment Lab</h1>
          <p>Inspect any sentence and see how the model scores its sentiment.</p>
        </div>
        <div class="badge-version">Federated model · v1</div>
      </div>

      <div>
        <div class="input-label">Enter text</div>
        <textarea id="input-text" placeholder="I didn’t expect to like this movie, but it was actually really good."></textarea>
      </div>

      <div class="controls-row">
        <button class="btn-primary" id="predict-btn">
          <span class="icon">⚙️</span>
          <span>Run sentiment analysis</span>
        </button>
        <span class="hint-text">Tip: click one of the samples below to try quickly.</span>
      </div>

      <div class="samples">
        <button type="button" class="sample-pill">This is the worst experience I’ve ever had.</button>
        <button type="button" class="sample-pill">I love how simple and useful this tool is.</button>
        <button type="button" class="sample-pill">It’s okay, not amazing but not terrible either.</button>
        <button type="button" class="sample-pill">The update completely broke my workflow.</button>
      </div>
    </section>

    <!-- Right: Result panel -->
    <section class="panel">
      <div class="result-header">
        <div class="result-title">Analysis result</div>
        <div id="status-pill" class="pill-main empty">Awaiting input</div>
      </div>

      <div id="result"></div>

      <div class="side-footer">
        
      </div>
    </section>
  </div>

  <script>
    const btn = document.getElementById("predict-btn");
    const input = document.getElementById("input-text");
    const resultBox = document.getElementById("result");
    const statusPill = document.getElementById("status-pill");
    const sampleButtons = document.querySelectorAll(".sample-pill");

    const labelNames = [
      "Very Negative",
      "Negative",
      "Positive",
      "Very Positive"
    ];

    function labelCssClass(name) {
      const lower = name.toLowerCase();
      if (lower.includes("very negative")) return "very-negative";
      if (lower.includes("negative")) return "negative";
      if (lower.includes("very positive")) return "very-positive";
      if (lower.includes("positive")) return "positive";
      return "";
    }

    sampleButtons.forEach(btnSample => {
      btnSample.addEventListener("click", () => {
        input.value = btnSample.textContent;
        input.focus();
      });
    });

    btn.addEventListener("click", async () => {
      const text = input.value.trim();
      if (!text) {
        alert("Please type a sentence first.");
        return;
      }

      btn.disabled = true;
      btn.innerHTML = '<span class="icon">⏱</span><span>Running...</span>';
      statusPill.textContent = "Running analysis…";
      statusPill.classList.remove("empty");
      statusPill.classList.remove("ok");

      try {
        const response = await fetch("/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text })
        });

        if (!response.ok) {
          throw new Error("Server error: " + response.status);
        }

        const data = await response.json();
        const pred = data.prediction;
        const labelName = data.label_name || labelNames[pred] || ("Class " + pred);
        const conf = data.confidence;
        const probs = data.probabilities || [];
        const nClasses = data.n_classes || probs.length;

        // Build result HTML
        const cssClass = labelCssClass(labelName);
        statusPill.textContent = labelName;
        statusPill.classList.add("ok");

        let html = "";

        html += '<div class="result-section">';
        html += `<div class="label-chip ${cssClass}">`;
        html += '<span class="label-chip-dot"></span>';
        html += `<span>${labelName}</span>`;
        html += "</div>";
        html += ` <span class="confidence-value"> · ${(conf * 100).toFixed(2)}% confidence</span>`;
        html += "</div>";

        if (probs.length > 0) {
          html += '<div class="result-section">';
          html += "<div style='font-size:0.8rem; text-transform:uppercase; letter-spacing:0.08em; color:#6b7280; margin-bottom:4px;'>Class breakdown</div>";
          html += "<ul class='prob-list'>";
          for (let i = 0; i < nClasses; i++) {
            const p = probs[i] || 0;
            const name = labelNames[i] || ("Class " + i);
            html += "<li class='prob-item'>";
            html += `<span class='label'>${name}</span>`;
            html += `<span class='value'>${(p * 100).toFixed(2)}%</span>`;
            html += "</li>";
          }
          html += "</ul>";
          html += "</div>";
        }

        resultBox.style.display = "block";
        resultBox.innerHTML = html;
      } catch (err) {
        console.error(err);
        statusPill.textContent = "Error";
        resultBox.style.display = "block";
        resultBox.textContent = "Error: " + err.message;
      } finally {
        btn.disabled = false;
        btn.innerHTML = '<span class="icon">⚙️</span><span>Run sentiment analysis</span>';
      }
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE)


# -----------------------------
# API ENDPOINT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "JSON should contain a 'text' field"}), 400

    text = data["text"]

    # Tokenize
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        confidence = float(max(probs))
        n_classes = int(model.config.num_labels)

    return jsonify({
        "prediction": pred,
        "label_name": LABEL_MAP.get(pred, f"Class {pred}"),
        "confidence": confidence,
        "probabilities": probs.tolist(),
        "n_classes": n_classes
    })


# -----------------------------
# RUN SERVER
# -----------------------------
if __name__ == "__main__":
    # Visit http://127.0.0.1:5000/ in the browser
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
