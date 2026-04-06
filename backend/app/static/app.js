const form = document.getElementById("upload-form");
const fileInput = document.getElementById("wav-file");
const statusEl = document.getElementById("status");
const processBtn = document.getElementById("process-btn");
const resultEl = document.getElementById("result");
const sliceCountEl = document.getElementById("slice-count");
const filenamesEl = document.getElementById("filenames");
const downloadLink = document.getElementById("download-link");
const waveformCanvas = document.getElementById("waveform");
const canvasCtx = waveformCanvas.getContext("2d");

let currentObjectUrl = null;

function setStatus(message, isError = false) {
  statusEl.textContent = message;
  statusEl.style.color = isError ? "#ff8a8a" : "#d8defa";
}

function resetResult() {
  resultEl.classList.add("hidden");
  sliceCountEl.textContent = "";
  filenamesEl.textContent = "";
  if (currentObjectUrl) {
    URL.revokeObjectURL(currentObjectUrl);
    currentObjectUrl = null;
  }
}

function drawWaveform(audioBuffer) {
  const width = waveformCanvas.width;
  const height = waveformCanvas.height;
  canvasCtx.clearRect(0, 0, width, height);
  canvasCtx.fillStyle = "#0d111a";
  canvasCtx.fillRect(0, 0, width, height);

  const channelData = audioBuffer.getChannelData(0);
  const step = Math.ceil(channelData.length / width);
  const amp = height / 2;

  canvasCtx.lineWidth = 1;
  canvasCtx.strokeStyle = "#9cb4ff";
  canvasCtx.beginPath();
  for (let i = 0; i < width; i += 1) {
    const start = i * step;
    const end = Math.min(start + step, channelData.length);
    let min = 1.0;
    let max = -1.0;
    for (let j = start; j < end; j += 1) {
      const val = channelData[j];
      if (val < min) min = val;
      if (val > max) max = val;
    }
    canvasCtx.moveTo(i, (1 + min) * amp);
    canvasCtx.lineTo(i, (1 + max) * amp);
  }
  canvasCtx.stroke();
}

async function previewWaveform(file) {
  try {
    const arrayBuffer = await file.arrayBuffer();
    const audioCtx = new AudioContext();
    const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer.slice(0));
    drawWaveform(audioBuffer);
    audioCtx.close();
  } catch (_) {
    setStatus("Unable to draw waveform preview, but processing still works.");
  }
}

fileInput.addEventListener("change", async () => {
  resetResult();
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Choose a WAV file to begin.");
    return;
  }
  setStatus(`Loaded ${file.name}. Ready to process.`);
  await previewWaveform(file);
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resetResult();
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setStatus("Please select a WAV file first.", true);
    return;
  }

  const formData = new FormData();
  formData.append("file", file);

  const settings = new FormData(form);
  for (const [key, value] of settings.entries()) {
    if (key === "normalize") {
      formData.append(key, "true");
    } else if (typeof value === "string") {
      formData.append(key, value);
    }
  }
  if (!settings.has("normalize")) {
    formData.append("normalize", "false");
  }

  processBtn.disabled = true;
  setStatus("Processing audio and creating ZIP...");

  try {
    const response = await fetch("/api/process", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      const detail = payload.detail || "Failed to process file.";
      throw new Error(detail);
    }

    const blob = await response.blob();
    const sliceCount = response.headers.get("X-Detected-Slices") || "Unknown";
    const filenames = response.headers.get("X-Exported-Filenames") || "";

    currentObjectUrl = URL.createObjectURL(blob);
    downloadLink.href = currentObjectUrl;
    resultEl.classList.remove("hidden");
    sliceCountEl.textContent = `Detected and exported ${sliceCount} slice(s).`;
    filenamesEl.textContent = filenames ? `Files: ${filenames}` : "";

    setStatus("Done. Your ZIP archive is ready.");
  } catch (error) {
    setStatus(error.message || "Unexpected error while processing.", true);
  } finally {
    processBtn.disabled = false;
  }
});
