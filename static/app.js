const cropSelect = document.querySelector("#crop");
const regionSelect = document.querySelector("#region");
const plantingDate = document.querySelector("#planting-date");
const predictionForm = document.querySelector("#prediction-form");
const predictButton = document.querySelector("#predict-button");
const analysisButton = document.querySelector("#analysis-button");
const toast = document.querySelector("#toast");
let currentPrediction = null;
let toastTimer;

const today = new Date();
plantingDate.value = [
  today.getFullYear(),
  String(today.getMonth() + 1).padStart(2, "0"),
  String(today.getDate()).padStart(2, "0"),
].join("-");

function showToast(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 4300);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });
  const body = await response.json();
  if (!response.ok) {
    const error = new Error(body.error || "Something went wrong. Please try again.");
    error.body = body;
    throw error;
  }
  return body;
}

function fillSelect(select, values, placeholder) {
  select.replaceChildren(new Option(placeholder, ""));
  for (const value of values) select.add(new Option(value, value));
}

async function loadOptions(crop = "") {
  const query = crop ? `?crop=${encodeURIComponent(crop)}` : "";
  const data = await requestJson(`/api/options${query}`);
  if (!crop) {
    fillSelect(cropSelect, data.crops, "Select a crop");
    regionSelect.disabled = true;
    fillSelect(regionSelect, [], "Choose a crop first");
    return;
  }
  fillSelect(regionSelect, data.regions, "Select a region");
  regionSelect.disabled = data.regions.length === 0;
}

function presentPrediction(result) {
  currentPrediction = result;
  document.querySelector("#empty-state").hidden = true;
  document.querySelector("#result-content").hidden = false;
  document.querySelector("#harvest-dates").textContent =
    `${result.harvest_window.start_label} — ${result.harvest_window.end_label}`;
  document.querySelector("#result-summary").textContent =
    `${result.crop} in ${result.region}, South Africa`;
  document.querySelector("#result-crop").textContent = result.crop;
  document.querySelector("#result-region").textContent = result.region;
  document.querySelector("#result-planted").textContent = result.planting_date_label;
  document.querySelector("#calendar-code").textContent = result.calendar_code;
  document.querySelector("#result-note").textContent = result.note;

  const modelBox = document.querySelector("#model-box");
  modelBox.replaceChildren();
  const heading = document.createElement("strong");
  heading.textContent = "RANDOM FOREST COMPARISON";
  const detail = document.createElement("span");
  detail.textContent = result.model_window
    ? `Experimental model estimate: ${result.model_window.start_label} – ${result.model_window.end_label}.`
    : "No model estimate is available. The regional calendar estimate is shown above.";
  modelBox.append(heading, detail);

  document.querySelector("#ai-result").hidden = true;
  document.querySelector("#analysis-text").textContent = "";
  analysisButton.disabled = false;
}

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  predictButton.disabled = true;
  predictButton.querySelector("span").textContent = "Calculating…";
  try {
    const result = await requestJson("/api/predict", {
      method: "POST",
      body: JSON.stringify({
        crop: cropSelect.value,
        region: regionSelect.value,
        planting_date: plantingDate.value,
      }),
    });
    presentPrediction(result);
  } catch (error) {
    if (error.body?.planting_window) {
      const { start_label, end_label } = error.body.planting_window;
      showToast(`${error.message} Expected planting period: ${start_label} – ${end_label}.`);
    } else {
      showToast(error.message);
    }
  } finally {
    predictButton.disabled = false;
    predictButton.querySelector("span").textContent = "Estimate harvest window";
  }
});

cropSelect.addEventListener("change", async () => {
  regionSelect.disabled = true;
  fillSelect(regionSelect, [], "Loading regions…");
  try {
    await loadOptions(cropSelect.value);
  } catch (error) {
    fillSelect(regionSelect, [], "Could not load regions");
    showToast(error.message);
  }
});

analysisButton.addEventListener("click", async () => {
  if (!currentPrediction) return;
  analysisButton.disabled = true;
  analysisButton.querySelector("span:nth-child(2)").textContent = "Generating guidance…";
  try {
    const result = await requestJson("/api/analysis", {
      method: "POST",
      body: JSON.stringify(currentPrediction),
    });
    document.querySelector("#analysis-text").textContent = result.analysis;
    document.querySelector("#ai-result").hidden = false;
  } catch (error) {
    showToast(error.message);
  } finally {
    analysisButton.disabled = false;
    analysisButton.querySelector("span:nth-child(2)").textContent = "Get AI field guidance";
  }
});

loadOptions().catch((error) => showToast(error.message));
