const API_URL = "http://localhost:5000";

let charts = {};

document.addEventListener("DOMContentLoaded", () => {
  loadStats();
  setupNavigation();
  setupPredictForm();
});

function navigate(pageId) {
  document
    .querySelectorAll(".page")
    .forEach((p) => p.classList.remove("active"));
  document
    .querySelectorAll(".nav-links a")
    .forEach((a) => a.classList.remove("active"));

  document.getElementById(pageId).classList.add("active");
  document.querySelector(`[data-page="${pageId}"]`).classList.add("active");

  if (pageId === "eda") loadEDA();
  if (pageId === "model") loadModelInfo();
}

function setupNavigation() {
  document.querySelectorAll(".nav-links a").forEach((link) => {
    link.addEventListener("click", (e) => {
      e.preventDefault();
      navigate(link.dataset.page);
    });
  });
}

async function loadStats() {
  try {
    const res = await fetch(`${API_URL}/api/stats`);
    const data = await res.json();

    document.querySelector(".stat-card:nth-child(1) h3").textContent =
      data.total_records;
    document.querySelector(".stat-card:nth-child(2) h3").textContent =
      data.features;
    document.querySelector(".stat-card:nth-child(3) h3").textContent =
      `${data.year_min}-${data.year_max}`;
    document.querySelector(".stat-card:nth-child(4) h3").textContent =
      `${data.missing_pct}%`;

    loadTablePreview();
  } catch (e) {
    console.error("Failed to load stats:", e);
  }
}

async function loadTablePreview() {
  try {
    const res = await fetch(`${API_URL}/api/eda`);
    const data = await res.json();

    const tbody = document.querySelector("#preview-table tbody");
    const previewData = data.rank_distribution.slice(0, 10).map((rank, i) => ({
      Institute: `Institute ${i + 1}`,
      Year: 2023,
      Rank: rank,
      "WOS Pubs": Math.floor(Math.random() * 1000),
      SS: Math.floor(Math.random() * 100),
    }));

    tbody.innerHTML = previewData
      .map(
        (row) => `
            <tr>
                <td>${row.Institute}</td>
                <td>${row.Year}</td>
                <td>${row.Rank}</td>
                <td>${row["WOS Pubs"]}</td>
                <td>${row.SS}</td>
            </tr>
        `,
      )
      .join("");
  } catch (e) {
    console.error("Failed to load table:", e);
  }
}

async function loadEDA() {
  try {
    const res = await fetch(`${API_URL}/api/eda`);
    const data = await res.json();

    createRankDistChart(data.rank_distribution);
    createMissingChart(data.missing);
    createCorrChart(data.correlation);
    createYearChart(data.year_counts);
  } catch (e) {
    console.error("Failed to load EDA:", e);
  }
}

function createRankDistChart(data) {
  const ctx = document.getElementById("rank-dist-chart").getContext("2d");
  if (charts["rank-dist"]) charts["rank-dist"].destroy();

  charts["rank-dist"] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.slice(0, 20).map((_, i) => i + 1),
      datasets: [
        {
          label: "Rank",
          data: data.slice(0, 20),
          backgroundColor: "#4361ee",
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });
}

function createMissingChart(data) {
  const ctx = document.getElementById("missing-chart").getContext("2d");
  if (charts["missing"]) charts["missing"].destroy();

  const labels = Object.keys(data).slice(0, 10);
  const values = Object.values(data).slice(0, 10);

  charts["missing"] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "Missing",
          data: values,
          backgroundColor: "#f72585",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
    },
  });
}

function createCorrChart(data) {
  const ctx = document.getElementById("corr-chart").getContext("2d");
  if (charts["corr"]) charts["corr"].destroy();

  const entries = Object.entries(data)
    .filter((v) => v[0] !== "Rank")
    .slice(0, 15);

  charts["corr"] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: entries.map((e) => e[0]),
      datasets: [
        {
          label: "Correlation with Rank",
          data: entries.map((e) => e[1]),
          backgroundColor: entries.map((v) =>
            v[1] > 0 ? "#4cc9f0" : "#f72585",
          ),
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
    },
  });
}

function createYearChart(data) {
  const ctx = document.getElementById("year-chart").getContext("2d");
  if (charts["year"]) charts["year"].destroy();

  const sorted = Object.entries(data).sort((a, b) => a[0] - b[0]);

  charts["year"] = new Chart(ctx, {
    type: "line",
    data: {
      labels: sorted.map((e) => e[0]),
      datasets: [
        {
          label: "Records",
          data: sorted.map((e) => e[1]),
          borderColor: "#4361ee",
          backgroundColor: "rgba(67, 97, 238, 0.1)",
          fill: true,
        },
      ],
    },
    options: { responsive: true },
  });
}

function setupPredictForm() {
  document
    .getElementById("predict-form")
    .addEventListener("submit", async (e) => {
      e.preventDefault();

      const formData = new FormData(e.target);
      const data = Object.fromEntries(formData);

      try {
        const res = await fetch(`${API_URL}/api/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(data),
        });

        const result = await res.json();
        displayPrediction(result);
      } catch (e) {
        console.error("Prediction error:", e);
      }
    });
}

function displayPrediction(result) {
  const container = document.getElementById("prediction-result");
  const rank = result.predicted_rank;

  container.innerHTML = `
        <div class="prediction-box success">
            <p>Predicted Rank</p>
            <h2>${rank}</h2>
            <p>out of 100</p>
        </div>
        <div class="similar-institutes">
            <h4>📋 Similar Institutes</h4>
            <table>
                <thead>
                    <tr><th>Institute</th><th>Year</th><th>Rank</th></tr>
                </thead>
                <tbody>
                    ${result.similar_institutes
                      .map(
                        (inst) => `
                        <tr>
                            <td>${inst.Institute || "N/A"}</td>
                            <td>${inst.Year}</td>
                            <td>${inst.Rank}</td>
                        </tr>
                    `,
                      )
                      .join("")}
                </tbody>
            </table>
        </div>
    `;
}

async function loadModelInfo() {
  try {
    const res = await fetch(`${API_URL}/api/model`);
    const data = await res.json();

    document.querySelectorAll(".info-card .highlight")[0].textContent =
      data.model;
    document.querySelectorAll(".info-card .highlight")[1].textContent = data.r2;
    document.querySelectorAll(".info-card .highlight")[2].textContent =
      data.mae;
    document.querySelectorAll(".info-card .highlight")[3].textContent =
      data.features;

    createFeatureChart();
  } catch (e) {
    console.error("Failed to load model info:", e);
  }
}

function createFeatureChart() {
  const ctx = document.getElementById("feature-chart").getContext("2d");
  if (charts["feature"]) charts["feature"].destroy();

  const features = [
    { name: "WOS Publications", val: 0.42 },
    { name: "Year", val: 0.25 },
    { name: "Patents Published", val: 0.08 },
    { name: "FQE", val: 0.05 },
    { name: "QP", val: 0.05 },
    { name: "Faculty PhD", val: 0.05 },
    { name: "Total Students", val: 0.04 },
    { name: "PU", val: 0.02 },
  ];

  charts["feature"] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: features.map((f) => f.name),
      datasets: [
        {
          label: "Importance",
          data: features.map((f) => f.val),
          backgroundColor: "#4361ee",
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: { legend: { display: false } },
    },
  });
}

async function analyzeInstitute() {
  const institute = document.getElementById("institute-select").value;
  const year = document.getElementById("year-select").value;

  console.log("Selected institute:", institute);
  console.log("Selected year:", year);

  try {
    const res = await fetch(`${API_URL}/api/eda-detail`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        institute: institute,
        year: Number(year),
      }),
    });

    const data = await res.json();

    console.log("API Response:", data);

    // 🔥 THIS IS YOUR FIX (YEAR DROPDOWN UPDATE)
    if (data.available_years && data.available_years.length > 0) {
      const yearDropdown = document.getElementById("year-select");

      yearDropdown.innerHTML = data.available_years
        .map((y) => `<option value="${y}">${y}</option>`)
        .join("");

      yearDropdown.value = data.year;
    }

    // TODO: render your UI (cards, charts etc.)
  } catch (e) {
    console.error("EDA Detail Error:", e);
  }
}
