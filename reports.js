document.querySelectorAll(".evidence-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document
      .querySelectorAll(".evidence-tab")
      .forEach((item) => item.classList.toggle("active", item === tab));
    const heatmap = tab.dataset.view === "heatmap";
    document.querySelector("#media-frame").classList.toggle("heatmap", heatmap);
    document.querySelector("#media-view-label").textContent = heatmap
      ? "ATTENTION MAP"
      : "ORIGINAL";
    document.querySelector("#evidence-caption").innerHTML = heatmap
      ? "<span></span> Heatmap shows the regions that contributed most to the verdict."
      : "<span></span> Original frame selected. Switch to heatmap to see the strongest signals.";
  }),
);

const savedReport = sessionStorage.getItem("truesight:last-report");
if (savedReport) {
  const report = JSON.parse(savedReport);
  const synthetic = report.verdict === "AI-Generated / Manipulated";
  const modelScore = Number(report.visual_confidence_score || 0);
  const score = synthetic ? modelScore : 100 - modelScore;
  const scoreText = score.toFixed(1);
  const verdictTitle = document.querySelector(".verdict-copy h3");
  const verdictDescription = document.querySelector(".verdict-copy p");
  const scoreLabel = document.querySelector(".confidence-label b");
  const scoreRing = document.querySelector(".verdict-ring b");
  const riskFill = document.querySelector(".risk-track i");
  const verdictCard = document.querySelector(".verdict-card");

  verdictTitle.innerHTML = synthetic
    ? "Likely <em>synthetic</em>"
    : "Likely <em>authentic</em>";
  const mediaType = (report.media_type || "Media").toLowerCase();
  verdictDescription.textContent = synthetic
    ? `The ${mediaType} model detected signals consistent with generated or manipulated media.`
    : `The ${mediaType} model found no material signs of generated or manipulated media.`;
  scoreLabel.textContent = `${scoreText}%`;
  scoreRing.innerHTML = `${Math.floor(score)}<span>.${scoreText.split(".")[1]}</span>`;
  riskFill.style.width = `${score}%`;
  verdictCard.style.borderLeftColor = synthetic
    ? "var(--danger)"
    : "var(--success)";
  scoreLabel.style.color = synthetic ? "var(--danger)" : "var(--success)";
  document.querySelector(".verdict-ring").style.background =
    `conic-gradient(${synthetic ? "var(--danger)" : "var(--success)"} 0 ${score * 3.6}deg, #edf0f3 ${score * 3.6}deg)`;

  const statValues = document.querySelectorAll(".quick-stat b");
  statValues[0].textContent = report.media_type || "Image";
  statValues[1].textContent = report.analyzedAt
    ? new Date(report.analyzedAt).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "Instant";
  statValues[2].textContent = report.analyzedAt
    ? new Date(report.analyzedAt).toLocaleDateString()
    : "Today";
  const metaValues = document.querySelectorAll(".verdict-meta b");
  metaValues[0].textContent = (report.media_type || "IMAGE").toUpperCase();
  metaValues[1].textContent = report.analyzedAt
    ? new Date(report.analyzedAt).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
      })
    : "JUST NOW";
  if (metaValues[2]) {
    metaValues[2].textContent = report.models_used
      ? `0${report.models_used}`
      : "02";
  }

  // Populate dynamic signal breakdown based on dual-model analysis
  const signalContainer = document.querySelector(".signal-list");
  if (signalContainer) {
    const globalScore = Number(report.global_ai_score ?? modelScore).toFixed(1);
    const hasFace = Boolean(report.face_detected);
    const faceScore =
      hasFace && report.facial_deepfake_score != null
        ? Number(report.facial_deepfake_score).toFixed(1)
        : null;
    const peakScore = Number(report.peak_anomaly_score ?? modelScore).toFixed(
      1,
    );

    const getSeverity = (val) => {
      const num = Number(val);
      if (num >= 65) return { cls: "high", label: "HIGH" };
      if (num >= 35) return { cls: "medium", label: "MED" };
      return { cls: "low", label: "LOW" };
    };

    const s1 = getSeverity(globalScore);
    const s2 = hasFace ? getSeverity(faceScore) : { cls: "low", label: "N/A" };
    const s3 = getSeverity(peakScore);
    const s4 = synthetic
      ? { cls: "high", label: "FLAGGED" }
      : { cls: "low", label: "CLEAR" };

    signalContainer.innerHTML = `
      <div class="signal">
        <div class="signal-icon ${s1.cls}"><svg viewBox="0 0 24 24"><path d="M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Z"/><path d="M9 10h.01M15 10h.01"/></svg></div>
        <div><b>Global Generative AI (ViT Detector)</b><small>Diffusion patterns & synthetic textures across full frame (${globalScore}%)</small></div>
        <em>${s1.label}</em>
      </div>
      <div class="signal">
        <div class="signal-icon ${s2.cls}"><svg viewBox="0 0 24 24"><path d="M12 3v10m0 4v.01M5.05 19h13.9c1.54 0 2.5-1.67 1.73-3L13.73 4c-.77-1.33-2.69-1.33-3.46 0L3.32 16c-.77 1.33.19 3 1.73 3Z"/></svg></div>
        <div><b>Facial Manipulation (SigLIP 2)</b><small>${hasFace ? `Face-swap & landmark blending check (${faceScore}%)` : "No human face detected in image"}</small></div>
        <em>${s2.label}</em>
      </div>
      <div class="signal">
        <div class="signal-icon ${s3.cls}"><svg viewBox="0 0 24 24"><path d="M4 19V5m5 14V9m5 10V3m5 16v-7"/></svg></div>
        <div><b>Peak Anomaly Intensity</b><small>Strongest detected generative artifact peak (${peakScore}%)</small></div>
        <em>${s3.label}</em>
      </div>
      <div class="signal">
        <div class="signal-icon ${s4.cls}"><svg viewBox="0 0 24 24"><path d="M4 4h16v16H4zM8 8h8m-8 4h8m-8 4h5"/></svg></div>
        <div><b>Dual-Model Consensus</b><small>${synthetic ? "Confirmed synthetic anomalies across model ensemble" : "Both models confirm authentic photographic properties"}</small></div>
        <em>${s4.label}</em>
      </div>
    `;
  }

  if (report.evidence_frame_base64) {
    const frame = document.querySelector("#media-frame");
    frame.style.backgroundImage = `url(data:image/jpeg;base64,${report.evidence_frame_base64})`;
    frame.style.backgroundPosition = "center";
    frame.style.backgroundSize = "cover";
  }
} else {
  document.querySelector(".verdict-copy h3").textContent = "No report selected";
  document.querySelector(".verdict-copy p").textContent =
    "Run an image or social-link analysis to view its forensic report.";
  document.querySelector(".confidence-label b").textContent = "—";
  document.querySelector(".risk-track i").style.width = "0%";
  document.querySelector(".verdict-ring b").textContent = "—";
  document.querySelector(".verdict-ring").style.background =
    "conic-gradient(#edf0f3 0 360deg)";
  document.querySelectorAll(".quick-stat b").forEach((item) => {
    item.textContent = "—";
  });
  document.querySelectorAll(".verdict-meta b").forEach((item) => {
    item.textContent = "—";
  });
}

document
  .querySelector(".secondary-button")
  ?.addEventListener("click", (event) => {
    event.preventDefault();
    const report = savedReport ? JSON.parse(savedReport) : {};
    const score = Number(report.visual_confidence_score || 0).toFixed(1);
    const gScore =
      report.global_ai_score != null ? `${report.global_ai_score}%` : "N/A";
    const fScore =
      report.facial_deepfake_score != null
        ? `${report.facial_deepfake_score}%`
        : report.face_detected
          ? "0.0%"
          : "N/A (No face)";
    const lines = [
      "TRUESIGHT AI — FORENSIC REPORT",
      "",
      `Media: ${report.filename || "Latest examination"}`,
      `Source: ${report.source || "Upload"}`,
      `Verdict: ${report.verdict || "No analysis available"}`,
      `Overall Fake Score: ${score}%`,
      `Global AI Detector (ViT): ${gScore}`,
      `Facial Deepfake Detector (SigLIP 2): ${fScore}`,
      `Face Detected: ${report.face_detected ? "Yes" : "No"}`,
      `Models Used: ${report.models_used || 2} (Ensemble)`,
      `Generated: ${report.analyzedAt ? new Date(report.analyzedAt).toLocaleString() : new Date().toLocaleString()}`,
    ];
    const escapePdf = (value) =>
      String(value)
        .replace(/[^\x20-\x7E]/g, "?")
        .replace(/[\\()]/g, "\\$&");
    const content = `BT /F1 16 Tf 50 760 Td ${lines.map((line, index) => `(${escapePdf(line)}) Tj${index < lines.length - 1 ? " 0 -24 Td" : ""}`).join("\n")} ET`;
    const objects = [
      "<< /Type /Catalog /Pages 2 0 R >>",
      "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
      "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
      `<< /Length ${content.length} >>\nstream\n${content}\nendstream`,
      "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ];
    let pdf = "%PDF-1.4\n";
    const offsets = [0];
    objects.forEach((object, index) => {
      offsets.push(pdf.length);
      pdf += `${index + 1} 0 obj\n${object}\nendobj\n`;
    });
    const xref = pdf.length;
    pdf += `xref\n0 ${objects.length + 1}\n0000000000 65535 f \n`;
    offsets.slice(1).forEach((offset) => {
      pdf += `${String(offset).padStart(10, "0")} 00000 n \n`;
    });
    pdf += `trailer\n<< /Size ${objects.length + 1} /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`;
    const url = URL.createObjectURL(
      new Blob([pdf], { type: "application/pdf" }),
    );
    const download = document.createElement("a");
    download.href = url;
    download.download = "truesight-forensic-report.pdf";
    download.click();
    URL.revokeObjectURL(url);
  });
