const select = (q, scope = document) => scope.querySelector(q);
const selectAll = (q, scope = document) => [...scope.querySelectorAll(q)];
let pickedFile = null;

function readableSize(bytes) {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}
function classify() {
  return ["IMG", "Image"];
}
function selectFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    select("#form-message").textContent =
      "Choose a JPG, PNG, WEBP, or other image file.";
    return;
  }
  if (file.size > 4 * 1024 * 1024) {
    select("#form-message").textContent = "Images must be 4 MB or smaller.";
    return;
  }
  pickedFile = file;
  const [code, type] = classify(file);
  select("#file-name").textContent = file.name;
  select("#file-meta").textContent = `${readableSize(file.size)} · ${type}`;
  select("#file-type-icon").textContent = code;
  select("#file-selection").hidden = false;
  select("#form-message").textContent = "";
}
function setProgress(value, active, label) {
  select("#progress-fill").style.width = `${value}%`;
  select("#progress-number").textContent = `${String(value).padStart(2, "0")}%`;
  select("#progress-heading").textContent = label;
  selectAll(".pipeline-steps li").forEach((item) => {
    const position = Number(item.dataset.step);
    item.classList.toggle("done", position < active || value === 100);
    item.classList.toggle("active", position === active && value < 100);
  });
}

selectAll(".intake-tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    selectAll(".intake-tab").forEach((item) =>
      item.classList.toggle("active", item === tab),
    );
    selectAll(".intake-pane").forEach((pane) =>
      pane.classList.toggle("active", pane.dataset.pane === tab.dataset.intake),
    );
  }),
);
select("#file-input").addEventListener("change", (event) =>
  selectFile(event.target.files[0]),
);
select("#clear-file").addEventListener("click", () => {
  pickedFile = null;
  select("#file-input").value = "";
  select("#file-selection").hidden = true;
});
["dragenter", "dragover"].forEach((eventName) =>
  select("#drop-zone").addEventListener(eventName, (event) => {
    event.preventDefault();
    select("#drop-zone").classList.add("dragover");
  }),
);
["dragleave", "drop"].forEach((eventName) =>
  select("#drop-zone").addEventListener(eventName, (event) => {
    event.preventDefault();
    select("#drop-zone").classList.remove("dragover");
  }),
);
select("#drop-zone").addEventListener("drop", (event) =>
  selectFile(event.dataTransfer.files[0]),
);
selectAll(".source-chip").forEach((chip) =>
  chip.addEventListener("click", () => {
    select("#media-url").value = chip.dataset.url;
  }),
);
select("#analyze-button").addEventListener("click", async () => {
  const onLinkTab = select('[data-pane="link"]').classList.contains("active");
  const socialUrl = select("#media-url").value.trim();
  if (onLinkTab && !/^https?:\/\//i.test(socialUrl)) {
    select("#form-message").textContent =
      "Add a complete public URL (starting with https:// or http://) to begin.";
    return;
  }
  if (!onLinkTab && !pickedFile) {
    select("#form-message").textContent = "Choose an image to begin.";
    return;
  }
  const button = select("#analyze-button");
  button.disabled = true;
  const mediaType = onLinkTab ? "Image" : classify(pickedFile)[1];
  button.querySelector("span").textContent = onLinkTab
    ? "Extracting image"
    : `Analyzing ${mediaType.toLowerCase()}`;
  select("#pipeline-name").textContent = "Examination in progress";
  select("#progress-section").scrollIntoView({
    behavior: "smooth",
    block: "center",
  });
  setProgress(15, 1, `Uploading and validating ${mediaType.toLowerCase()}`);

  let progressStep = 1;
  const progressTimer = setInterval(() => {
    if (progressStep === 1) {
      progressStep = 2;
      setProgress(45, 2, "Extracting facial and forensic regions");
    } else if (progressStep === 2) {
      progressStep = 3;
      setProgress(75, 3, "Running dual-model ensemble analysis");
    }
  }, 1200);

  const apiBase = window.API_BASE_URL || "";
  try {
    let response;
    if (onLinkTab) {
      response = await fetch(`${apiBase}/api/analyze-social-image`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url: socialUrl }),
      });
    } else {
      const formData = new FormData();
      formData.append("file", pickedFile);
      response = await fetch(`${apiBase}/api/analyze-deepfake`, {
        method: "POST",
        body: formData,
      });
    }
    clearInterval(progressTimer);

    const result = await response.json().catch(() => ({}));
    if (!response.ok) {
      let errorMsg = "The analysis could not be completed.";
      if (result) {
        if (typeof result.detail === "string") {
          errorMsg = result.detail;
        } else if (Array.isArray(result.detail)) {
          errorMsg = result.detail
            .map((err) => err.msg || JSON.stringify(err))
            .join("; ");
        } else if (result.error) {
          errorMsg =
            typeof result.error === "string"
              ? result.error
              : JSON.stringify(result.error);
        }
      }
      throw new Error(errorMsg);
    }

    setProgress(100, 4, "Preparing your report");
    const savedReport = {
      ...result,
      filename: onLinkTab ? new URL(socialUrl).hostname : pickedFile.name,
      source: onLinkTab ? "Social link" : "Upload",
      media_type: mediaType,
      analyzedAt: new Date().toISOString(),
    };
    sessionStorage.setItem(
      "truesight:last-report",
      JSON.stringify(savedReport),
    );
    const history = JSON.parse(
      localStorage.getItem("truesight:reports") || "[]",
    );
    localStorage.setItem(
      "truesight:reports",
      JSON.stringify([savedReport, ...history].slice(0, 20)),
    );
    const completeCallout = select("#complete-callout");
    if (completeCallout) completeCallout.hidden = false;
    window.location.href = "reports.html";
  } catch (error) {
    clearInterval(progressTimer);
    select("#form-message").textContent =
      error.message || "Unable to reach the detection API.";
    select("#pipeline-name").textContent = "Analysis failed";
    setProgress(0, 0, "Ready when you are");
  } finally {
    clearInterval(progressTimer);
    button.disabled = false;
    button.querySelector("span").textContent = "Analyze media";
  }
});
