// TrueSight AI API Configuration
// When deployed to Vercel or running on the same origin (e.g. port 8000), use same-origin ('').
// When running a local frontend dev server on another port (e.g. 5500, 3000), default to 'http://localhost:8000'.
(function () {
  if (typeof window.API_BASE_URL === "undefined") {
    const isLocalhost =
      window.location.hostname === "localhost" ||
      window.location.hostname === "127.0.0.1";
    const isSeparatePort =
      window.location.port && window.location.port !== "8000";
    window.API_BASE_URL =
      isLocalhost && isSeparatePort ? "http://localhost:8000" : "";
  }
})();
