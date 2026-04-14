require("dotenv").config({ path: require("node:path").join(__dirname, "..", ".env") });

const { spawnSync } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const ragRoot = path.join(__dirname, "..", "rag-api");
const isWin = process.platform === "win32";
const venvPy = isWin
  ? path.join(ragRoot, ".venv", "Scripts", "python.exe")
  : path.join(ragRoot, ".venv", "bin", "python");

if (!fs.existsSync(venvPy)) {
  console.warn("Skipping rag-api pytest: venv not found at", venvPy);
  process.exit(0);
}

const r = spawnSync(venvPy, ["-m", "pytest", "-q"], {
  cwd: ragRoot,
  stdio: "inherit",
});
process.exit(r.status ?? 1);
