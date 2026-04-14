/**
 * Run FastAPI rag-api with the local venv's Python (Windows + Unix).
 */
require("dotenv").config({ path: require("node:path").join(__dirname, "..", ".env") });

const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");

const ragRoot = path.join(__dirname, "..", "rag-api");
const isWin = process.platform === "win32";
const venvPy = isWin
  ? path.join(ragRoot, ".venv", "Scripts", "python.exe")
  : path.join(ragRoot, ".venv", "bin", "python");

if (!fs.existsSync(venvPy)) {
  console.error(
    "rag-api venv missing. Run: cd rag-api && python -m venv .venv && .venv\\Scripts\\pip install -r requirements.txt",
  );
  process.exit(1);
}

const port = process.env.RAG_API_PORT || "3001";
const child = spawn(
  venvPy,
  ["-m", "uvicorn", "app.main:app", "--reload", "--host", "127.0.0.1", "--port", port],
  { cwd: ragRoot, stdio: "inherit", shell: false },
);
child.on("exit", (code) => process.exit(code ?? 0));
