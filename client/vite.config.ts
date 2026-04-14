import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import path from "node:path";
import { defineConfig, loadEnv } from "vite";

const repoRoot = path.resolve(__dirname, "..");

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, repoRoot, "");
  const ragApiUrl = env.RAG_API_URL || "http://127.0.0.1:3001";

  return {
    envDir: repoRoot,
    plugins: [react(), tailwindcss()],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "src"),
      },
    },
    server: {
      port: 5173,
      /** Prefer this URL in the browser instead of `localhost` if connection fails. */
      host: "127.0.0.1",
      proxy: {
        "/api": {
          target: ragApiUrl,
          changeOrigin: true,
        },
      },
    },
  };
});
