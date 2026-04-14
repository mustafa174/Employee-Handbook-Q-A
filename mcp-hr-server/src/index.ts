import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { config as loadEnv } from "dotenv";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..", "..");

loadEnv({ path: path.join(repoRoot, ".env") });

const defaultEmployeesPath = path.join(repoRoot, "fixtures", "employees.json");

const rawEmployees = process.env.EMPLOYEES_JSON_PATH;
const employeesPath = rawEmployees
  ? path.isAbsolute(rawEmployees)
    ? rawEmployees
    : path.resolve(repoRoot, rawEmployees)
  : defaultEmployeesPath;

type Employee = {
  employee_id: string;
  name: string;
  pto_days_remaining: number;
  sick_days_remaining: number;
  language_pref: string;
};

const loadEmployees = (): { employees: Employee[] } => {
  const raw = fs.readFileSync(employeesPath, "utf8");
  return JSON.parse(raw) as { employees: Employee[] };
};

const server = new McpServer({
  name: "hr-employee-db",
  version: "0.1.0",
});

server.registerTool(
  "get_leave_balance",
  {
    description:
      "Simulated HR: return PTO and sick balances for employee_id (fixtures/employees.json).",
    inputSchema: {
      employee_id: z.string().describe("e.g. E001"),
    },
  },
  async ({ employee_id }) => {
    const id = employee_id.trim();
    try {
      const data = loadEmployees();
      const emp = data.employees.find((e) => e.employee_id === id);
      if (!emp) {
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify({ error: "employee_not_found", employee_id: id }),
            },
          ],
        };
      }
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify(
              {
                employee_id: emp.employee_id,
                name: emp.name,
                pto_days_remaining: emp.pto_days_remaining,
                sick_days_remaining: emp.sick_days_remaining,
                language_pref: emp.language_pref,
              },
              null,
              2,
            ),
          },
        ],
      };
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      return {
        content: [{ type: "text" as const, text: `Error: ${msg}` }],
        isError: true,
      };
    }
  },
);

const transport = new StdioServerTransport();
await server.connect(transport);
