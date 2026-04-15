import { HandbookQA } from "./components/HandbookQA";
import { useQuery } from "@tanstack/react-query";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "./theme";
import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { SettingsPage } from "./pages/Settings";
import { useChatHistory } from "./state/ChatHistoryContext";
import { apiUrl } from "./apiBase";

const EMPLOYEE_OPTIONS = [
  { employee_id: "E001", name: "Alex Chen" },
  { employee_id: "E002", name: "Sara Khan" },
] as const;

type EmployeeOption = {
  employee_id: string;
  name: string;
};

const fetchEmployees = async (): Promise<EmployeeOption[]> => {
  const res = await fetch(apiUrl("/api/employees"));
  if (!res.ok) throw new Error("Employees endpoint unavailable");
  const body = (await res.json()) as EmployeeOption[];
  if (!Array.isArray(body)) return [];
  return body.filter((row) => typeof row?.employee_id === "string" && typeof row?.name === "string");
};

const App = () => {
  const { theme, toggleTheme } = useTheme();
  const navigate = useNavigate();
  const location = useLocation();
  const {
    chatHistory,
    setChatHistory,
    selectedEmployeeId,
    setSelectedEmployeeId,
  } = useChatHistory();
  const [employeeOptions, setEmployeeOptions] = useState<EmployeeOption[]>([...EMPLOYEE_OPTIONS]);
  const employeesQuery = useQuery({
    queryKey: ["employees", "options"],
    queryFn: fetchEmployees,
    staleTime: 60_000,
  });

  useEffect(() => {
    if (employeesQuery.data && employeesQuery.data.length > 0) {
      setEmployeeOptions(employeesQuery.data);
      return;
    }
    if (employeesQuery.isError) {
      setEmployeeOptions([...EMPLOYEE_OPTIONS]);
    }
  }, [employeesQuery.data, employeesQuery.isError]);

  useEffect(() => {
    if (employeeOptions.length === 0) return;
    const isValid = employeeOptions.some((row) => row.employee_id === selectedEmployeeId);
    if (!isValid) {
      setSelectedEmployeeId(employeeOptions[0].employee_id);
    }
  }, [employeeOptions, selectedEmployeeId, setSelectedEmployeeId]);

  const isSettings = location.pathname === "/settings";
  const isAssistantPage = isSettings === false;
  const pageTitle = useMemo(
    () => (isSettings ? "Settings" : "Employee Handbook Q&A"),
    [isSettings],
  );

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 transition-colors dark:bg-zinc-950 dark:text-zinc-100">
      <header className="border-b border-zinc-200 bg-white/90 backdrop-blur dark:border-zinc-800 dark:bg-zinc-900/80">
        <div className="mx-auto flex max-w-6xl items-center justify-between gap-4 px-6 py-4">
          <h1 className="text-lg font-semibold tracking-tight">
            {pageTitle}
          </h1>
          <div className="flex items-center gap-3">
            <nav className="flex items-center gap-1 rounded-lg border border-zinc-200 bg-zinc-100/80 p-1 dark:border-zinc-700 dark:bg-zinc-800/80">
              <button
                type="button"
                onClick={() => navigate("/")}
                className={[
                  "rounded-md px-3 py-1.5 text-xs font-medium transition-all duration-200",
                  isAssistantPage
                    ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-zinc-50"
                    : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-100",
                ].join(" ")}
              >
                Assistant
              </button>
              <button
                type="button"
                onClick={() => navigate("/settings")}
                className={[
                  "rounded-md px-3 py-1.5 text-xs font-medium transition-all duration-200",
                  isSettings
                    ? "bg-white text-zinc-900 shadow-sm dark:bg-zinc-700 dark:text-zinc-50"
                    : "text-zinc-600 hover:text-zinc-900 dark:text-zinc-300 dark:hover:text-zinc-100",
                ].join(" ")}
              >
                Settings
              </button>
            </nav>
            <span className="hidden text-xs text-zinc-500 sm:inline dark:text-zinc-500">
              Chroma · OpenAI · LangGraph
            </span>
            <button
              type="button"
              onClick={toggleTheme}
              className="inline-flex items-center gap-2 rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm font-medium text-zinc-800 shadow-sm hover:bg-zinc-50 dark:border-zinc-600 dark:bg-zinc-800 dark:text-zinc-100 dark:hover:bg-zinc-700"
              aria-label={theme === "light" ? "Switch to dark theme" : "Switch to light theme"}
            >
              {theme === "light" ? (
                <>
                  <Moon className="size-4" aria-hidden />
                  Dark
                </>
              ) : (
                <>
                  <Sun className="size-4" aria-hidden />
                  Light
                </>
              )}
            </button>
            <label className="hidden items-center gap-2 text-xs text-zinc-500 sm:flex dark:text-zinc-400">
              User
              <select
                value={selectedEmployeeId}
                onChange={(e) => setSelectedEmployeeId(e.target.value)}
                className="rounded-lg border border-zinc-300 bg-white px-2 py-1.5 text-xs font-medium text-zinc-800 outline-none transition-colors focus:border-sky-500 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
              >
                {employeeOptions.map((employee) => (
                  <option key={employee.employee_id} value={employee.employee_id}>
                    {employee.name} ({employee.employee_id})
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>
      </header>
      <main className="mx-auto px-6 py-2 max-w-7xl">
        <section className={isSettings ? "hidden" : "block"}>
          <HandbookQA
            chatHistory={chatHistory}
            onChatHistoryChange={setChatHistory}
            selectedEmployeeId={selectedEmployeeId}
          />
        </section>
        <section className={isSettings ? "block" : "hidden"}>
          <SettingsPage />
        </section>
      </main>
    </div>
  );
};

export default App;
