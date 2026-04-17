import { useEffect, useState } from "react";

type HarassmentReportPayload = {
  employee_id?: string;
  employee_name?: string;
  message: string;
};

type HarassmentReportModalProps = {
  payload: HarassmentReportPayload | null;
  onClose: () => void;
};

const HarassmentReportModal = ({ payload, onClose }: HarassmentReportModalProps) => {
  const [description, setDescription] = useState("");
  const [status, setStatus] = useState<"draft" | "sent">("draft");
  const [ticketId, setTicketId] = useState<string | null>(null);

  useEffect(() => {
    if (!payload) return;
    setDescription(payload.message);
    setStatus("draft");
    setTicketId(null);
  }, [payload]);

  if (!payload) return null;

  const employeeName = payload.employee_name?.trim() || "Unknown Employee";
  const employeeId = payload.employee_id?.trim() || "Unknown Employee";

  const onSendDemo = () => {
    const id = Math.floor(Math.random() * 9000) + 1000;
    setTicketId(`HR-${id}`);
    setStatus("sent");
  };

  const descriptionId = "harassment-report-description";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/45 p-4">
      <div className="w-full max-w-2xl rounded-xl border border-zinc-200 bg-white p-5 shadow-xl dark:border-zinc-700 dark:bg-zinc-900">
        <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-100">HR Incident Report (Demo)</h3>
        <div className="mt-4 space-y-1 text-sm text-zinc-700 dark:text-zinc-300">
          <p>
            <span className="font-medium">Employee Name:</span> {employeeName}
          </p>
          <p>
            <span className="font-medium">Employee ID:</span> {employeeId}
          </p>
        </div>

        <div className="mt-4">
          <label htmlFor={descriptionId} className="mb-1 block text-sm font-medium text-zinc-800 dark:text-zinc-200">
            Description
          </label>
          <textarea
            id={descriptionId}
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            rows={7}
            className="w-full rounded-lg border border-zinc-300 bg-zinc-50 px-3 py-2 text-sm text-zinc-900 outline-none focus:border-sky-400 focus:ring-2 focus:ring-sky-400/25 dark:border-zinc-700 dark:bg-zinc-950 dark:text-zinc-100"
          />
        </div>

        {status === "sent" && ticketId ? (
          <div className="mt-4 rounded-lg border border-emerald-300 bg-emerald-50 p-3 text-sm text-emerald-800 dark:border-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-200">
            <p>Report sent (simulated).</p>
            <p className="mt-1 font-medium">Ticket ID: {ticketId}</p>
            <p className="mt-1 text-xs">In a real system, this would be delivered to HR.</p>
          </div>
        ) : null}

        <div className="mt-5 flex justify-end gap-2">
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg border border-zinc-300 px-4 py-2 text-sm font-medium text-zinc-700 hover:bg-zinc-100 dark:border-zinc-700 dark:text-zinc-200 dark:hover:bg-zinc-800"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onSendDemo}
            className="rounded-lg bg-sky-600 px-4 py-2 text-sm font-medium text-white hover:bg-sky-500"
          >
            Send to HR (Demo)
          </button>
        </div>
      </div>
    </div>
  );
};

export { HarassmentReportModal };
