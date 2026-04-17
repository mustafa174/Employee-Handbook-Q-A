import {
  ArrowRight,
  BookOpen,
  BrainCircuit,
  CheckCircle2,
  FileText,
  Lock,
  MessageSquare,
  Network,
  ShieldAlert,
  Users,
} from "lucide-react";
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";

const LandingPage = () => {
  const navigate = useNavigate();

  return (
    <div className="relative min-h-screen bg-[#f0f4f8] bg-[radial-gradient(at_0%_0%,_rgba(79,70,229,0.15)_0px,_transparent_50%),radial-gradient(at_100%_0%,_rgba(6,182,212,0.15)_0px,_transparent_50%),radial-gradient(at_50%_100%,_rgba(219,39,119,0.1)_0px,_transparent_50%)] font-sans text-[#1e293b] selection:bg-indigo-100 selection:text-indigo-900">
      <nav className="fixed top-0 z-50 w-full border-b border-white/40 bg-white/20 backdrop-blur-md">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
          <div className="flex items-center gap-2">
            <div className="text-[#4f46e5]">
              <BookOpen className="h-7 w-7" strokeWidth={2.5} />
            </div>
            <span className="text-2xl font-extrabold tracking-tight text-[#4f46e5]">Employee Handbook Q&A System</span>
          </div>
          <div className="hidden items-center gap-8 text-sm font-medium text-slate-600 md:flex">
            <a href="#features" className="transition-colors hover:text-indigo-600">
              Features
            </a>
            <a href="#how-it-works" className="transition-colors hover:text-indigo-600">
              How it Works
            </a>
            <a href="#security" className="transition-colors hover:text-indigo-600">
              Security
            </a>
          </div>
          <div className="flex items-center gap-4">
            <button
              type="button"
              onClick={() => navigate("/assistant")}
              className="rounded-full bg-[#4f46e5] px-6 py-2.5 text-sm font-semibold text-white shadow-[0_4px_12px_rgba(79,70,229,0.2)] transition-all hover:bg-indigo-700 hover:shadow-[0_6px_16px_rgba(79,70,229,0.3)]"
            >
              Assistant
            </button>
          </div>
        </div>
      </nav>

      <section className="mx-auto grid max-w-7xl items-center gap-12 px-4 pb-20 pt-32 sm:px-6 lg:grid-cols-2 lg:px-8">
        <div className="max-w-2xl">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-white/40 bg-white/65 px-3.5 py-1.5 text-xs font-semibold text-[#4f46e5] backdrop-blur-sm">
            <BrainCircuit className="h-4 w-4" />
            <span>Agentic AI for Modern HR Teams</span>
          </div>
          <h1 className="mb-6 text-5xl font-extrabold leading-[1.1] tracking-tight text-[#0f172a] lg:text-6xl lg:-tracking-[2px]">
            Stop answering the same HR questions{" "}
            <span className="bg-gradient-to-r from-indigo-600 to-violet-600 bg-clip-text text-transparent">
              20 times a day.
            </span>
          </h1>
          <p className="mb-10 text-lg leading-[1.6] text-[#475569]">
            An intelligent chatbot that knows your employee handbook inside out. It answers questions accurately with
            policy citations, and seamlessly escalates sensitive issues to human HR.
          </p>
          <div className="flex flex-col gap-4 sm:flex-row">
            <button
              type="button"
              onClick={() => navigate("/assistant")}
              className="group flex cursor-pointer items-center justify-center gap-2 rounded-full bg-[#4f46e5] px-8 py-3.5 font-semibold text-white shadow-[0_4px_12px_rgba(79,70,229,0.2)] transition-all hover:bg-indigo-700"
            >
              Get Started
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </button>
            <button
              type="button"
              onClick={() => navigate("/assistant")}
              className="flex cursor-pointer items-center justify-center gap-2 rounded-full border border-white/40 bg-white/65 px-8 py-3.5 font-semibold text-[#1e293b] shadow-[0_4px_12px_rgba(0,0,0,0.05)] backdrop-blur-sm transition-all hover:bg-white"
            >
              <MessageSquare className="h-4 w-4" />
              Try Interactive Demo
            </button>
          </div>
          <div className="mt-8 flex items-center gap-4 text-sm text-slate-500">
            <div className="flex items-center gap-1">
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
              <span>RAG-Powered Accuracy</span>
            </div>
            <div className="flex items-center gap-1">
              <CheckCircle2 className="h-4 w-4 text-emerald-500" />
              <span>MCP Integration</span>
            </div>
          </div>
        </div>

        <div className="relative">
          <div className="absolute inset-0 -z-10 scale-105 rotate-3 rounded-3xl bg-gradient-to-tr from-indigo-100 to-violet-50" />
          <div className="flex h-[480px] flex-col rounded-[24px] border border-white/40 bg-white/65 p-6 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] backdrop-blur-[16px]">
            <div className="mb-5 flex items-center gap-3 border-b border-black/5 pb-4">
              <div className="flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-[#4f46e5] to-[#06b6d4] shadow-sm">
                <BookOpen className="h-4 w-4 text-white" />
              </div>
              <div>
                  <div className="text-sm font-bold text-[#1e293b]">Our Assistant</div>
                <div className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-[#10b981]">
                  ● Online
                </div>
              </div>
            </div>
            <div className="flex flex-1 flex-col gap-3 overflow-y-auto">
              <div className="max-w-[85%] self-end rounded-2xl rounded-br-sm bg-[#4f46e5] px-4 py-3 text-sm leading-[1.5] text-white">
                How many vacation days do I get after 2 years?
              </div>
              <div className="max-w-[85%] self-start rounded-2xl rounded-bl-sm border border-black/5 bg-white px-4 py-3 text-sm leading-[1.5] text-[#1e293b]">
                <p>
                  Based on the Employee Handbook, employees with 2 years of tenure receive{" "}
                  <strong>15 days of Paid Time Off (PTO)</strong> per year.
                </p>
                <div className="mt-2 inline-flex items-center gap-1 rounded-md bg-[#f1f5f9] px-2 py-1 text-[11px] font-semibold text-[#4f46e5]">
                  <FileText className="h-3 w-3" />
                  Section 4.1 - Paid Time Off Accrual (Page 22)
                </div>
              </div>
            </div>
            <div className="mt-2.5 flex h-11 items-center rounded-xl bg-black/5 px-4 text-[13px] text-[#94a3b8]">
              Ask a question about the handbook...
            </div>
          </div>
        </div>
      </section>

      <section id="features" className="relative py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="mx-auto mb-16 max-w-3xl text-center">
            <h2 className="mb-4 text-3xl font-bold tracking-tight text-slate-900">Intelligent, safe, and context-aware.</h2>
            <p className="text-lg text-slate-600">
              Built on advanced RAG and agentic workflows to ensure every answer is accurate, cited, and appropriate.
            </p>
          </div>
          <div className="grid gap-8 md:grid-cols-3">
            <FeatureCard
              icon={<FileText className="h-6 w-6 text-indigo-600" />}
              title="RAG-Powered Citations"
              description="Every answer includes direct policy section grounding from your handbook."
            />
            <FeatureCard
              icon={<ShieldAlert className="h-6 w-6 text-rose-600" />}
              title="Smart Escalation"
              description="Sensitive topics like harassment or accommodations are routed to HR."
            />
            <FeatureCard
              icon={<Network className="h-6 w-6 text-violet-600" />}
              title="MCP Agentic Architecture"
              description="Securely merges employee context with handbook policy retrieval."
            />
          </div>
        </div>
      </section>

      <section id="how-it-works" className="relative py-20">
        <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center gap-12 md:flex-row">
            <div className="flex-1">
              <h2 className="mb-8 text-3xl font-bold tracking-tight text-[#0f172a]">How Our Assistant Works</h2>
              <div className="space-y-8">
                <Step number="1" title="Upload your Handbook" description="Ingest your employee handbook into Chroma." />
                <Step number="2" title="AI Indexes & Retrieves" description="RAG and retrieval find the best grounded sections." />
                <Step number="3" title="Employees Ask Questions" description="Instant policy and profile-aware responses." />
                <Step number="4" title="HR Handles Sensitive Cases" description="Escalation flows trigger when required." />
              </div>
            </div>
            <div id="security" className="w-full flex-1">
              <div className="rounded-[24px] border border-white/40 bg-white/65 p-8 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] backdrop-blur-[16px]">
                <div className="mb-6 flex items-center gap-4 border-b border-black/5 pb-6">
                  <div className="rounded-xl bg-[#4f46e5]/10 p-3">
                    <Users className="h-6 w-6 text-[#4f46e5]" />
                  </div>
                  <div>
                    <h3 className="font-bold text-[#1e293b]">HR Time Saved</h3>
                    <p className="text-sm text-[#475569]">Average weekly metrics</p>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-sm text-[#475569]">
                  <Lock className="h-4 w-4" />
                  <span>SOC2 Type II style security posture and internal policy guardrails.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

const FeatureCard = ({ icon, title, description }: { icon: ReactNode; title: string; description: string }) => {
  return (
    <div className="group rounded-[24px] border border-white/40 bg-white/65 p-8 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] transition-all hover:shadow-[0_12px_40px_0_rgba(31,38,135,0.1)] backdrop-blur-[16px]">
      <div className="mb-6 flex h-12 w-12 items-center justify-center rounded-2xl bg-white shadow-[0_4px_12px_rgba(0,0,0,0.05)] transition-transform group-hover:scale-110">
        {icon}
      </div>
      <h3 className="mb-3 text-xl font-bold text-[#1e293b]">{title}</h3>
      <p className="leading-[1.6] text-[#475569]">{description}</p>
    </div>
  );
};

const Step = ({ number, title, description }: { number: string; title: string; description: string }) => {
  return (
    <div className="flex gap-4">
      <div className="mt-1 shrink-0">
        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-[#4f46e5]/10 text-sm font-bold text-[#4f46e5]">
          {number}
        </div>
      </div>
      <div>
        <h3 className="mb-2 text-xl font-bold text-[#1e293b]">{title}</h3>
        <p className="leading-[1.6] text-[#475569]">{description}</p>
      </div>
    </div>
  );
};

export { LandingPage };
