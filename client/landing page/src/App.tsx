import React from 'react';
import { motion } from 'motion/react';
import { 
  BookOpen, 
  ShieldAlert, 
  MessageSquare, 
  BrainCircuit, 
  ArrowRight, 
  CheckCircle2, 
  FileText, 
  Network,
  Lock,
  Users
} from 'lucide-react';

export default function App() {
  return (
    <div className="min-h-screen bg-[#f0f4f8] font-sans text-[#1e293b] selection:bg-indigo-100 selection:text-indigo-900 bg-[radial-gradient(at_0%_0%,_rgba(79,70,229,0.15)_0px,_transparent_50%),radial-gradient(at_100%_0%,_rgba(6,182,212,0.15)_0px,_transparent_50%),radial-gradient(at_50%_100%,_rgba(219,39,119,0.1)_0px,_transparent_50%)] relative">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-white/20 backdrop-blur-md z-50 border-b border-white/40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="text-[#4f46e5]">
              <BookOpen className="w-7 h-7" strokeWidth={2.5} />
            </div>
            <span className="font-extrabold text-2xl tracking-tight text-[#4f46e5]">PolicyPal AI</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm font-medium text-slate-600">
            <a href="#features" className="hover:text-indigo-600 transition-colors">Features</a>
            <a href="#how-it-works" className="hover:text-indigo-600 transition-colors">How it Works</a>
            <a href="#security" className="hover:text-indigo-600 transition-colors">Security</a>
          </div>
          <div className="flex items-center gap-4">
            <button className="hidden md:block text-sm font-medium text-slate-600 hover:text-indigo-600 transition-colors">
              Sign In
            </button>
            <button className="bg-[#4f46e5] hover:bg-indigo-700 text-white px-6 py-2.5 rounded-full text-sm font-semibold transition-all shadow-[0_4px_12px_rgba(79,70,229,0.2)] hover:shadow-[0_6px_16px_rgba(79,70,229,0.3)]">
              Book Demo
            </button>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="pt-32 pb-20 px-4 sm:px-6 lg:px-8 max-w-7xl mx-auto">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="max-w-2xl"
          >
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-white/65 text-[#4f46e5] text-xs font-semibold mb-6 border border-white/40 backdrop-blur-sm">
              <BrainCircuit className="w-4 h-4" />
              <span>Agentic AI for Modern HR Teams</span>
            </div>
            <h1 className="text-5xl lg:text-6xl font-extrabold tracking-tight text-[#0f172a] leading-[1.1] mb-6 -tracking-[2px]">
              Stop answering the same HR questions <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-600 to-violet-600">20 times a day.</span>
            </h1>
            <p className="text-lg text-[#475569] mb-10 leading-[1.6]">
              An intelligent chatbot that knows your employee handbook inside out. It answers questions accurately with policy citations, and seamlessly escalates sensitive issues to human HR.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <button className="bg-[#4f46e5] hover:bg-indigo-700 text-white px-8 py-3.5 rounded-full font-semibold transition-all shadow-[0_4px_12px_rgba(79,70,229,0.2)] flex items-center justify-center gap-2 group cursor-pointer">
                Get Started Free
                <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
              </button>
              <button className="bg-white/65 backdrop-blur-sm hover:bg-white text-[#1e293b] border border-white/40 px-8 py-3.5 rounded-full font-semibold transition-all flex items-center justify-center gap-2 cursor-pointer shadow-[0_4px_12px_rgba(0,0,0,0.05)]">
                <MessageSquare className="w-4 h-4" />
                Try Interactive Demo
              </button>
            </div>
            <div className="mt-8 flex items-center gap-4 text-sm text-slate-500">
              <div className="flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span>RAG-Powered Accuracy</span>
              </div>
              <div className="flex items-center gap-1">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span>MCP Integration</span>
              </div>
            </div>
          </motion.div>

          {/* Hero Visual - Chat Mockup */}
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="relative"
          >
            <div className="absolute inset-0 bg-gradient-to-tr from-indigo-100 to-violet-50 rounded-3xl transform rotate-3 scale-105 -z-10"></div>
            <div className="bg-white/65 backdrop-blur-[16px] border border-white/40 rounded-[24px] shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] flex flex-col h-[480px] p-6">
              <div className="flex items-center gap-3 pb-4 border-b border-black/5 mb-5">
                <div className="w-8 h-8 bg-gradient-to-br from-[#4f46e5] to-[#06b6d4] rounded-full flex items-center justify-center shadow-sm">
                  <BookOpen className="w-4 h-4 text-white" />
                </div>
                <div>
                  <div className="font-bold text-sm text-[#1e293b]">Policy Assistant</div>
                  <div className="text-[10px] text-[#10b981] font-semibold tracking-wider flex items-center gap-1 uppercase">
                    ● Online
                  </div>
                </div>
                <div className="ml-auto text-[10px] uppercase tracking-wider px-2 py-0.5 rounded bg-[#10b981]/10 text-[#10b981]">
                  Agentic Mode
                </div>
              </div>
              <div className="flex-1 overflow-y-auto flex flex-col gap-3">
                {/* User Message */}
                <div className="self-end max-w-[85%]">
                  <div className="bg-[#4f46e5] text-white px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-[1.5]">
                    How many vacation days do I get after 2 years?
                  </div>
                </div>
                {/* AI Message */}
                <div className="self-start max-w-[85%]">
                  <div className="bg-white text-[#1e293b] px-4 py-3 rounded-2xl rounded-bl-sm text-sm leading-[1.5] border border-black/5">
                    <p>Based on the Employee Handbook, employees with 2 years of tenure receive <strong>15 days of Paid Time Off (PTO)</strong> per year.</p>
                    <div className="inline-flex items-center gap-1 mt-2 px-2 py-1 bg-[#f1f5f9] rounded-md text-[11px] font-semibold text-[#4f46e5]">
                      <FileText className="w-3 h-3" />
                      Section 4.1 - Paid Time Off Accrual (Page 22)
                    </div>
                  </div>
                </div>
                {/* User Message - Sensitive */}
                <div className="self-end max-w-[85%] mt-1">
                  <div className="bg-[#4f46e5] text-white px-4 py-3 rounded-2xl rounded-br-sm text-sm leading-[1.5]">
                    I need to request a medical accommodation for my desk.
                  </div>
                </div>
                {/* AI Message - Escalation */}
                <div className="self-start max-w-[85%]">
                  <div className="bg-white text-[#1e293b] px-4 py-3 rounded-2xl rounded-bl-sm text-sm leading-[1.5] border border-black/5">
                    <p>I've detected this as a sensitive matter. I am escalating this conversation to <strong>Sarah (HR Benefits)</strong> immediately. She will reach out within 1 business day.</p>
                    <div className="mt-2 text-[#4f46e5] text-xs font-semibold">Escalation initiated...</div>
                  </div>
                </div>
              </div>
              <div className="h-11 bg-black/5 rounded-xl mt-2.5 flex items-center px-4 text-[13px] text-[#94a3b8]">
                Ask a question about the handbook...
              </div>
            </div>
            <div className="absolute -bottom-[30px] right-10 left-10 h-[60px] bg-white rounded-2xl flex items-center justify-around shadow-[0_10px_25px_rgba(0,0,0,0.05)] px-5 z-10">
                <div className="text-center">
                    <span className="block font-bold text-base text-[#4f46e5]">2,400+</span>
                    <span className="text-[10px] text-[#475569] uppercase">Monthly Queries</span>
                </div>
                <div className="w-px h-[30px] bg-[#e2e8f0]"></div>
                <div className="text-center">
                    <span className="block font-bold text-base text-[#4f46e5]">99.2%</span>
                    <span className="text-[10px] text-[#475569] uppercase">RAG Accuracy</span>
                </div>
                <div className="w-px h-[30px] bg-[#e2e8f0]"></div>
                <div className="text-center">
                    <span className="block font-bold text-base text-[#4f46e5]">12s</span>
                    <span className="text-[10px] text-[#475569] uppercase">Avg Response</span>
                </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="text-3xl font-bold tracking-tight text-slate-900 mb-4">Intelligent, safe, and context-aware.</h2>
            <p className="text-lg text-slate-600">
              Built on advanced RAG (Retrieval-Augmented Generation) and Agentic workflows to ensure every answer is accurate, cited, and appropriate.
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <FeatureCard 
              icon={<FileText className="w-6 h-6 text-indigo-600" />}
              title="RAG-Powered Citations"
              description="Ingests your specific handbook. Every answer includes a direct citation to the exact section and page number, eliminating hallucinations."
            />
            <FeatureCard 
              icon={<ShieldAlert className="w-6 h-6 text-rose-600" />}
              title="Smart Escalation"
              description="Automatically detects sensitive topics like termination, harassment, or accommodations and gracefully routes them to human HR professionals."
            />
            <FeatureCard 
              icon={<Network className="w-6 h-6 text-violet-600" />}
              title="MCP Agentic Architecture"
              description="Uses Model Context Protocol (MCP) to securely connect with your HRIS (Workday, Gusto) to provide personalized answers based on employee context."
            />
          </div>
        </div>
      </section>

      {/* How it Works */}
      <section id="how-it-works" className="py-20 relative">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex flex-col md:flex-row gap-12 items-center">
            <div className="flex-1">
              <h2 className="text-3xl font-bold tracking-tight text-[#0f172a] mb-8">How PolicyPal Works</h2>
              <div className="space-y-8">
                <Step 
                  number="1"
                  title="Upload your Handbook"
                  description="Simply upload your PDF, Word doc, or connect your Notion/Confluence workspace. Our system parses and indexes the text."
                />
                <Step 
                  number="2"
                  title="AI Processes & Indexes (RAG)"
                  description="We create a semantic search index. The agentic AI learns the structure, policies, and nuances of your company's rules."
                />
                <Step 
                  number="3"
                  title="Employees Ask Questions"
                  description="Available via Slack, Microsoft Teams, or a web portal. Employees get instant, accurate answers 24/7."
                />
                <Step 
                  number="4"
                  title="HR Handles the Complex Stuff"
                  description="When ambiguity arises or sensitive topics are detected, the AI creates a ticket with full context for your HR team to take over."
                />
              </div>
            </div>
            <div className="flex-1 w-full">
              <div className="bg-white/65 backdrop-blur-[16px] p-8 rounded-[24px] border border-white/40 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)]">
                <div className="flex items-center gap-4 mb-6 pb-6 border-b border-black/5">
                  <div className="bg-[#4f46e5]/10 p-3 rounded-xl">
                    <Users className="w-6 h-6 text-[#4f46e5]" />
                  </div>
                  <div>
                    <h3 className="font-bold text-[#1e293b]">HR Time Saved</h3>
                    <p className="text-sm text-[#475569]">Average weekly metrics</p>
                  </div>
                </div>
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-[#475569]">Routine Questions Automated</span>
                      <span className="font-bold text-[#4f46e5]">85%</span>
                    </div>
                    <div className="h-2 bg-black/5 rounded-full overflow-hidden">
                      <div className="h-full bg-[#4f46e5] rounded-full w-[85%]"></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-[#475569]">Time to Resolution</span>
                      <span className="font-bold text-[#10b981]">&lt; 2 seconds</span>
                    </div>
                    <div className="h-2 bg-black/5 rounded-full overflow-hidden">
                      <div className="h-full bg-[#10b981] rounded-full w-[95%]"></div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-2">
                      <span className="text-[#475569]">Sensitive Issues Escalated</span>
                      <span className="font-bold text-rose-500">100%</span>
                    </div>
                    <div className="h-2 bg-black/5 rounded-full overflow-hidden">
                      <div className="h-full bg-rose-500 rounded-full w-[100%]"></div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative overflow-hidden mx-4 sm:mx-6 lg:mx-8 mb-12 rounded-[32px] bg-white/65 backdrop-blur-[16px] border border-white/40 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)]">
        <div className="max-w-4xl mx-auto px-4 text-center relative z-10">
          <h2 className="text-4xl font-extrabold mb-6 text-[#0f172a] tracking-tight">Ready to give your HR team their time back?</h2>
          <p className="text-[#475569] text-lg mb-10 max-w-2xl mx-auto leading-[1.6]">
            Join hundreds of forward-thinking companies using PolicyPal to automate routine HR inquiries while maintaining a human touch for what matters.
          </p>
          <div className="flex flex-col sm:flex-row justify-center gap-4">
            <button className="bg-[#4f46e5] text-white px-8 py-4 rounded-full font-semibold hover:bg-indigo-700 transition-colors shadow-[0_4px_12px_rgba(79,70,229,0.2)] cursor-pointer">
              Request a Demo
            </button>
            <button className="bg-white/65 backdrop-blur-sm border border-white/40 text-[#1e293b] px-8 py-4 rounded-full font-semibold hover:bg-white transition-colors cursor-pointer">
              View Pricing
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-slate-950 text-slate-400 py-12 border-t border-slate-900">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 grid grid-cols-2 md:grid-cols-4 gap-8">
          <div className="col-span-2 md:col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <BookOpen className="w-5 h-5 text-indigo-500" />
              <span className="font-semibold text-white">PolicyPal AI</span>
            </div>
            <p className="text-sm">Agentic HR assistance powered by RAG and MCP.</p>
          </div>
          <div>
            <h4 className="text-white font-medium mb-4">Product</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="hover:text-white transition-colors">Features</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Integrations</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Security</a></li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-medium mb-4">Company</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="hover:text-white transition-colors">About Us</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Careers</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Contact</a></li>
            </ul>
          </div>
          <div>
            <h4 className="text-white font-medium mb-4">Legal</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="hover:text-white transition-colors">Privacy Policy</a></li>
              <li><a href="#" className="hover:text-white transition-colors">Terms of Service</a></li>
            </ul>
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mt-12 pt-8 border-t border-slate-800 text-sm flex justify-between items-center">
          <p>© 2026 PolicyPal AI. All rights reserved.</p>
          <div className="flex items-center gap-2">
            <Lock className="w-4 h-4" />
            <span>SOC2 Type II Certified</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode, title: string, description: string }) {
  return (
    <div className="bg-white/65 backdrop-blur-[16px] p-8 rounded-[24px] border border-white/40 shadow-[0_8px_32px_0_rgba(31,38,135,0.07)] hover:shadow-[0_12px_40px_0_rgba(31,38,135,0.1)] transition-all group">
      <div className="bg-white w-12 h-12 rounded-2xl flex items-center justify-center shadow-[0_4px_12px_rgba(0,0,0,0.05)] mb-6 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-bold text-[#1e293b] mb-3">{title}</h3>
      <p className="text-[#475569] leading-[1.6]">{description}</p>
    </div>
  );
}

function Step({ number, title, description }: { number: string, title: string, description: string }) {
  return (
    <div className="flex gap-4">
      <div className="flex-shrink-0 mt-1">
        <div className="w-8 h-8 rounded-full bg-[#4f46e5]/10 text-[#4f46e5] flex items-center justify-center font-bold text-sm">
          {number}
        </div>
      </div>
      <div>
        <h3 className="text-xl font-bold text-[#1e293b] mb-2">{title}</h3>
        <p className="text-[#475569] leading-[1.6]">{description}</p>
      </div>
    </div>
  );
}
