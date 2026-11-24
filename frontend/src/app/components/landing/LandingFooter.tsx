import { Zap, Github } from "lucide-react";

export function LandingFooter() {
  return (
    <footer className="border-t border-gray-200/70 bg-white/40 backdrop-blur-sm">
      <div className="container-wide py-12">
        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#6C5CE7] to-[#4F46E5] flex items-center justify-center shadow-md">
              <Zap className="w-4 h-4 text-white" strokeWidth={2.5} />
            </div>
            <div>
              <div className="text-sm font-semibold text-gray-900">InvoiceFlow</div>
              <div className="text-[10px] text-gray-500 -mt-0.5 tracking-wide uppercase">
                Document AI
              </div>
            </div>
          </div>

          <div className="flex items-center gap-8 text-sm text-gray-600">
            <a href="#features" className="hover:text-gray-900 smooth-transition">
              Features
            </a>
            <a href="#pipeline" className="hover:text-gray-900 smooth-transition">
              Pipeline
            </a>
            <a href="#demo" className="hover:text-gray-900 smooth-transition">
              Demo
            </a>
            <a
              href="#"
              className="flex items-center gap-1.5 hover:text-gray-900 smooth-transition"
            >
              <Github className="w-4 h-4" /> GitHub
            </a>
          </div>

          <div className="text-xs text-gray-500">
            Built for IDFC GenAI · Convolve 4.0
          </div>
        </div>
      </div>
    </footer>
  );
}
