import { FileInput, ScanLine, Eye, Sparkles, ShieldCheck, FileJson, ArrowRight } from "lucide-react";
import { motion } from "motion/react";

const stages = [
  {
    icon: FileInput,
    title: "Ingestion",
    desc: "Accepts PDFs, scans, and phone photos. Rasterized at 300 DPI, deskewed, denoised.",
    tech: "PyMuPDF · OpenCV",
    color: "#6C5CE7",
  },
  {
    icon: ScanLine,
    title: "OCR",
    desc: "Multilingual text + bounding boxes for English, Hindi, Gujarati.",
    tech: "PaddleOCR PP-OCRv4",
    color: "#8b5cf6",
  },
  {
    icon: Eye,
    title: "Vision",
    desc: "Fine-tuned detector locates dealer signatures and stamps.",
    tech: "YOLOv8n",
    color: "#ec4899",
  },
  {
    icon: Sparkles,
    title: "Extraction",
    desc: "Anchored regex first, then a 1.5B SLM fallback for tricky layouts.",
    tech: "Rules + Qwen2.5",
    color: "#3b82f6",
  },
  {
    icon: ShieldCheck,
    title: "Validation",
    desc: "Fuzzy match against masters, range checks, and consistency rules.",
    tech: "RapidFuzz · Pydantic",
    color: "#10b981",
  },
  {
    icon: FileJson,
    title: "Output",
    desc: "Schema-validated JSON with per-field confidence and timing.",
    tech: "Pydantic v2",
    color: "#f59e0b",
  },
];

export function Pipeline() {
  return (
    <section id="pipeline" className="section relative bg-gradient-to-b from-transparent via-purple-50/30 to-transparent">
      <div className="container-wide relative">
        <div className="max-w-2xl mx-auto text-center mb-16">
          <span className="stage-badge mb-5">The Pipeline</span>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 tracking-tight mb-5">
            Six stages, all running <span className="gradient-text">on your machine.</span>
          </h2>
          <p className="text-lg text-gray-600 leading-relaxed">
            No API keys. No outbound calls. Every model and tokenizer ships inside the bundle so
            you can run it air-gapped in a banking environment.
          </p>
        </div>

        {/* Desktop horizontal flow */}
        <div className="hidden lg:block">
          <div className="relative">
            {/* connecting line */}
            <div className="absolute top-[68px] left-[8%] right-[8%] h-px bg-gradient-to-r from-transparent via-purple-300/60 to-transparent" />

            <div className="grid grid-cols-6 gap-4 relative">
              {stages.map((s, i) => {
                const Icon = s.icon;
                return (
                  <motion.div
                    key={s.title}
                    initial={{ opacity: 0, y: 30 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, margin: "-80px" }}
                    transition={{ duration: 0.5, delay: i * 0.1 }}
                    className="relative"
                  >
                    {/* Stage number */}
                    <div className="text-center mb-3">
                      <span className="text-xs font-bold text-gray-400 tracking-widest">
                        STAGE {String(i + 1).padStart(2, "0")}
                      </span>
                    </div>

                    {/* Icon node */}
                    <div className="flex justify-center mb-5 relative">
                      <div
                        className="w-16 h-16 rounded-2xl bg-white border border-gray-200 flex items-center justify-center shadow-lg relative z-10"
                        style={{
                          boxShadow: `0 8px 24px -8px ${s.color}40`,
                        }}
                      >
                        <Icon className="w-7 h-7" style={{ color: s.color }} strokeWidth={2} />
                      </div>
                      {/* glow ring */}
                      <div
                        className="absolute inset-0 m-auto w-16 h-16 rounded-2xl opacity-40 blur-md"
                        style={{ background: s.color }}
                      />
                    </div>

                    <div className="text-center px-1">
                      <h3 className="text-base font-bold text-gray-900 mb-1.5">{s.title}</h3>
                      <p className="text-xs text-gray-600 leading-relaxed mb-3">{s.desc}</p>
                      <div
                        className="inline-block text-[10px] font-semibold px-2 py-1 rounded-md"
                        style={{
                          background: `${s.color}15`,
                          color: s.color,
                        }}
                      >
                        {s.tech}
                      </div>
                    </div>

                    {/* Arrow between stages */}
                    {i < stages.length - 1 && (
                      <div className="absolute top-[68px] -right-2 z-20">
                        <ArrowRight className="w-4 h-4 text-purple-400" />
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>
          </div>
        </div>

        {/* Mobile/tablet vertical flow */}
        <div className="lg:hidden space-y-4 max-w-md mx-auto">
          {stages.map((s, i) => {
            const Icon = s.icon;
            return (
              <motion.div
                key={s.title}
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                viewport={{ once: true }}
                transition={{ duration: 0.5, delay: i * 0.05 }}
                className="frost-card p-5 flex items-start gap-4"
              >
                <div
                  className="w-12 h-12 rounded-xl flex-shrink-0 flex items-center justify-center shadow-sm border border-white/60"
                  style={{
                    background: `linear-gradient(135deg, ${s.color}25, ${s.color}10)`,
                  }}
                >
                  <Icon className="w-6 h-6" style={{ color: s.color }} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="text-[10px] font-bold text-gray-400 tracking-widest mb-0.5">
                    STAGE {String(i + 1).padStart(2, "0")}
                  </div>
                  <h3 className="text-base font-bold text-gray-900 mb-1">{s.title}</h3>
                  <p className="text-sm text-gray-600 leading-relaxed mb-2">{s.desc}</p>
                  <div
                    className="inline-block text-[10px] font-semibold px-2 py-1 rounded-md"
                    style={{ background: `${s.color}15`, color: s.color }}
                  >
                    {s.tech}
                  </div>
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
