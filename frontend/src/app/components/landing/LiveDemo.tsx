import { motion } from "motion/react";
import { useEffect, useState } from "react";
import { Copy, Check, FileText, PenLine, Stamp } from "lucide-react";

const SAMPLE_JSON = `{
  "doc_id": "invoice_001",
  "fields": {
    "dealer_name": "ABC Tractors Pvt Ltd",
    "model_name": "Mahindra 575 DI",
    "horse_power": 50,
    "asset_cost": 525000,
    "signature": {
      "present": true,
      "bbox": [100, 200, 300, 250]
    },
    "stamp": {
      "present": true,
      "bbox": [400, 500, 500, 550]
    }
  },
  "confidence": 0.96,
  "processing_time_sec": 3.8,
  "cost_estimate_usd": 0.0002
}`;

export function LiveDemo() {
  const [typed, setTyped] = useState("");
  const [started, setStarted] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!started) return;
    let i = 0;
    const interval = setInterval(() => {
      i += 4;
      setTyped(SAMPLE_JSON.slice(0, i));
      if (i >= SAMPLE_JSON.length) clearInterval(interval);
    }, 18);
    return () => clearInterval(interval);
  }, [started]);

  const handleCopy = () => {
    navigator.clipboard.writeText(SAMPLE_JSON);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <section id="demo" className="section relative">
      <div className="container-wide relative">
        <div className="max-w-2xl mx-auto text-center mb-14">
          <span className="stage-badge mb-5">Live Output</span>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 tracking-tight mb-5">
            One file in. <span className="gradient-text">One JSON out.</span>
          </h2>
          <p className="text-lg text-gray-600 leading-relaxed">
            Schema-validated, confidence-scored, ready for direct database insertion or downstream
            credit decisioning.
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          onViewportEnter={() => setStarted(true)}
          transition={{ duration: 0.6 }}
          className="grid lg:grid-cols-2 gap-6 items-stretch"
        >
          {/* Left: doc preview */}
          <div className="frost-card-elevated p-6 lg:p-8 relative overflow-hidden">
            <div className="flex items-center justify-between mb-5">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#6C5CE7] to-[#4F46E5] flex items-center justify-center">
                  <FileText className="w-4 h-4 text-white" />
                </div>
                <div>
                  <div className="text-sm font-semibold text-gray-900">invoice_001.png</div>
                  <div className="text-xs text-gray-500">Tractor Quotation · Image · 1.2 MB</div>
                </div>
              </div>
              <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-green-50 border border-green-200">
                <div className="w-1.5 h-1.5 rounded-full bg-green-500 pulse-dot" />
                <span className="text-[10px] font-semibold text-green-700">EXTRACTED</span>
              </div>
            </div>

            {/* Mock doc */}
            <div className="relative aspect-[3/4] bg-gradient-to-br from-gray-50 to-white rounded-xl border border-gray-200 overflow-hidden p-6">
              {/* Letterhead */}
              <div className="text-center mb-4 pb-3 border-b border-gray-200">
                <div className="text-base font-bold text-gray-800 tracking-wide">
                  ABC TRACTORS PVT LTD
                </div>
                <div className="text-[10px] text-gray-500">Authorized Mahindra Dealer · Gujarat</div>
              </div>

              {/* Quote rows */}
              <div className="space-y-2 mb-6">
                <DocRow label="Quote No." value="Q-2024-0142" />
                <DocRow label="Date" value="14 Apr 2024" />
                <DocRow label="Customer" value="[redacted]" muted />
              </div>

              <div className="border-t border-gray-200 pt-3 space-y-2 mb-6">
                <DocRow label="Tractor Model" value="Mahindra 575 DI" highlighted />
                <DocRow label="Horse Power" value="50 HP" highlighted />
                <DocRow label="Engine" value="4-cyl, 2730 cc" />
              </div>

              <div className="border-t border-gray-200 pt-3 mb-8">
                <div className="flex items-center justify-between bg-purple-50 -mx-2 px-2 py-1.5 rounded">
                  <span className="text-xs font-bold text-gray-700">TOTAL COST</span>
                  <span className="text-base font-bold text-[#6C5CE7]">₹ 5,25,000</span>
                </div>
              </div>

              {/* Footer with sig + stamp areas */}
              <div className="absolute bottom-6 left-6 right-6 flex items-end justify-between">
                <div className="relative">
                  <div className="w-32 h-12 italic text-gray-700 text-lg" style={{ fontFamily: "cursive" }}>
                    Signature
                  </div>
                  <div className="text-[9px] text-gray-500 mt-1">Dealer Signature</div>
                  <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 1, duration: 0.4 }}
                    className="absolute -inset-1.5 border-2 border-green-500 rounded-md"
                  >
                    <span
                      className="absolute -top-5 left-0 text-[9px] font-bold px-1.5 py-0.5 rounded bg-green-500 text-white tracking-wider"
                    >
                      <PenLine className="w-2.5 h-2.5 inline -mt-0.5 mr-0.5" />
                      SIG · 0.94
                    </span>
                  </motion.div>
                </div>

                <div className="relative">
                  <div className="w-20 h-20 rounded-full border-2 border-gray-400 flex items-center justify-center text-[8px] text-gray-500 text-center px-1">
                    AUTH<br/>STAMP
                  </div>
                  <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    transition={{ delay: 1.3, duration: 0.4 }}
                    className="absolute -inset-1.5 border-2 border-blue-500 rounded-md"
                  >
                    <span
                      className="absolute -top-5 left-0 text-[9px] font-bold px-1.5 py-0.5 rounded bg-blue-500 text-white tracking-wider"
                    >
                      <Stamp className="w-2.5 h-2.5 inline -mt-0.5 mr-0.5" />
                      STAMP · 0.91
                    </span>
                  </motion.div>
                </div>
              </div>
            </div>
          </div>

          {/* Right: JSON output */}
          <div className="frost-card-elevated overflow-hidden flex flex-col">
            <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200/70 bg-gradient-to-r from-gray-50/50 to-transparent">
              <div className="flex items-center gap-2">
                <div className="flex gap-1.5">
                  <div className="w-2.5 h-2.5 rounded-full bg-red-400" />
                  <div className="w-2.5 h-2.5 rounded-full bg-yellow-400" />
                  <div className="w-2.5 h-2.5 rounded-full bg-green-400" />
                </div>
                <span className="text-xs font-semibold text-gray-600 ml-3">result.json</span>
              </div>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1.5 text-xs font-semibold text-gray-600 hover:text-[#6C5CE7] smooth-transition px-2.5 py-1.5 rounded-md hover:bg-purple-50"
              >
                {copied ? (
                  <>
                    <Check className="w-3.5 h-3.5" /> Copied
                  </>
                ) : (
                  <>
                    <Copy className="w-3.5 h-3.5" /> Copy
                  </>
                )}
              </button>
            </div>

            <div className="flex-1 bg-gray-900 p-6 font-mono text-sm overflow-auto custom-scrollbar relative">
              <pre className="text-gray-100 leading-relaxed">
                <code>{colorizeJson(typed)}</code>
                {typed.length < SAMPLE_JSON.length && (
                  <span className="cursor-blink text-purple-400">▊</span>
                )}
              </pre>
            </div>

            <div className="px-6 py-3 border-t border-gray-200/70 bg-gradient-to-r from-purple-50/30 to-transparent flex items-center justify-between text-xs">
              <div className="flex items-center gap-4 text-gray-600">
                <span className="font-semibold">Confidence: <span className="text-green-600">96%</span></span>
                <span>Processed in <span className="font-semibold text-gray-900">3.8s</span></span>
              </div>
              <span className="text-gray-500">All 6 fields ✓</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}

function DocRow({
  label,
  value,
  highlighted,
  muted,
}: {
  label: string;
  value: string;
  highlighted?: boolean;
  muted?: boolean;
}) {
  return (
    <div className="flex items-center justify-between text-xs">
      <span className="text-gray-500">{label}</span>
      <span
        className={`font-semibold ${
          highlighted ? "text-[#6C5CE7]" : muted ? "text-gray-400 italic" : "text-gray-900"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

function colorizeJson(text: string) {
  if (!text) return null;
  // Simple syntax highlighting
  const parts: { text: string; cls: string }[] = [];
  const regex = /("(?:\\.|[^"\\])*"\s*:)|("(?:\\.|[^"\\])*")|(\b(?:true|false|null)\b)|(-?\d+\.?\d*)|([{}\[\],])/g;
  let lastIdx = 0;
  let m: RegExpExecArray | null;
  while ((m = regex.exec(text)) !== null) {
    if (m.index > lastIdx) parts.push({ text: text.slice(lastIdx, m.index), cls: "text-gray-300" });
    if (m[1]) parts.push({ text: m[1], cls: "text-purple-300" });
    else if (m[2]) parts.push({ text: m[2], cls: "text-emerald-300" });
    else if (m[3]) parts.push({ text: m[3], cls: "text-orange-300" });
    else if (m[4]) parts.push({ text: m[4], cls: "text-amber-200" });
    else if (m[5]) parts.push({ text: m[5], cls: "text-gray-400" });
    lastIdx = regex.lastIndex;
  }
  if (lastIdx < text.length) parts.push({ text: text.slice(lastIdx), cls: "text-gray-300" });
  return parts.map((p, i) => (
    <span key={i} className={p.cls}>
      {p.text}
    </span>
  ));
}
