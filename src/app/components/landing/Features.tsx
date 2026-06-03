import { Building2, Hash, Gauge, IndianRupee, PenLine, Stamp } from "lucide-react";
import { motion } from "motion/react";

const features = [
  {
    icon: Building2,
    title: "Dealer Name",
    desc: "Fuzzy-matched against your master list to absorb spelling variations, abbreviations, and OCR noise.",
    matchType: "Fuzzy ≥ 90%",
    accent: "from-purple-100 to-purple-50",
    text: "text-[#6C5CE7]",
  },
  {
    icon: Hash,
    title: "Model Name",
    desc: "Tractor or asset model identifier matched exactly against the asset master for clean reconciliation.",
    matchType: "Exact Match",
    accent: "from-pink-100 to-pink-50",
    text: "text-pink-600",
  },
  {
    icon: Gauge,
    title: "Horse Power",
    desc: "Numeric extraction with vernacular anchor support — handles HP, H.P., एचपी, बल across Indic scripts.",
    matchType: "±5% Tolerance",
    accent: "from-indigo-100 to-indigo-50",
    text: "text-indigo-600",
  },
  {
    icon: IndianRupee,
    title: "Asset Cost",
    desc: "Currency-aware total cost extraction. Strips ₹, Rs., INR symbols and grouping commas automatically.",
    matchType: "±5% Tolerance",
    accent: "from-emerald-100 to-emerald-50",
    text: "text-emerald-600",
  },
  {
    icon: PenLine,
    title: "Dealer Signature",
    desc: "Fine-tuned YOLOv8 detector locates handwritten signatures and returns precise bounding box coordinates.",
    matchType: "IoU ≥ 0.5",
    accent: "from-amber-100 to-amber-50",
    text: "text-amber-600",
  },
  {
    icon: Stamp,
    title: "Dealer Stamp",
    desc: "Identifies circular and rectangular ink stamps even on noisy scans, returning bbox + presence flag.",
    matchType: "IoU ≥ 0.5",
    accent: "from-cyan-100 to-cyan-50",
    text: "text-cyan-600",
  },
];

export function Features() {
  return (
    <section id="features" className="section relative">
      <div className="container-wide relative">
        <div className="max-w-2xl mx-auto text-center mb-16">
          <span className="stage-badge mb-5">Six Fields, One JSON</span>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 tracking-tight mb-5">
            Every signal you need, <span className="gradient-text">extracted with confidence.</span>
          </h2>
          <p className="text-lg text-gray-600 leading-relaxed">
            From typed quotations to handwritten photos in three languages, every field is
            scored, validated, and tied back to its location on the page.
          </p>
        </div>

        <div className="perspective-container">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => {
              const Icon = f.icon;
              return (
                <motion.div
                  key={f.title}
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: "-50px" }}
                  transition={{ duration: 0.5, delay: i * 0.08 }}
                  className="frost-card p-7 tilt-card group cursor-default"
                >
                  <div
                    className={`w-12 h-12 rounded-xl bg-gradient-to-br ${f.accent} border border-white/80 flex items-center justify-center mb-5 shadow-sm`}
                    style={{ transform: "translateZ(20px)" }}
                  >
                    <Icon className={`w-6 h-6 ${f.text}`} strokeWidth={2} />
                  </div>

                  <h3 className="text-xl font-bold text-gray-900 mb-2">{f.title}</h3>
                  <p className="text-sm text-gray-600 leading-relaxed mb-5">{f.desc}</p>

                  <div className="flex items-center gap-2">
                    <div className={`w-1.5 h-1.5 rounded-full ${f.text.replace("text-", "bg-")}`} />
                    <span className={`text-xs font-bold uppercase tracking-wider ${f.text}`}>
                      {f.matchType}
                    </span>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </div>
    </section>
  );
}
