import { Link } from "react-router";
import { ArrowRight, Sparkles, FileText, CheckCircle2, Building2, Hash, IndianRupee, PenLine, Stamp } from "lucide-react";
import { motion, useMotionValue, useSpring, useTransform } from "motion/react";
import { useRef } from "react";

export function Hero() {
  const ref = useRef<HTMLDivElement>(null);
  const mx = useMotionValue(0);
  const my = useMotionValue(0);

  const rotateX = useSpring(useTransform(my, [-0.5, 0.5], [10, -10]), {
    stiffness: 150,
    damping: 20,
  });
  const rotateY = useSpring(useTransform(mx, [-0.5, 0.5], [-15, 15]), {
    stiffness: 150,
    damping: 20,
  });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    mx.set(x);
    my.set(y);
  };

  const handleMouseLeave = () => {
    mx.set(0);
    my.set(0);
  };

  return (
    <section className="relative pt-28 pb-20 md:pt-36 md:pb-32 overflow-hidden">
      {/* Background decoration */}
      <div className="mesh-bg">
        <div className="mesh-blob mesh-blob-1" />
        <div className="mesh-blob mesh-blob-2" />
        <div className="mesh-blob mesh-blob-3" />
      </div>
      <div className="absolute inset-0 grid-pattern" />

      <div className="container-wide relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 lg:gap-8 items-center">
          {/* Left: copy */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/80 border border-purple-200/60 shadow-sm backdrop-blur-sm mb-6">
              <Sparkles className="w-3.5 h-3.5 text-[#6C5CE7]" />
              <span className="text-xs font-semibold text-gray-700">
                100% Offline · Multilingual · Open-Source
              </span>
            </div>

            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold text-gray-900 leading-[1.05] tracking-tight mb-6">
              Turn any invoice into{" "}
              <span className="gradient-text">structured data</span>
              <span className="text-gray-900"> in seconds.</span>
            </h1>

            <p className="text-lg md:text-xl text-gray-600 leading-relaxed mb-8 max-w-xl">
              InvoiceFlow reads PDFs, scanned pages, phone photos, and handwritten invoices in
              English, Hindi, and Gujarati then emits clean JSON with bounding boxes, confidence
              scores, and zero cloud calls.
            </p>

            <div className="flex flex-wrap items-center gap-3 mb-6">
              <Link to="/upload" className="btn-primary-3d">
                Try It Now
                <ArrowRight className="w-4 h-4" />
              </Link>
              <Link to="/dashboard" className="btn-secondary-3d">
                View Dashboard
              </Link>
            </div>

            {/* Supported formats */}
            <div className="flex items-center gap-2 mb-10">
              <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400">
                Accepts
              </span>
              <div className="flex items-center gap-1.5">
                {["PDF", "PNG", "JPG", "JPEG"].map((fmt) => (
                  <span
                    key={fmt}
                    className="px-2 py-0.5 rounded-md text-[10px] font-bold bg-white/80 border border-gray-200 text-gray-700 shadow-sm"
                  >
                    {fmt}
                  </span>
                ))}
              </div>
            </div>

            {/* Mini stats */}
            <div className="flex flex-wrap items-center gap-x-8 gap-y-3">
              <Stat label="Document Accuracy" value="≥95%" />
              <div className="w-px h-8 bg-gray-200 hidden sm:block" />
              <Stat label="Latency / doc" value="<30s" />
              <div className="w-px h-8 bg-gray-200 hidden sm:block" />
              <Stat label="Cost / doc" value="$0" />
            </div>
          </motion.div>

          {/* Right: 3D document */}
          <motion.div
            ref={ref}
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            className="perspective-container relative h-[520px] lg:h-[580px]"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.7, delay: 0.2 }}
          >
            {/* Floating field chips behind the doc */}
            <FloatingChip
              icon={<Building2 className="w-3.5 h-3.5" />}
              label="DEALER"
              value="ABC Tractors Pvt Ltd"
              className="top-[8%] -left-2 lg:-left-8"
              delay={0.6}
            />
            <FloatingChip
              icon={<Hash className="w-3.5 h-3.5" />}
              label="MODEL"
              value="Mahindra 575 DI"
              className="top-[68%] -left-4 lg:-left-12"
              delay={0.9}
              variant="pink"
            />
            <FloatingChip
              icon={<IndianRupee className="w-3.5 h-3.5" />}
              label="ASSET COST"
              value="₹ 5,25,000"
              className="top-[18%] -right-4 lg:-right-8"
              delay={0.75}
              variant="green"
            />
            <FloatingChip
              icon={<PenLine className="w-3.5 h-3.5" />}
              label="HORSEPOWER"
              value="50 HP"
              className="top-[78%] -right-2 lg:-right-4"
              delay={1.05}
              variant="cyan"
            />

            {/* The 3D document */}
            <motion.div
              style={{
                rotateX,
                rotateY,
                transformStyle: "preserve-3d",
              }}
              className="relative w-full h-full flex items-center justify-center"
            >
              <div
                className="relative w-[340px] sm:w-[400px] aspect-[3/4] frost-card-elevated p-7 float-slow"
                style={{ transform: "translateZ(40px)", transformStyle: "preserve-3d" }}
              >
                {/* Doc header */}
                <div className="flex items-center justify-between mb-5 pb-4 border-b border-gray-100">
                  <div className="flex items-center gap-2.5">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#6C5CE7] to-[#4F46E5] flex items-center justify-center">
                      <FileText className="w-4 h-4 text-white" />
                    </div>
                    <div>
                      <div className="text-[11px] uppercase tracking-wider text-gray-500 font-semibold">
                        Tractor Quotation
                      </div>
                      <div className="text-xs text-gray-400">invoice_001.png</div>
                    </div>
                  </div>
                  <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-green-50 border border-green-200">
                    <div className="w-1.5 h-1.5 rounded-full bg-green-500 pulse-dot" />
                    <span className="text-[10px] font-semibold text-green-700">PARSED</span>
                  </div>
                </div>

                {/* Field rows */}
                <div className="space-y-3.5">
                  <FieldRow label="Dealer Name" value="ABC Tractors Pvt Ltd" highlight />
                  <FieldRow label="Model Name" value="Mahindra 575 DI" />
                  <FieldRow label="Horse Power" value="50 HP" />
                  <FieldRow label="Asset Cost" value="₹ 5,25,000" highlight />
                </div>

                {/* Bounding boxes simulation */}
                <div className="mt-5 pt-4 border-t border-gray-100 relative">
                  <div className="text-[10px] uppercase tracking-wider text-gray-500 font-semibold mb-3">
                    Visual Detection
                  </div>
                  <div className="grid grid-cols-2 gap-2.5">
                    <DetectChip icon={<PenLine className="w-3 h-3" />} label="Signature" color="green" />
                    <DetectChip icon={<Stamp className="w-3 h-3" />} label="Stamp" color="blue" />
                  </div>
                </div>

                {/* Decorative bbox overlays */}
                <div
                  className="bbox-overlay"
                  style={{
                    top: "78%",
                    left: "10%",
                    width: "30%",
                    height: "8%",
                    color: "#22c55e",
                    transform: "translateZ(20px)",
                  }}
                >
                  <span className="bbox-label" style={{ background: "#22c55e", color: "white" }}>
                    Signature
                  </span>
                </div>
                <div
                  className="bbox-overlay"
                  style={{
                    top: "78%",
                    right: "10%",
                    width: "28%",
                    height: "8%",
                    color: "#3b82f6",
                    transform: "translateZ(20px)",
                  }}
                >
                  <span className="bbox-label" style={{ background: "#3b82f6", color: "white" }}>
                    Stamp
                  </span>
                </div>
              </div>

              {/* Floating layer behind for depth */}
              <div
                className="absolute inset-0 m-auto w-[340px] sm:w-[400px] aspect-[3/4] frost-card opacity-50"
                style={{
                  transform: "translateZ(-40px) translateX(20px) translateY(20px) rotate(3deg)",
                  filter: "blur(2px)",
                }}
              />
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-2xl font-bold text-gray-900">{value}</div>
      <div className="text-xs text-gray-500 uppercase tracking-wider font-medium">{label}</div>
    </div>
  );
}

function FieldRow({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={`flex items-center justify-between py-2 px-3 rounded-lg ${
        highlight ? "bg-purple-50/60 border border-purple-100" : "bg-gray-50/60"
      }`}
    >
      <span className="text-[11px] uppercase tracking-wider font-semibold text-gray-500">
        {label}
      </span>
      <span className={`text-sm font-semibold ${highlight ? "text-[#6C5CE7]" : "text-gray-900"}`}>
        {value}
      </span>
    </div>
  );
}

function DetectChip({
  icon,
  label,
  color,
}: {
  icon: React.ReactNode;
  label: string;
  color: "green" | "blue";
}) {
  const styles =
    color === "green"
      ? "bg-green-50 border-green-200 text-green-700"
      : "bg-blue-50 border-blue-200 text-blue-700";
  return (
    <div className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border ${styles}`}>
      {icon}
      <span className="text-[11px] font-semibold">{label}</span>
      <CheckCircle2 className="w-3 h-3 ml-auto" />
    </div>
  );
}

function FloatingChip({
  icon,
  label,
  value,
  className = "",
  delay = 0,
  variant = "purple",
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  className?: string;
  delay?: number;
  variant?: "purple" | "green" | "pink" | "cyan";
}) {
  const variants = {
    purple: "from-purple-50 to-white border-purple-200 text-[#6C5CE7]",
    green: "from-green-50 to-white border-green-200 text-green-700",
    pink: "from-pink-50 to-white border-pink-200 text-pink-700",
    cyan: "from-cyan-50 to-white border-cyan-200 text-cyan-700",
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      transition={{ duration: 0.5, delay }}
      className={`absolute z-20 float ${className}`}
    >
      <div
        className={`flex items-center gap-2 px-3 py-2 rounded-xl bg-gradient-to-br ${variants[variant]} border shadow-lg shadow-black/5 backdrop-blur-sm`}
      >
        <div className="opacity-80">{icon}</div>
        <div className="leading-tight">
          <div className="text-[9px] uppercase tracking-wider opacity-70 font-semibold">
            {label}
          </div>
          <div className="text-xs font-bold text-gray-900">{value}</div>
        </div>
      </div>
    </motion.div>
  );
}
