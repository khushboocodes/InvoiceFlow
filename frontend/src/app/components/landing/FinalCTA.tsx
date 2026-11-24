import { Link } from "react-router";
import { ArrowRight, Upload as UploadIcon } from "lucide-react";
import { motion } from "motion/react";

export function FinalCTA() {
  return (
    <section className="section">
      <div className="container-wide">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-100px" }}
          transition={{ duration: 0.6 }}
          className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-[#6C5CE7] via-[#4F46E5] to-[#7c3aed] p-12 md:p-20 text-center"
        >
          {/* Decorative blobs inside CTA */}
          <div className="absolute inset-0 overflow-hidden">
            <div
              className="absolute -top-32 -right-32 w-96 h-96 rounded-full opacity-30"
              style={{
                background: "radial-gradient(circle, white 0%, transparent 70%)",
              }}
            />
            <div
              className="absolute -bottom-32 -left-32 w-96 h-96 rounded-full opacity-30"
              style={{
                background: "radial-gradient(circle, #ec4899 0%, transparent 70%)",
              }}
            />
            <div
              className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] opacity-20"
              style={{
                background:
                  "conic-gradient(from 0deg, transparent, rgba(255,255,255,0.2), transparent)",
                animation: "spin 30s linear infinite",
                borderRadius: "50%",
              }}
            />
          </div>

          {/* Grid pattern */}
          <div
            className="absolute inset-0 opacity-10"
            style={{
              backgroundImage:
                "linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)",
              backgroundSize: "40px 40px",
            }}
          />

          <div className="relative z-10">
            <h2 className="text-4xl md:text-6xl font-bold text-white tracking-tight mb-5 leading-[1.1]">
              Ready to digitize <br className="hidden md:block" />
              your invoice stack?
            </h2>
            <p className="text-lg md:text-xl text-purple-100/90 max-w-xl mx-auto mb-10 leading-relaxed">
              Open the dashboard, drop in a PDF or image, and watch six fields fall into place.
              Runs entirely on your machine, costs nothing.
            </p>
            <div className="flex flex-wrap items-center justify-center gap-3">
              <Link
                to="/upload"
                className="inline-flex items-center gap-2 bg-white text-[#4F46E5] hover:text-[#3730a3] font-bold px-7 py-4 rounded-xl shadow-2xl shadow-black/20 hover:shadow-black/30 hover:-translate-y-0.5 smooth-transition"
              >
                <UploadIcon className="w-4 h-4" />
                Upload an Invoice
              </Link>
              <Link
                to="/dashboard"
                className="inline-flex items-center gap-2 bg-white/10 backdrop-blur-sm border border-white/20 text-white hover:bg-white/20 font-bold px-7 py-4 rounded-xl smooth-transition"
              >
                Explore Dashboard
                <ArrowRight className="w-4 h-4" />
              </Link>
            </div>

            <div className="flex flex-wrap items-center justify-center gap-6 mt-10 text-xs text-purple-200/80 uppercase tracking-wider font-semibold">
              <span>No Sign-up</span>
              <span className="w-1 h-1 rounded-full bg-purple-300/50" />
              <span>No Credit Card</span>
              <span className="w-1 h-1 rounded-full bg-purple-300/50" />
              <span>100% Offline</span>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
