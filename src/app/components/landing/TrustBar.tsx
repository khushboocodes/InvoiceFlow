import { Shield, Zap, Globe, Lock, Cpu, Cloud } from "lucide-react";

export function TrustBar() {
  const items = [
    { icon: Shield, label: "≥95% Accuracy" },
    { icon: Zap, label: "<30s per Doc" },
    { icon: Lock, label: "100% Offline" },
    { icon: Globe, label: "EN · HI · GU" },
    { icon: Cpu, label: "Runs on CPU" },
    { icon: Cloud, label: "Zero Cloud Calls" },
  ];

  return (
    <section className="relative py-12 border-y border-gray-200/70 bg-white/40 backdrop-blur-sm">
      <div className="container-wide">
        <div className="text-center text-xs font-semibold uppercase tracking-wider text-gray-500 mb-7">
          Built for production banking workflows
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-y-6 gap-x-4">
          {items.map((it) => {
            const Icon = it.icon;
            return (
              <div
                key={it.label}
                className="flex items-center justify-center gap-2.5 group"
              >
                <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-purple-50 to-indigo-50 border border-purple-100 flex items-center justify-center group-hover:scale-105 smooth-transition">
                  <Icon className="w-4 h-4 text-[#6C5CE7]" strokeWidth={2.2} />
                </div>
                <span className="text-sm font-semibold text-gray-700">{it.label}</span>
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
