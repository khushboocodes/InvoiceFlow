import { Link } from "react-router";
import { Zap, ArrowRight } from "lucide-react";
import { useEffect, useState } from "react";

export function LandingNav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  const links = [
    { label: "Features", href: "#features" },
    { label: "Pipeline", href: "#pipeline" },
    { label: "Demo", href: "#demo" },
    { label: "Languages", href: "#languages" },
  ];

  return (
    <header
      className={`fixed top-0 left-0 right-0 z-50 smooth-transition ${
        scrolled ? "frost-nav" : "bg-transparent"
      }`}
    >
      <div className="container-wide flex items-center justify-between h-16">
        <Link to="/" className="flex items-center gap-2.5 group">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#6C5CE7] to-[#4F46E5] flex items-center justify-center shadow-lg shadow-purple-500/30 group-hover:scale-105 smooth-transition">
            <Zap className="w-[18px] h-[18px] text-white" strokeWidth={2.5} />
          </div>
          <div className="leading-tight">
            <div className="text-base font-semibold text-gray-900">InvoiceFlow</div>
            <div className="text-[10px] text-gray-500 -mt-0.5 tracking-wide uppercase">
              Document AI
            </div>
          </div>
        </Link>

        <nav className="hidden md:flex items-center gap-8">
          {links.map((l) => (
            <a
              key={l.label}
              href={l.href}
              className="text-sm font-medium text-gray-600 hover:text-gray-900 smooth-transition"
            >
              {l.label}
            </a>
          ))}
        </nav>

        <div className="flex items-center gap-3">
          <Link to="/dashboard" className="btn-primary-3d text-sm py-2.5 px-5">
            Open Dashboard
            <ArrowRight className="w-4 h-4" />
          </Link>
        </div>
      </div>
    </header>
  );
}
