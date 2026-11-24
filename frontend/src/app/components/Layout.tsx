import { Outlet, NavLink, Link, useLocation } from "react-router";
import { useEffect } from "react";
import { LayoutDashboard, FileText, Upload, BarChart3, Search, Zap, Home, Sparkles } from "lucide-react";
import { useDocumentStore } from "../store/documentStore";
import { fetchDocuments } from "../lib/api";

export function Layout() {
  const location = useLocation();
  const setDocuments = useDocumentStore((s) => s.setDocuments);
  const setHydrated = useDocumentStore((s) => s.setHydrated);

  // Server-side hydrate. Runs once on mount AND every 5 seconds so that
  // documents uploaded from a different browser show up here within a few
  // seconds. Falls back gracefully when the bridge is offline — the local
  // localStorage cache is already populated by the persist middleware.
  useEffect(() => {
    let cancelled = false;
    const sync = async () => {
      try {
        const docs = await fetchDocuments();
        if (!cancelled) {
          setDocuments(docs as never);
        }
      } catch {
        // Bridge unreachable — keep whatever we have locally.
      } finally {
        if (!cancelled) setHydrated(true);
      }
    };
    sync();
    const interval = setInterval(sync, 5000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [setDocuments, setHydrated]);

  const navigation = [
    { name: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
    { name: "Documents", path: "/documents", icon: FileText },
    { name: "Upload", path: "/upload", icon: Upload },
    { name: "Analytics", path: "/analytics", icon: BarChart3 },
  ];

  const getPageTitle = () => {
    if (location.pathname === "/dashboard") return "Dashboard";
    if (location.pathname === "/documents") return "Documents";
    if (location.pathname === "/upload") return "Upload Document";
    if (location.pathname === "/analytics") return "Analytics";
    if (location.pathname.startsWith("/results")) return "Extraction Result";
    return "Dashboard";
  };

  const getPageSubtitle = () => {
    if (location.pathname === "/dashboard") return "Real-time view of your extraction pipeline";
    if (location.pathname === "/documents") return "All processed and pending documents";
    if (location.pathname === "/upload") return "Drop a file to start AI extraction";
    if (location.pathname === "/analytics") return "Performance and accuracy insights";
    if (location.pathname.startsWith("/results")) return "Extracted fields and JSON output";
    return "";
  };

  return (
    <div className="flex h-screen bg-[#F8FAFC] overflow-hidden relative">
      {/* Ambient mesh backdrop — subtle, fixed behind everything */}
      <div className="fixed inset-0 pointer-events-none z-0">
        <div className="mesh-bg">
          <div className="mesh-blob mesh-blob-1" style={{ opacity: 0.25 }} />
          <div className="mesh-blob mesh-blob-2" style={{ opacity: 0.2 }} />
          <div className="mesh-blob mesh-blob-3" style={{ opacity: 0.2 }} />
        </div>
      </div>

      {/* Sidebar — light, frosted, premium SaaS */}
      <aside className="relative z-10 w-64 bg-white/80 backdrop-blur-xl border-r border-gray-200/70 flex flex-col">
        {/* Logo */}
        <Link to="/" className="p-6 pb-7 block hover:opacity-90 smooth-transition">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-[#6C5CE7] to-[#4F46E5] flex items-center justify-center shadow-lg shadow-purple-500/30">
                <Zap className="w-5 h-5 text-white" strokeWidth={2.5} />
              </div>
              <div className="absolute -bottom-0.5 -right-0.5 w-3 h-3 rounded-full bg-green-500 border-2 border-white pulse-dot" />
            </div>
            <div>
              <h1 className="text-lg text-gray-900 font-bold leading-tight">InvoiceFlow</h1>
              <p className="text-[10px] text-gray-500 uppercase tracking-wider font-semibold">
                Document AI
              </p>
            </div>
          </div>
        </Link>

        {/* Workspace card */}
        <div className="px-4 mb-4">
          <div className="rounded-xl bg-gradient-to-br from-purple-50 via-white to-indigo-50 border border-purple-100 p-3">
            <div className="flex items-center gap-2 mb-1.5">
              <Sparkles className="w-3.5 h-3.5 text-[#6C5CE7]" />
              <span className="text-[10px] uppercase tracking-wider text-[#6C5CE7] font-bold">
                Workspace
              </span>
            </div>
            <div className="text-sm font-semibold text-gray-900">My Pipeline</div>
            <div className="text-[10px] text-gray-500 mt-0.5">Local · Offline</div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 space-y-1">
          <div className="px-3 mb-2">
            <span className="text-[10px] uppercase tracking-wider text-gray-400 font-semibold">
              Menu
            </span>
          </div>
          {navigation.map((item) => {
            const isActive = location.pathname.startsWith(item.path);
            const Icon = item.icon;

            return (
              <NavLink
                key={item.name}
                to={item.path}
                className={`
                  group relative flex items-center gap-3 px-3 py-2.5 rounded-lg smooth-transition
                  ${
                    isActive
                      ? "bg-gradient-to-r from-[#6C5CE7] to-[#4F46E5] text-white shadow-md shadow-purple-500/30"
                      : "text-gray-600 hover:bg-gray-100/70 hover:text-gray-900"
                  }
                `}
              >
                <Icon className="w-4 h-4" strokeWidth={isActive ? 2.4 : 2} />
                <span className="text-sm font-medium">{item.name}</span>
                {isActive && (
                  <div className="absolute right-2 w-1.5 h-1.5 rounded-full bg-white/80" />
                )}
              </NavLink>
            );
          })}
        </nav>

        {/* Bottom — back to landing only */}
        <div className="p-3 border-t border-gray-200/70">
          <Link
            to="/"
            className="flex items-center gap-3 px-3 py-2.5 w-full text-gray-500 hover:bg-gray-100/70 hover:text-gray-900 rounded-lg smooth-transition"
          >
            <Home className="w-4 h-4" />
            <span className="text-sm font-medium">Back to Site</span>
          </Link>
        </div>
      </aside>

      {/* Main Content */}
      <div className="relative z-10 flex-1 flex flex-col overflow-hidden">
        {/* Top Bar — frosted */}
        <header className="h-20 bg-white/70 backdrop-blur-xl border-b border-gray-200/70 flex items-center justify-between px-8 shrink-0">
          <div>
            <h2 className="text-xl font-bold text-gray-900 leading-tight">
              {getPageTitle()}
            </h2>
            {getPageSubtitle() && (
              <p className="text-xs text-gray-500 mt-0.5">{getPageSubtitle()}</p>
            )}
          </div>

          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                placeholder="Search documents..."
                className="pl-10 pr-4 py-2.5 w-80 rounded-xl bg-gray-50/80 border border-gray-200 text-sm text-gray-700 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#6C5CE7]/25 focus:border-[#6C5CE7] focus:bg-white smooth-transition"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
                <kbd className="text-[10px] font-semibold text-gray-400 bg-white border border-gray-200 rounded px-1.5 py-0.5">
                  ⌘K
                </kbd>
              </div>
            </div>

            <div className="flex items-center gap-2 px-3 py-2 rounded-full bg-green-50 border border-green-200">
              <div className="w-2 h-2 rounded-full bg-green-500 pulse-dot" />
              <span className="text-xs font-semibold text-green-700">System Online</span>
            </div>
          </div>
        </header>

        <main className="flex-1 overflow-auto custom-scrollbar">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
