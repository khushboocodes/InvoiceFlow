import { FileText, CheckCircle2, Zap, Clock, Upload as UploadIcon, Eye, TrendingUp, Sparkles } from "lucide-react";
import { Link } from "react-router";
import { useDocumentStore } from "../store/documentStore";
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

export function Dashboard() {
  const documents = useDocumentStore((state) => state.documents);

  const totalDocs = documents.length;
  const completedDocs = documents.filter((d) => d.status === "completed").length;
  const accuracyRate = totalDocs > 0 ? ((completedDocs / totalDocs) * 100).toFixed(0) : 0;
  const avgConfidence =
    totalDocs > 0
      ? documents.filter((d) => d.confidence).reduce((acc, d) => acc + (d.confidence || 0), 0) /
          completedDocs || 0
      : 0;
  const avgProcessing =
    totalDocs > 0
      ? documents.filter((d) => d.processingTime).reduce(
          (acc, d) => acc + parseFloat(d.processingTime || "0"),
          0
        ) / completedDocs || 0
      : 0;

  // Real volume data: count how many documents were uploaded on each of the
  // last 7 calendar days. Falls back to a flat empty series when there are
  // no documents yet.
  const volumeData = (() => {
    const today = new Date();
    const days: { date: string; value: number }[] = [];
    for (let i = 6; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(today.getDate() - i);
      const key = d.toISOString().slice(0, 10); // YYYY-MM-DD
      const label = d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
      const count = documents.filter((doc) => doc.uploadedAt.slice(0, 10) === key).length;
      days.push({ date: label, value: count });
    }
    return days;
  })();

  // Real confidence distribution across the four standard buckets.
  const confidenceData = (() => {
    const buckets = { "90-100%": 0, "80-90%": 0, "70-80%": 0, "<70%": 0 };
    for (const d of documents) {
      if (d.status !== "completed" || !d.confidence) continue;
      if (d.confidence >= 90) buckets["90-100%"]++;
      else if (d.confidence >= 80) buckets["80-90%"]++;
      else if (d.confidence >= 70) buckets["70-80%"]++;
      else buckets["<70%"]++;
    }
    return Object.entries(buckets).map(([range, value]) => ({ range, value }));
  })();

  // Real language distribution. Uses the language tag the backend returns.
  const languageData = (() => {
    const counts = new Map<string, number>();
    for (const d of documents) {
      if (d.status !== "completed") continue;
      const lang = d.language || "Unknown";
      counts.set(lang, (counts.get(lang) ?? 0) + 1);
    }
    const palette = ["#6C5CE7", "#10b981", "#f59e0b", "#ec4899", "#0ea5e9", "#8b5cf6"];
    return Array.from(counts.entries()).map(([name, value], i) => ({
      name,
      value,
      color: palette[i % palette.length],
    }));
  })();

  return (
    <div className="p-8">
      {/* Welcome Section + Action */}
      <div className="flex items-start justify-between mb-8 gap-6 flex-wrap">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-purple-50 border border-purple-100 mb-3">
            <Sparkles className="w-3 h-3 text-[#6C5CE7]" />
            <span className="text-[10px] font-bold text-[#6C5CE7] uppercase tracking-wider">
              All Systems Operational
            </span>
          </div>
          <h1 className="text-3xl font-bold text-gray-900 mb-1.5 tracking-tight">
            Welcome back <span className="inline-block">👋</span>
          </h1>
          <p className="text-gray-600 text-sm">
            Here's what's happening with your documents today.
          </p>
        </div>

        <Link to="/upload">
          <button className="btn-primary-3d text-sm">
            <UploadIcon className="w-4 h-4" />
            Process Document
          </button>
        </Link>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5 mb-8">
        <StatCard
          icon={<FileText className="w-5 h-5" />}
          iconBg="from-purple-100 to-purple-50"
          iconColor="text-[#6C5CE7]"
          value={totalDocs.toString()}
          label="Total Documents"
          change="+12%"
          accent="#6C5CE7"
        />
        <StatCard
          icon={<CheckCircle2 className="w-5 h-5" />}
          iconBg="from-emerald-100 to-emerald-50"
          iconColor="text-emerald-600"
          value={`${accuracyRate}%`}
          label="Accuracy Rate"
          change="+3%"
          accent="#10b981"
        />
        <StatCard
          icon={<Zap className="w-5 h-5" />}
          iconBg="from-amber-100 to-amber-50"
          iconColor="text-amber-600"
          value={`${avgConfidence.toFixed(1)}%`}
          label="Avg Confidence"
          accent="#f59e0b"
        />
        <StatCard
          icon={<Clock className="w-5 h-5" />}
          iconBg="from-pink-100 to-pink-50"
          iconColor="text-pink-600"
          value={avgProcessing > 0 ? `${avgProcessing.toFixed(1)}s` : "0s"}
          label="Avg Processing"
          accent="#ec4899"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-8">
        <ChartCard
          title="Documents (7d)"
          subtitle="Daily processing volume"
          empty={totalDocs === 0}
        >
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={volumeData}>
              <defs>
                <linearGradient id="lineFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6C5CE7" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#6C5CE7" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
              <XAxis
                dataKey="date"
                tick={{ fontSize: 11, fill: "#6B7280" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "#6B7280" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                }}
              />
              <Line
                type="monotone"
                dataKey="value"
                stroke="#6C5CE7"
                strokeWidth={2.5}
                fill="url(#lineFill)"
                dot={{ fill: "#6C5CE7", r: 4, strokeWidth: 2, stroke: "#fff" }}
                activeDot={{ r: 6, strokeWidth: 2, stroke: "#fff" }}
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Confidence Breakdown"
          subtitle="Distribution by confidence range"
          empty={completedDocs === 0}
        >
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={confidenceData}>
              <defs>
                <linearGradient id="barFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6C5CE7" stopOpacity={1} />
                  <stop offset="100%" stopColor="#4F46E5" stopOpacity={0.7} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
              <XAxis
                dataKey="range"
                tick={{ fontSize: 11, fill: "#6B7280" }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 11, fill: "#6B7280" }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  borderRadius: "8px",
                  border: "1px solid #e5e7eb",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
                }}
              />
              <Bar dataKey="value" fill="url(#barFill)" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Language Distribution"
          subtitle="Documents by detected language"
          empty={completedDocs === 0}
          emptyText="No language data yet"
        >
          <div className="flex items-center justify-center h-[200px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={languageData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={75}
                  paddingAngle={2}
                  dataKey="value"
                >
                  {languageData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div className="ml-4 space-y-1.5">
              {languageData.map((entry) => (
                <div key={entry.name} className="flex items-center gap-2">
                  <div
                    className="w-2.5 h-2.5 rounded-full"
                    style={{ backgroundColor: entry.color }}
                  />
                  <span className="text-sm text-gray-700 font-medium">
                    {entry.name} ({entry.value})
                  </span>
                </div>
              ))}
            </div>
          </div>
        </ChartCard>
      </div>

      {/* Recent Documents or Empty State */}
      {totalDocs === 0 ? (
        <div className="frost-card-elevated rounded-2xl p-12 lg:p-16 text-center relative overflow-hidden">
          <div
            className="absolute inset-0 pointer-events-none opacity-50"
            style={{
              background:
                "radial-gradient(circle at 50% 0%, rgba(108, 92, 231, 0.08), transparent 60%)",
            }}
          />
          <div className="relative">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-purple-100 to-indigo-50 border border-purple-100 flex items-center justify-center mx-auto mb-5 shadow-lg shadow-purple-500/15 float">
              <UploadIcon className="w-9 h-9 text-[#6C5CE7]" />
            </div>
            <h3 className="text-xl font-bold text-gray-900 mb-2">No documents yet</h3>
            <p className="text-gray-600 mb-7 max-w-md mx-auto text-sm">
              Upload your first invoice to get started with AI extraction. Drop a PDF, PNG, or
              JPG and InvoiceFlow handles the rest.
            </p>
            <Link to="/upload">
              <button className="btn-primary-3d">
                <UploadIcon className="w-4 h-4" />
                Upload Document
              </button>
            </Link>
          </div>
        </div>
      ) : (
        <div className="frost-card-elevated rounded-2xl p-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-bold text-gray-900">Recent Documents</h3>
              <span className="text-xs font-semibold text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                {documents.length}
              </span>
            </div>
            <Link
              to="/documents"
              className="text-sm text-[#6C5CE7] hover:text-[#4F46E5] font-semibold flex items-center gap-1 smooth-transition"
            >
              View all
              <span>→</span>
            </Link>
          </div>

          <div className="overflow-x-auto -mx-2">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left text-[10px] font-bold text-gray-500 uppercase tracking-wider pb-3 px-2">
                    File
                  </th>
                  <th className="text-left text-[10px] font-bold text-gray-500 uppercase tracking-wider pb-3 px-2">
                    Status
                  </th>
                  <th className="text-left text-[10px] font-bold text-gray-500 uppercase tracking-wider pb-3 px-2">
                    Confidence
                  </th>
                  <th className="text-left text-[10px] font-bold text-gray-500 uppercase tracking-wider pb-3 px-2">
                    Language
                  </th>
                  <th className="text-left text-[10px] font-bold text-gray-500 uppercase tracking-wider pb-3 px-2">
                    Time
                  </th>
                  <th className="text-left text-[10px] font-bold text-gray-500 uppercase tracking-wider pb-3 px-2">
                    Action
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {documents.slice(0, 5).map((doc) => (
                  <tr key={doc.id} className="hover:bg-purple-50/30 smooth-transition">
                    <td className="py-3.5 px-2">
                      <div className="flex items-center gap-3">
                        <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-purple-100 to-purple-50 border border-purple-100 flex items-center justify-center shrink-0">
                          <FileText className="w-[18px] h-[18px] text-[#6C5CE7]" />
                        </div>
                        <span className="text-sm text-gray-900 font-medium truncate max-w-xs">
                          {doc.fileName}
                        </span>
                      </div>
                    </td>
                    <td className="py-3.5 px-2">
                      <span
                        className={`inline-flex items-center px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider ${
                          doc.status === "completed"
                            ? "bg-emerald-50 text-emerald-700 border border-emerald-200"
                            : doc.status === "processing"
                            ? "bg-purple-50 text-purple-700 border border-purple-200"
                            : "bg-red-50 text-red-700 border border-red-200"
                        }`}
                      >
                        {doc.status}
                      </span>
                    </td>
                    <td className="py-3.5 px-2">
                      {doc.confidence ? (
                        <div className="flex items-center gap-2">
                          <div className="flex-1 max-w-[100px] h-1.5 bg-gray-100 rounded-full overflow-hidden">
                            <div
                              className="h-full bg-gradient-to-r from-emerald-400 to-emerald-500 rounded-full"
                              style={{ width: `${doc.confidence}%` }}
                            />
                          </div>
                          <span className="text-sm font-semibold text-gray-900">
                            {doc.confidence}%
                          </span>
                        </div>
                      ) : (
                        <span className="text-sm text-gray-400">—</span>
                      )}
                    </td>
                    <td className="py-3.5 px-2">
                      <span className="text-sm text-gray-700">{doc.language || "—"}</span>
                    </td>
                    <td className="py-3.5 px-2">
                      <span className="text-sm text-gray-700 font-mono">
                        {doc.processingTime || "—"}
                      </span>
                    </td>
                    <td className="py-3.5 px-2">
                      {doc.status === "completed" && (
                        <Link to={`/results/${doc.id}`}>
                          <button className="px-2.5 py-1.5 rounded-lg text-[#6C5CE7] hover:bg-purple-100/50 smooth-transition flex items-center gap-1.5 text-sm font-semibold">
                            <Eye className="w-3.5 h-3.5" />
                            View
                          </button>
                        </Link>
                      )}
                      {doc.status === "processing" && (
                        <button className="px-2.5 py-1.5 rounded-lg text-[#6C5CE7] hover:bg-purple-100/50 smooth-transition flex items-center gap-1.5 text-sm font-semibold">
                          <Eye className="w-3.5 h-3.5" />
                          View
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  iconBg: string;
  iconColor: string;
  value: string;
  label: string;
  change?: string;
  accent: string;
}

function StatCard({ icon, iconBg, iconColor, value, label, change, accent }: StatCardProps) {
  return (
    <div
      className="relative frost-card rounded-2xl p-5 smooth-transition hover:-translate-y-0.5 group overflow-hidden"
      style={{
        boxShadow: "0 1px 2px rgba(0,0,0,0.03), 0 8px 24px -8px rgba(108, 92, 231, 0.08)",
      }}
    >
      {/* subtle accent corner */}
      <div
        className="absolute top-0 right-0 w-24 h-24 rounded-full blur-2xl opacity-25 group-hover:opacity-40 smooth-transition pointer-events-none"
        style={{ background: accent }}
      />
      <div className="relative">
        <div className="flex items-start justify-between mb-4">
          <div
            className={`w-11 h-11 rounded-xl bg-gradient-to-br ${iconBg} border border-white/80 flex items-center justify-center shadow-sm`}
          >
            <span className={iconColor}>{icon}</span>
          </div>
          {change && (
            <div className="flex items-center gap-1 px-2 py-0.5 rounded-md bg-emerald-50 border border-emerald-100">
              <TrendingUp className="w-3 h-3 text-emerald-600" />
              <span className="text-[10px] font-bold text-emerald-700">{change}</span>
            </div>
          )}
        </div>
        <div className="text-2xl font-bold text-gray-900 mb-0.5 tracking-tight">{value}</div>
        <div className="text-xs text-gray-500 font-medium">{label}</div>
      </div>
    </div>
  );
}

interface ChartCardProps {
  title: string;
  subtitle: string;
  children: React.ReactNode;
  empty?: boolean;
  emptyText?: string;
}

function ChartCard({ title, subtitle, children, empty, emptyText = "No data yet" }: ChartCardProps) {
  return (
    <div
      className="frost-card rounded-2xl p-6"
      style={{
        boxShadow: "0 1px 2px rgba(0,0,0,0.03), 0 8px 24px -8px rgba(108, 92, 231, 0.08)",
      }}
    >
      <div className="flex items-center justify-between mb-1">
        <h3 className="font-bold text-gray-900 text-sm">{title}</h3>
      </div>
      <p className="text-xs text-gray-500 mb-5">{subtitle}</p>
      {empty ? (
        <div className="h-[200px] flex flex-col items-center justify-center gap-2">
          <div className="w-10 h-10 rounded-full bg-gray-50 border border-gray-100 flex items-center justify-center">
            <TrendingUp className="w-4 h-4 text-gray-300" />
          </div>
          <p className="text-xs text-gray-400">{emptyText}</p>
        </div>
      ) : (
        children
      )}
    </div>
  );
}
