import { FileText, CheckCircle2, TrendingUp, Languages } from "lucide-react";
import { useDocumentStore } from "../store/documentStore";
import { LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

export function Analytics() {
  const documents = useDocumentStore((state) => state.documents);
  
  const completedDocs = documents.filter(d => d.status === "completed").length;
  const totalDocs = documents.length;

  // Mock data for analytics
  const volumeData = [
    { date: "Mar 28", value: 0 },
    { date: "Mar 29", value: 0 },
    { date: "Mar 30", value: 0 },
    { date: "Mar 31", value: 0 },
    { date: "Apr 1", value: 0 },
    { date: "Apr 2", value: 0 },
    { date: "Apr 3", value: 0 },
    { date: "Apr 4", value: 0 },
    { date: "Apr 5", value: 0 },
    { date: "Apr 6", value: 0 },
    { date: "Apr 7", value: 0 },
    { date: "Apr 8", value: 0 },
    { date: "Apr 9", value: completedDocs > 0 ? 1 : 0 },
    { date: "Apr 10", value: completedDocs > 0 ? 1 : 0 },
  ];

  const fieldSuccessData = [
    { field: "Dealer Name", success: completedDocs > 0 ? 100 : 0 },
    { field: "Model Name", success: completedDocs > 0 ? 100 : 0 },
    { field: "Horse Power", success: completedDocs > 0 ? 100 : 0 },
    { field: "Asset Cost", success: completedDocs > 0 ? 100 : 0 },
    { field: "Signature", success: completedDocs > 0 ? 100 : 0 },
    { field: "Stamp", success: completedDocs > 0 ? 100 : 0 },
  ];

  const languageData = [
    { name: "English", value: completedDocs > 0 ? 100 : 0, color: "#6C5CE7" },
  ];

  const avgConfidence = completedDocs > 0
    ? documents.filter(d => d.confidence).reduce((acc, d) => acc + (d.confidence || 0), 0) / completedDocs || 0
    : 0;

  const avgProcessing = completedDocs > 0
    ? documents.filter(d => d.processingTime).reduce((acc, d) => acc + parseFloat(d.processingTime || "0"), 0) / completedDocs || 0
    : 0;

  if (totalDocs === 0) {
    return (
      <div className="p-8">
        <div className="mb-8">
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">Analytics</h2>
          <p className="text-gray-600">Performance and insight data across all processed documents</p>
        </div>
        <div className="glass-card rounded-xl p-16 text-center">
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No data to analyze yet</h3>
          <p className="text-gray-600">Process some documents to see analytics here.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Analytics</h2>
        <p className="text-gray-600">Performance and insight data across all processed documents</p>
      </div>

      {/* Summary Stats Bar */}
      <div className="glass-card rounded-xl p-6 mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-[#6C5CE7]"></div>
            <span className="text-sm text-gray-600">Total</span>
            <span className="text-base font-semibold text-gray-900 ml-2">{totalDocs}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-green-500"></div>
            <span className="text-sm text-gray-600">Completed</span>
            <span className="text-base font-semibold text-gray-900 ml-2">{completedDocs}</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-orange-500"></div>
            <span className="text-sm text-gray-600">Avg Confidence</span>
            <span className="text-base font-semibold text-gray-900 ml-2">{avgConfidence.toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-purple-500"></div>
            <span className="text-sm text-gray-600">Avg Time</span>
            <span className="text-base font-semibold text-gray-900 ml-2">{avgProcessing.toFixed(1)}s</span>
          </div>
        </div>
      </div>

      {/* Volume Chart */}
      <div className="glass-card rounded-xl p-6 mb-8">
        <h3 className="font-semibold text-gray-900 mb-1">Document Volume (14 days)</h3>
        <p className="text-sm text-gray-500 mb-6">Daily processed documents and completion rate</p>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={volumeData}>
            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
            <XAxis 
              dataKey="date" 
              tick={{ fontSize: 11, fill: '#6B7280' }}
              axisLine={false}
              tickLine={false}
            />
            <YAxis 
              tick={{ fontSize: 11, fill: '#6B7280' }}
              axisLine={false}
              tickLine={false}
            />
            <Tooltip />
            <Line 
              type="monotone" 
              dataKey="value" 
              stroke="#6C5CE7" 
              strokeWidth={3} 
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-2 gap-6">
        {/* Field Success Chart */}
        <div className="glass-card rounded-xl p-6">
          <h3 className="font-semibold text-gray-900 mb-1">Field Extraction Success</h3>
          <p className="text-sm text-gray-500 mb-6">Percentage of documents with each field detected</p>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={fieldSuccessData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" horizontal={false} />
              <XAxis 
                type="number" 
                tick={{ fontSize: 11, fill: '#6B7280' }}
                axisLine={false}
                tickLine={false}
                domain={[0, 100]}
              />
              <YAxis 
                dataKey="field" 
                type="category" 
                tick={{ fontSize: 11, fill: '#6B7280' }} 
                width={100}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip />
              <Bar 
                dataKey="success" 
                fill="#6C5CE7" 
                radius={[0, 4, 4, 0]}
                barSize={24}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Language Distribution Chart */}
        <div className="glass-card rounded-xl p-6">
          <h3 className="font-semibold text-gray-900 mb-1">Language Distribution</h3>
          <p className="text-sm text-gray-500 mb-6">Detected document languages</p>
          
          {completedDocs === 0 ? (
            <div className="h-[300px] flex items-center justify-center">
              <p className="text-sm text-gray-400">No language data yet</p>
            </div>
          ) : (
            <div className="flex items-center justify-center h-[300px]">
              <ResponsiveContainer width="60%" height="100%">
                <PieChart>
                  <Pie
                    data={languageData}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={100}
                    dataKey="value"
                    startAngle={90}
                    endAngle={450}
                  >
                    {languageData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
              <div className="ml-4">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded-full bg-[#6C5CE7]"></div>
                  <span className="text-sm text-gray-900 font-medium">English ({completedDocs})</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}