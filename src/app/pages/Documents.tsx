import { FileText, Filter, Search, Upload as UploadIcon, Eye, Trash2 } from "lucide-react";
import { Link } from "react-router";
import { useDocumentStore } from "../store/documentStore";
import { deleteServerDocument } from "../lib/api";
import { useState } from "react";

export function Documents() {
  const documents = useDocumentStore((state) => state.documents);
  const removeDocument = useDocumentStore((state) => state.removeDocument);
  const clearDocuments = useDocumentStore((state) => state.clearDocuments);
  const [filterStatus, setFilterStatus] = useState<string>("all");
  const [search, setSearch] = useState<string>("");

  const filteredDocuments = documents.filter((doc) => {
    if (filterStatus !== "all" && doc.status !== filterStatus) return false;
    if (search.trim()) {
      const q = search.trim().toLowerCase();
      const haystack = [
        doc.fileName,
        doc.extractedFields?.dealerName ?? "",
        doc.extractedFields?.modelName ?? "",
      ]
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(q)) return false;
    }
    return true;
  });

  const handleDelete = async (id: string, fileName: string) => {
    if (!window.confirm(`Delete "${fileName}"? This cannot be undone.`)) return;
    // Optimistic local removal so the UI feels instant.
    removeDocument(id);
    try {
      await deleteServerDocument(id);
    } catch (err) {
      console.warn("Server-side delete failed:", err);
    }
  };

  const handleClearAll = async () => {
    if (
      !window.confirm(
        `Delete all ${documents.length} document(s)? This cannot be undone.`,
      )
    )
      return;
    const ids = documents.map((d) => d.id);
    clearDocuments();
    await Promise.allSettled(ids.map((id) => deleteServerDocument(id)));
  };

  return (
    <div className="p-8">
      {/* Search and Filters */}
      <div className="mb-6">
        <div className="flex items-center gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search by filename, dealer, model..."
              className="pl-10 pr-4 py-3 w-full rounded-lg glass-card border border-gray-200 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-[#6C5CE7]/20 focus:border-[#6C5CE7] smooth-transition"
            />
          </div>

          {/* Filter Button */}
          <button className="p-3 rounded-lg glass-card border border-gray-200 hover:bg-gray-50 smooth-transition">
            <Filter className="w-4 h-4 text-gray-600" />
          </button>

          {documents.length > 0 && (
            <button
              onClick={handleClearAll}
              title="Delete all documents"
              className="p-3 rounded-lg glass-card border border-gray-200 hover:bg-red-50 hover:border-red-200 hover:text-red-600 smooth-transition text-gray-600"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>

        {/* Filter Chips Row */}
        <div className="flex items-center justify-between mt-4">
          <p className="text-sm text-gray-600">
            Showing {filteredDocuments.length} of {documents.length} document{documents.length !== 1 ? 's' : ''}
          </p>
          
          <div className="flex items-center gap-2">
            <FilterChip 
              label="All" 
              active={filterStatus === "all"}
              onClick={() => setFilterStatus("all")}
            />
            <FilterChip 
              label="Completed" 
              active={filterStatus === "completed"}
              onClick={() => setFilterStatus("completed")}
            />
            <FilterChip 
              label="Processing" 
              active={filterStatus === "processing"}
              onClick={() => setFilterStatus("processing")}
            />
            <FilterChip 
              label="Failed" 
              active={filterStatus === "failed"}
              onClick={() => setFilterStatus("failed")}
            />
          </div>
        </div>
      </div>

      {/* Documents Table or Empty State */}
      {documents.length === 0 ? (
        <div className="glass-card rounded-xl p-16 text-center">
          <div className="w-20 h-20 rounded-2xl bg-purple-50 flex items-center justify-center mx-auto mb-4">
            <UploadIcon className="w-10 h-10 text-[#6C5CE7]" />
          </div>
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No documents yet</h3>
          <p className="text-gray-600 mb-6">Upload your first document to get started</p>
          <Link to="/upload">
            <button className="gradient-button text-white px-6 py-3 rounded-lg font-medium inline-flex items-center gap-2">
              <UploadIcon className="w-4 h-4" />
              Upload First Document
            </button>
          </Link>
        </div>
      ) : (
        <div className="glass-card rounded-xl p-6">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider pb-3 pl-4">Document</th>
                  <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider pb-3">Status</th>
                  <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider pb-3">Confidence</th>
                  <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider pb-3">Fields Extracted</th>
                  <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider pb-3">Language</th>
                  <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider pb-3">Date</th>
                  <th className="text-left text-xs font-medium text-gray-500 uppercase tracking-wider pb-3">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {filteredDocuments.map((doc) => (
                  <tr key={doc.id} className="hover:bg-gray-50/50 smooth-transition">
                    <td className="py-4 pl-4">
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-lg bg-purple-50 flex items-center justify-center">
                          <FileText className="w-5 h-5 text-[#6C5CE7]" />
                        </div>
                        <div>
                          <div className="text-sm text-gray-900 font-medium truncate max-w-xs">{doc.fileName}</div>
                          <div className="text-xs text-gray-500">{doc.fileSize}</div>
                        </div>
                      </div>
                    </td>
                    <td className="py-4">
                      <span className={`
                        inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold uppercase
                        ${doc.status === "completed" ? "bg-green-100 text-green-700" : ""}
                        ${doc.status === "processing" ? "bg-purple-100 text-purple-700" : ""}
                        ${doc.status === "failed" ? "bg-red-100 text-red-700" : ""}
                      `}>
                        {doc.status}
                      </span>
                    </td>
                    <td className="py-4">
                      {doc.confidence ? (
                        <div className="flex items-center gap-2">
                          <div className="flex-1 max-w-[120px] h-2 bg-gray-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-green-500 rounded-full smooth-transition"
                              style={{ width: `${doc.confidence}%` }}
                            ></div>
                          </div>
                          <span className="text-sm font-medium text-gray-900 min-w-[35px]">{doc.confidence}%</span>
                        </div>
                      ) : (
                        <span className="text-sm text-gray-400">—</span>
                      )}
                    </td>
                    <td className="py-4">
                      {doc.status === "completed" ? (
                        <div className="flex items-center gap-2">
                          <div className="flex items-center gap-0.5">
                            {[1, 2, 3, 4, 5, 6].map((i) => (
                              <div key={i} className="w-1.5 h-1.5 rounded-full bg-[#6C5CE7]"></div>
                            ))}
                          </div>
                          <span className="text-sm text-gray-900 font-medium">4/4</span>
                        </div>
                      ) : (
                        <span className="text-sm text-gray-400">0/4</span>
                      )}
                    </td>
                    <td className="py-4">
                      <span className="text-sm text-gray-600">{doc.language || "—"}</span>
                    </td>
                    <td className="py-4">
                      <span className="text-sm text-gray-600">
                        {new Date(doc.uploadedAt).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                      </span>
                    </td>
                    <td className="py-4">
                      <div className="flex items-center gap-2">
                        {doc.status === "completed" && (
                          <Link to={`/results/${doc.id}`}>
                            <button className="p-2 rounded-lg text-[#6C5CE7] hover:bg-purple-50 smooth-transition flex items-center gap-1 text-sm font-medium">
                              <Eye className="w-4 h-4" />
                              View
                            </button>
                          </Link>
                        )}
                        {doc.status === "processing" && (
                          <button className="p-2 rounded-lg text-[#6C5CE7] hover:bg-purple-50 smooth-transition flex items-center gap-1 text-sm font-medium">
                            <Eye className="w-4 h-4" />
                            View
                          </button>
                        )}
                        <button
                          onClick={() => handleDelete(doc.id, doc.fileName)}
                          title="Delete document"
                          className="p-2 rounded-lg text-gray-400 hover:bg-red-50 hover:text-red-600 smooth-transition"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
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

interface FilterChipProps {
  label: string;
  active: boolean;
  onClick: () => void;
}

function FilterChip({ label, active, onClick }: FilterChipProps) {
  return (
    <button
      onClick={onClick}
      className={`
        px-4 py-2 rounded-full text-sm font-medium smooth-transition
        ${active 
          ? "bg-[#6C5CE7] text-white shadow-md" 
          : "bg-white text-gray-600 border border-gray-200 hover:bg-gray-50"
        }
      `}
    >
      {label}
    </button>
  );
}