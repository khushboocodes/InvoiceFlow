import { useCallback, useEffect, useState } from "react";
import {
  Upload as UploadIcon,
  FileText,
  X,
  ChevronLeft,
  Loader2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { Link, useNavigate } from "react-router";
import { useDocumentStore } from "../store/documentStore";
import { checkBridgeHealth, uploadAndExtract } from "../lib/api";

/**
 * One row in the upload queue. We don't drive the global document store
 * for queued/in-flight items because they don't have a server-assigned id
 * yet — we mirror them into the store only AFTER the bridge persists them.
 */
interface QueueItem {
  /** Local UUID, used only for React keys and progress wiring. */
  localId: string;
  file: File;
  /** "queued" → "processing" → "completed" or "failed". */
  status: "queued" | "processing" | "completed" | "failed";
  /** Server-assigned id once the upload succeeds. */
  serverId?: string;
  /** Error message when status === "failed". */
  error?: string;
}

/** Read a File into a base64 data URL — used for the immediate local preview. */
function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      if (typeof reader.result === "string") resolve(reader.result);
      else reject(new Error("FileReader produced a non-string result"));
    };
    reader.onerror = () => reject(reader.error ?? new Error("FileReader error"));
    reader.readAsDataURL(file);
  });
}

const MAX_PREVIEW_BYTES = 4 * 1024 * 1024;
const ALLOWED_EXTENSIONS = [".pdf", ".png", ".jpg", ".jpeg"];

function isSupported(file: File): boolean {
  const lower = file.name.toLowerCase();
  return ALLOWED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export function Upload() {
  const navigate = useNavigate();
  const addDocument = useDocumentStore((s) => s.addDocument);
  const updateDocument = useDocumentStore((s) => s.updateDocument);
  const removeDocument = useDocumentStore((s) => s.removeDocument);

  const [queue, setQueue] = useState<QueueItem[]>([]);
  const [bridgeAvailable, setBridgeAvailable] = useState<boolean | null>(null);
  const [busy, setBusy] = useState(false);
  // Index of the pipeline step currently animating in the right-hand panel.
  // Cycles 0 → pipelineSteps.length-1 while a document is in flight, and
  // resets to 0 between documents.
  const [activeStep, setActiveStep] = useState(0);

  // Probe the local bridge once on mount.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      const health = await checkBridgeHealth();
      if (!cancelled) setBridgeAvailable(health !== null);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const pipelineSteps = [
    { number: 1, title: "Document Ingestion", description: "Converts PDF/image to analyzable format" },
    { number: 2, title: "OCR Extraction", description: "Reads embedded text and layout structure" },
    { number: 3, title: "Field Detection", description: "Regex + AI identifies key entities" },
    { number: 4, title: "Vision Analysis", description: "Detects signatures and stamps" },
    { number: 5, title: "Structured Output", description: "Generates JSON with confidence scores" },
  ];

  const extractedFields = [
    "Dealer Name",
    "Model Name",
    "Horse Power",
    "Asset Cost",
    "Dealer Signature",
    "Dealer Stamp",
  ];

  // ---------------------------------------------------------------- queue I/O

  const addFiles = (files: File[]) => {
    const accepted: QueueItem[] = [];
    for (const f of files) {
      if (!isSupported(f)) {
        console.warn("Skipping unsupported file:", f.name);
        continue;
      }
      accepted.push({
        localId: `q_${Math.random().toString(36).slice(2, 10)}`,
        file: f,
        status: "queued",
      });
    }
    if (accepted.length) {
      setQueue((q) => [...q, ...accepted]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : [];
    addFiles(files);
    // Reset the input so the same filename can be selected again.
    e.target.value = "";
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const files = e.dataTransfer.files ? Array.from(e.dataTransfer.files) : [];
    addFiles(files);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
  }, []);

  const removeFromQueue = (localId: string) => {
    setQueue((q) => q.filter((item) => item.localId !== localId));
  };

  const updateQueue = (
    localId: string,
    updates: Partial<QueueItem>,
  ) => {
    setQueue((q) =>
      q.map((item) => (item.localId === localId ? { ...item, ...updates } : item)),
    );
  };

  const clearAllQueued = () => {
    setQueue((q) => q.filter((item) => item.status === "processing"));
  };

  // ---------------------------------------------------------------- runner

  const processOne = async (item: QueueItem): Promise<string | null> => {
    updateQueue(item.localId, { status: "processing" });
    // Restart the pipeline-step animation for each document so the
    // right-hand panel walks through the stages while the bridge runs.
    setActiveStep(0);
    const stepInterval = setInterval(() => {
      setActiveStep((s) => Math.min(s + 1, pipelineSteps.length - 1));
    }, 1500);

    // Local-only preview while we wait for the server response so the
    // dashboard placeholder card has something to show.
    let localPreview: string | undefined;
    if (item.file.size <= MAX_PREVIEW_BYTES) {
      try {
        localPreview = await fileToDataUrl(item.file);
      } catch (err) {
        console.warn("preview read failed:", err);
      }
    }

    // Insert a placeholder into the global store so the Documents/Dashboard
    // pages reflect "currently processing" right away.
    addDocument({
      id: item.localId,
      fileName: item.file.name,
      fileSize: `${(item.file.size / 1024).toFixed(2)} KB`,
      fileType: item.file.name.split(".").pop()?.toLowerCase() ?? "unknown",
      previewUrl: localPreview,
      mimeType: item.file.type || "application/octet-stream",
      status: "processing",
      uploadedAt: new Date().toISOString(),
    });

    if (!bridgeAvailable) {
      // No backend → simulated extraction so the UX still demos.
      await new Promise((r) => setTimeout(r, 1500));
      clearInterval(stepInterval);
      setActiveStep(pipelineSteps.length - 1);
      updateDocument(item.localId, {
        status: "completed",
        confidence: 95,
        processingTime: "2.1s",
        language: "English",
        documentType: "Tractor Quotation",
        extractedFields: {
          dealerName: "—",
          modelName: "—",
          horsePower: "—",
          assetCost: "—",
          signatureDetected: false,
          stampDetected: false,
        },
      });
      updateQueue(item.localId, { status: "completed", serverId: item.localId });
      return item.localId;
    }

    try {
      const serverDoc = await uploadAndExtract(item.file);
      clearInterval(stepInterval);
      setActiveStep(pipelineSteps.length - 1);
      // Replace the temp placeholder with the canonical server doc.
      removeDocument(item.localId);
      addDocument({
        ...serverDoc,
        previewUrl: serverDoc.previewUrl ?? localPreview,
      } as never);
      updateQueue(item.localId, { status: "completed", serverId: serverDoc.id });
      return serverDoc.id;
    } catch (err) {
      clearInterval(stepInterval);
      console.error("extraction failed for", item.file.name, err);
      updateDocument(item.localId, {
        status: "failed",
      });
      updateQueue(item.localId, {
        status: "failed",
        error: err instanceof Error ? err.message : String(err),
      });
      return null;
    }
  };

  const handleProcessAll = async () => {
    if (busy) return;
    const pending = queue.filter((q) => q.status === "queued");
    if (pending.length === 0) return;
    setBusy(true);
    let lastSuccessId: string | null = null;
    // Process sequentially — the backend pipeline holds GPU + SLM and can't
    // safely run two extractions at once.
    for (const item of pending) {
      const id = await processOne(item);
      if (id) lastSuccessId = id;
    }
    setBusy(false);
    if (lastSuccessId && pending.length === 1) {
      // Single-file flow: jump straight to the result page like before.
      setTimeout(() => navigate(`/results/${lastSuccessId}`), 400);
    }
  };

  // ---------------------------------------------------------------- render

  const queuedCount = queue.filter((q) => q.status === "queued").length;
  const processingItem = queue.find((q) => q.status === "processing") ?? null;
  const completedCount = queue.filter((q) => q.status === "completed").length;
  const failedCount = queue.filter((q) => q.status === "failed").length;
  const isProcessing = busy || processingItem !== null;
  const showEmptyState = queue.length === 0;

  return (
    <div className="p-8">
      {/* Back Button */}
      <Link
        to="/dashboard"
        className="inline-flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 mb-6 smooth-transition"
      >
        <ChevronLeft className="w-4 h-4" />
        Back to Dashboard
      </Link>

      {/* Header */}
      <div className="mb-8">
        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Upload Documents</h2>
        <p className="text-gray-600">
          Drop one or more invoices (PDF or image) — they'll process one after another.
        </p>
        {bridgeAvailable !== null && (
          <div className="mt-3 inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border bg-white/60 backdrop-blur-sm">
            <div
              className={`w-1.5 h-1.5 rounded-full ${
                bridgeAvailable ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span className="text-gray-700">
              {bridgeAvailable
                ? "Connected to local pipeline · Real extraction"
                : "Pipeline offline · Using simulated demo"}
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-3 gap-8">
        {/* Left: drop zone + queue */}
        <div className="col-span-2 space-y-6">
          {/* Drop zone — always visible so users can append more files mid-batch */}
          <div
            className={`glass-card rounded-xl p-10 border-2 border-dashed text-center smooth-transition ${
              isProcessing
                ? "border-purple-300 bg-purple-50/30"
                : "border-gray-300 hover:border-[#6C5CE7] hover:bg-purple-50/30"
            }`}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <div
              className={`w-14 h-14 rounded-2xl flex items-center justify-center mx-auto mb-3 ${
                isProcessing ? "bg-purple-50 processing-pulse" : "bg-gray-100"
              }`}
            >
              {isProcessing ? (
                <Loader2 className="w-7 h-7 text-[#6C5CE7] animate-spin" />
              ) : (
                <UploadIcon className="w-7 h-7 text-gray-400" />
              )}
            </div>
            <h3 className="text-base font-semibold text-gray-900 mb-1">
              {isProcessing ? "Processing in progress" : "Drop your documents here"}
            </h3>
            <p className="text-sm text-gray-500 mb-3">
              {isProcessing
                ? "You can keep adding more — they'll join the queue."
                : "Drop one or many · PDF, PNG, JPG"}
            </p>

            <input
              type="file"
              id="file-upload"
              className="hidden"
              accept=".pdf,.png,.jpg,.jpeg"
              multiple
              onChange={handleFileSelect}
            />
            <label htmlFor="file-upload">
              <span className="inline-block cursor-pointer">
                <div className="gradient-button text-white px-5 py-2.5 rounded-lg font-medium inline-flex items-center gap-2 text-sm">
                  <UploadIcon className="w-4 h-4" />
                  Browse Files
                </div>
              </span>
            </label>
          </div>

          {/* Queue summary + action */}
          {queue.length > 0 && (
            <div className="glass-card rounded-xl p-5">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <h3 className="font-semibold text-gray-900">Upload Queue</h3>
                  <span className="text-xs font-semibold text-gray-500 bg-gray-100 px-2 py-0.5 rounded-full">
                    {queue.length}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs text-gray-600">
                  {processingItem && (
                    <span className="text-[#6C5CE7] font-medium flex items-center gap-1">
                      <Loader2 className="w-3 h-3 animate-spin" />
                      processing
                    </span>
                  )}
                  {completedCount > 0 && (
                    <span className="text-emerald-600 font-medium">
                      {completedCount} done
                    </span>
                  )}
                  {failedCount > 0 && (
                    <span className="text-red-600 font-medium">{failedCount} failed</span>
                  )}
                  {queuedCount > 0 && !isProcessing && (
                    <button
                      onClick={clearAllQueued}
                      className="text-gray-500 hover:text-red-600 smooth-transition"
                    >
                      Clear queued
                    </button>
                  )}
                </div>
              </div>

              <div className="space-y-2">
                {queue.map((item) => (
                  <QueueRow
                    key={item.localId}
                    item={item}
                    onRemove={() => removeFromQueue(item.localId)}
                    onView={
                      item.status === "completed" && item.serverId
                        ? () => navigate(`/results/${item.serverId}`)
                        : undefined
                    }
                  />
                ))}
              </div>

              {queuedCount > 0 && (
                <button
                  onClick={handleProcessAll}
                  disabled={busy}
                  className="mt-5 w-full gradient-button text-white px-6 py-3.5 rounded-lg font-medium flex items-center justify-center gap-2 text-base shadow-lg disabled:opacity-60 disabled:cursor-not-allowed"
                >
                  {busy ? (
                    <>
                      <Loader2 className="w-5 h-5 animate-spin" />
                      Processing {processingItem?.file.name ?? "..."}
                    </>
                  ) : (
                    <>
                      <UploadIcon className="w-5 h-5" />
                      Process {queuedCount} Document{queuedCount === 1 ? "" : "s"}
                    </>
                  )}
                </button>
              )}
            </div>
          )}

          {/* Extracted Fields preview card */}
          {showEmptyState && (
            <div className="glass-card rounded-xl p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-5 h-5 rounded-full bg-purple-100 flex items-center justify-center">
                  <span className="text-xs font-medium text-[#6C5CE7]">i</span>
                </div>
                <h3 className="font-semibold text-gray-900">Extracted Fields</h3>
              </div>
              <div className="grid grid-cols-2 gap-4">
                {extractedFields.map((field, index) => (
                  <div key={index} className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-[#6C5CE7]"></div>
                    <span className="text-sm text-gray-600">{field}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right: AI pipeline panel */}
        <div className="col-span-1">
          <div className="glass-card rounded-xl p-6 sticky top-8">
            <h3 className="font-semibold text-gray-900 mb-2">AI Pipeline</h3>
            <p className="text-sm text-gray-600 mb-6">How InvoiceFlow processes your document</p>

            <div className="space-y-4">
              {pipelineSteps.map((step, index) => (
                <div
                  key={step.number}
                  className={`
                    p-4 rounded-lg smooth-transition
                    ${isProcessing && index === activeStep
                      ? "bg-purple-50 border-2 border-[#6C5CE7]"
                      : isProcessing && index < activeStep
                      ? "bg-green-50 border border-green-200"
                      : "bg-gray-50 border border-gray-200"
                    }
                  `}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className={`
                        w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0
                        ${isProcessing && index === activeStep
                          ? "bg-[#6C5CE7] text-white"
                          : isProcessing && index < activeStep
                          ? "bg-green-500 text-white"
                          : "bg-white text-gray-600"
                        }
                      `}
                    >
                      <span className="text-sm font-semibold">{step.number}</span>
                    </div>
                    <div className="flex-1">
                      <h4 className="font-medium text-gray-900 text-sm mb-1">
                        {step.title}
                      </h4>
                      <p className="text-xs text-gray-600">{step.description}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface QueueRowProps {
  item: QueueItem;
  onRemove: () => void;
  onView?: () => void;
}

function QueueRow({ item, onRemove, onView }: QueueRowProps) {
  const sizeKb = `${(item.file.size / 1024).toFixed(1)} KB`;
  const stateBadge = (() => {
    switch (item.status) {
      case "queued":
        return (
          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-gray-100 text-gray-600 border border-gray-200">
            Queued
          </span>
        );
      case "processing":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-purple-100 text-[#6C5CE7] border border-purple-200">
            <Loader2 className="w-2.5 h-2.5 animate-spin" />
            Processing
          </span>
        );
      case "completed":
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-emerald-50 text-emerald-700 border border-emerald-200">
            <CheckCircle2 className="w-2.5 h-2.5" />
            Done
          </span>
        );
      case "failed":
        return (
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold uppercase bg-red-50 text-red-700 border border-red-200"
            title={item.error}
          >
            <AlertCircle className="w-2.5 h-2.5" />
            Failed
          </span>
        );
    }
  })();

  return (
    <div
      className={`flex items-center gap-3 p-3 rounded-lg border smooth-transition ${
        item.status === "processing"
          ? "border-purple-200 bg-purple-50/50"
          : "border-gray-200 bg-white"
      }`}
    >
      <div className="w-9 h-9 rounded-lg bg-purple-50 flex items-center justify-center flex-shrink-0">
        <FileText className="w-5 h-5 text-[#6C5CE7]" />
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-0.5">
          <span className="text-sm font-medium text-gray-900 truncate">
            {item.file.name}
          </span>
          {stateBadge}
        </div>
        <div className="text-xs text-gray-500">{sizeKb}</div>
      </div>
      <div className="flex items-center gap-1">
        {onView && (
          <button
            onClick={onView}
            className="px-2.5 py-1 rounded-lg text-[#6C5CE7] hover:bg-purple-100/50 smooth-transition text-xs font-semibold"
          >
            View
          </button>
        )}
        {item.status !== "processing" && (
          <button
            onClick={onRemove}
            title="Remove from queue"
            className="p-1.5 rounded-lg text-gray-400 hover:bg-gray-100 hover:text-gray-700 smooth-transition"
          >
            <X className="w-4 h-4" />
          </button>
        )}
      </div>
    </div>
  );
}
