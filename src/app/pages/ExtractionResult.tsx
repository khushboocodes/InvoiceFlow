import { ChevronLeft, FileText, Clock, Languages, FileType, Sparkles, ExternalLink, Copy, CheckCircle2 } from "lucide-react";
import { Link, useParams } from "react-router";
import { useDocumentStore, type Document } from "../store/documentStore";
import { useEffect, useRef, useState } from "react";

export function ExtractionResult() {
  const { id } = useParams();
  const documents = useDocumentStore((state) => state.documents);
  const document = documents.find(d => d.id === id);
  const [copied, setCopied] = useState(false);

  if (!document) {
    return (
      <div className="p-8">
        <div className="glass-card rounded-xl p-16 text-center">
          <h3 className="text-xl font-semibold text-gray-900 mb-2">Document not found</h3>
          <Link to="/documents" className="text-[#6C5CE7] hover:text-[#4F46E5]">
            Go back to documents
          </Link>
        </div>
      </div>
    );
  }

  const mockJsonOutput = {
    "document_id": document.id,
    "filename": document.fileName,
    "document_type": document.documentType,
    "confidence_score": document.confidence,
    "processing_time_seconds": parseFloat(document.processingTime || "0"),
    "language": document.language,
    "extracted_fields": {
      "dealer_name": {
        "value": document.extractedFields?.dealerName,
        "confidence": 0.95,
        "bounding_box": null
      },
      "model_name": {
        "value": document.extractedFields?.modelName,
        "confidence": 0.98,
        "bounding_box": null
      },
      "horse_power": {
        "value": document.extractedFields?.horsePower,
        "confidence": 0.92,
        "bounding_box": null
      },
      "asset_cost": {
        "value": document.extractedFields?.assetCost,
        "confidence": 0.96,
        "bounding_box": null
      }
    },
    "detected_objects": {
      "signature": {
        "detected": document.extractedFields?.signatureDetected,
        "confidence": 0.91,
        "bounding_box": document.extractedFields?.signatureBbox ?? null
      },
      "stamp": {
        "detected": document.extractedFields?.stampDetected,
        "confidence": 0.85,
        "bounding_box": document.extractedFields?.stampBbox ?? null
      }
    },
    "metadata": {
      "processed_at": document.uploadedAt,
      "file_size": document.fileSize,
      "file_format": document.fileType
    }
  };

  const handleCopyJson = () => {
    navigator.clipboard.writeText(JSON.stringify(mockJsonOutput, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-8">
      {/* Back Button */}
      <Link to="/documents" className="inline-flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 mb-6 smooth-transition">
        <ChevronLeft className="w-4 h-4" />
        Documents
      </Link>

      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div className="flex-1">
          <h2 className="text-2xl font-semibold text-gray-900 mb-2">{document.fileName}</h2>
          <p className="text-sm text-gray-600">
            Processed 6 hours ago • {document.documentType}
          </p>
        </div>

        <div className="flex items-center gap-3">
          <span className="inline-flex items-center px-3 py-1.5 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">
            COMPLETED
          </span>
          <button className="px-4 py-2 rounded-lg glass-card border border-gray-200 text-sm font-medium text-gray-700 hover:bg-gray-50 smooth-transition flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-[#6C5CE7]" />
            Explain AI
          </button>
          <button className="px-4 py-2 rounded-lg glass-card border border-gray-200 text-sm font-medium text-gray-700 hover:bg-gray-50 smooth-transition flex items-center gap-2">
            <ExternalLink className="w-4 h-4" />
            View File
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-4 gap-6 mb-8">
        <StatCard
          icon={<Sparkles className="w-4 h-4 text-yellow-600" />}
          iconBg="bg-yellow-50"
          label="Confidence"
          value={`${document.confidence}%`}
        />
        <StatCard
          icon={<Clock className="w-4 h-4 text-purple-600" />}
          iconBg="bg-purple-50"
          label="Processing Time"
          value={document.processingTime || "—"}
        />
        <StatCard
          icon={<Languages className="w-4 h-4 text-blue-600" />}
          iconBg="bg-blue-50"
          label="Language"
          value={document.language || "—"}
        />
        <StatCard
          icon={<FileType className="w-4 h-4 text-green-600" />}
          iconBg="bg-green-50"
          label="Doc Type"
          value={document.documentType || "—"}
        />
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-2 gap-8 mb-8">
        {/* Left - Document Preview */}
        <div className="glass-card rounded-xl p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-gray-900">Document Preview</h3>
            <div className="flex items-center gap-2 text-xs text-green-600">
              <div className="w-2 h-2 rounded-full bg-green-500"></div>
              <span className="font-medium">{document.confidence ?? 0}%</span>
            </div>
          </div>

          <DocumentPreview document={document} />
        </div>

        {/* Right - Extracted Fields */}
        <div className="glass-card rounded-xl p-6">
          <h3 className="font-semibold text-gray-900 mb-1">Extracted Fields</h3>
          <p className="text-sm text-gray-500 mb-6">AI extracted structured data from your document</p>

          <div className="space-y-4">
            <FieldItem 
              label="DEALER NAME" 
              value={document.extractedFields?.dealerName || "—"} 
            />
            <FieldItem 
              label="MODEL NAME" 
              value={document.extractedFields?.modelName || "—"} 
            />
            <FieldItem 
              label="HORSE POWER" 
              value={document.extractedFields?.horsePower || "—"} 
            />
            <FieldItem 
              label="ASSET COST" 
              value={document.extractedFields?.assetCost || "—"} 
            />
            
            <div className="pt-4 border-t border-gray-200">
              <DetectionItem
                label="DEALER SIGNATURE"
                detected={document.extractedFields?.signatureDetected}
                coords={
                  document.extractedFields?.signatureBbox
                    ? `[${document.extractedFields.signatureBbox.join(", ")}]`
                    : "—"
                }
              />
              <DetectionItem
                label="DEALER STAMP"
                detected={document.extractedFields?.stampDetected}
                coords={
                  document.extractedFields?.stampBbox
                    ? `[${document.extractedFields.stampBbox.join(", ")}]`
                    : "—"
                }
              />
            </div>
          </div>
        </div>
      </div>

      {/* JSON Output */}
      <div className="glass-card rounded-xl p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-gray-900">JSON Output</h3>
          <button 
            onClick={handleCopyJson}
            className="px-3 py-1.5 rounded-lg text-sm font-medium text-[#6C5CE7] hover:bg-purple-50 smooth-transition flex items-center gap-2"
          >
            {copied ? (
              <>
                <CheckCircle2 className="w-4 h-4" />
                Copied
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                Copy
              </>
            )}
          </button>
        </div>

        <div className="bg-gray-900 rounded-lg p-4 overflow-x-auto custom-scrollbar">
          <pre className="text-xs text-gray-100 font-mono">
            {JSON.stringify(mockJsonOutput, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}

interface StatCardProps {
  icon: React.ReactNode;
  iconBg: string;
  label: string;
  value: string;
}

function StatCard({ icon, iconBg, label, value }: StatCardProps) {
  return (
    <div className="glass-card rounded-xl p-4">
      <div className="flex items-center gap-3 mb-2">
        <div className={`w-8 h-8 rounded-lg ${iconBg} flex items-center justify-center`}>
          {icon}
        </div>
        <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">{label}</span>
      </div>
      <div className="text-xl font-semibold text-gray-900">{value}</div>
    </div>
  );
}

interface FieldItemProps {
  label: string;
  value: string;
}

function FieldItem({ label, value }: FieldItemProps) {
  return (
    <div className="border-l-4 border-[#6C5CE7] pl-4">
      <div className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className="text-base font-semibold text-gray-900">{value}</div>
    </div>
  );
}

interface DetectionItemProps {
  label: string;
  detected?: boolean;
  coords: string;
}

function DetectionItem({ label, detected, coords }: DetectionItemProps) {
  return (
    <div className="py-3 border-b border-gray-100 last:border-0">
      <div className="flex items-center justify-between mb-2">
        <div className="text-xs font-medium text-gray-500 uppercase tracking-wider">
          {label}
        </div>
        {detected && (
          <div className="flex items-center gap-1 text-green-600">
            <CheckCircle2 className="w-4 h-4" />
            <span className="text-xs font-medium">Detected</span>
          </div>
        )}
      </div>
      <div className="text-xs text-gray-500 font-mono">{coords}</div>
    </div>
  );
}


/**
 * Renders the actual uploaded document (image or PDF) with the YOLO-detected
 * signature/stamp bounding boxes overlaid in the right pixel coordinates.
 *
 * The bboxes from the backend are in pixel coordinates of the OCR-rendered
 * page (typically 300 DPI). We use the loaded image's natural dimensions to
 * compute the percent-based offsets so the boxes stay aligned regardless of
 * the preview container size.
 */
function DocumentPreview({ document: doc }: { document: Document }) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const [imgError, setImgError] = useState(false);

  const isPdf =
    doc.mimeType === "application/pdf" ||
    doc.fileName.toLowerCase().endsWith(".pdf");
  const isImage =
    !isPdf &&
    (doc.mimeType?.startsWith("image/") ||
      /\.(png|jpe?g|webp|gif|bmp)$/i.test(doc.fileName));

  // Capture natural pixel dimensions once the image loads, so the bbox
  // overlay can convert from absolute px → % of container.
  useEffect(() => {
    if (!imgRef.current) return;
    if (imgRef.current.complete && imgRef.current.naturalWidth > 0) {
      setNaturalSize({
        w: imgRef.current.naturalWidth,
        h: imgRef.current.naturalHeight,
      });
    }
  }, [doc.previewUrl]);

  const sigBbox = doc.extractedFields?.signatureBbox;
  const stampBbox = doc.extractedFields?.stampBbox;

  // Convert a pixel-space bbox to percent-of-container coordinates so the
  // overlay survives the responsive image scaling.
  const toPercent = (
    bbox: [number, number, number, number] | null | undefined,
  ): { left: string; top: string; width: string; height: string } | null => {
    if (!bbox || !naturalSize) return null;
    const [x1, y1, x2, y2] = bbox;
    return {
      left: `${(x1 / naturalSize.w) * 100}%`,
      top: `${(y1 / naturalSize.h) * 100}%`,
      width: `${((x2 - x1) / naturalSize.w) * 100}%`,
      height: `${((y2 - y1) / naturalSize.h) * 100}%`,
    };
  };

  const sigStyle = toPercent(sigBbox);
  const stampStyle = toPercent(stampBbox);

  // No preview URL (e.g. the user reloaded the page after uploading): fall
  // back to the placeholder card.
  if (!doc.previewUrl || imgError) {
    return (
      <div className="relative bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-8 aspect-[3/4] border border-gray-200">
        <div className="absolute inset-4 bg-white rounded shadow-sm border border-gray-200 flex items-center justify-center">
          <div className="text-center">
            <FileText className="w-16 h-16 text-gray-200 mx-auto mb-3" />
            <p className="text-xs text-gray-400 font-medium">Document Preview</p>
            <p className="text-xs text-gray-300">{doc.fileName}</p>
            <p className="text-xs text-gray-300 mt-2">
              Re-upload to see the original page.
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (isPdf) {
    return (
      <div className="relative bg-gray-50 rounded-lg overflow-hidden border border-gray-200">
        <embed
          src={doc.previewUrl}
          type="application/pdf"
          className="w-full"
          style={{ height: "640px" }}
        />
      </div>
    );
  }

  if (isImage) {
    return (
      <div className="relative bg-white rounded-lg overflow-hidden border border-gray-200">
        <img
          ref={imgRef}
          src={doc.previewUrl}
          alt={doc.fileName}
          className="w-full h-auto block"
          onLoad={(e) => {
            const el = e.currentTarget;
            setNaturalSize({ w: el.naturalWidth, h: el.naturalHeight });
          }}
          onError={() => setImgError(true)}
        />

        {sigStyle && (
          <div
            className="absolute border-2 border-green-500 rounded bg-green-500/10 z-10 pointer-events-none"
            style={sigStyle}
          >
            <div className="absolute -top-6 left-0 bg-green-500 text-white text-[10px] px-2 py-0.5 rounded font-medium shadow-md whitespace-nowrap">
              Signature
            </div>
          </div>
        )}

        {stampStyle && (
          <div
            className="absolute border-2 border-blue-500 rounded bg-blue-500/10 z-10 pointer-events-none"
            style={stampStyle}
          >
            <div className="absolute -top-6 left-0 bg-blue-500 text-white text-[10px] px-2 py-0.5 rounded font-medium shadow-md whitespace-nowrap">
              Stamp
            </div>
          </div>
        )}
      </div>
    );
  }

  // Unknown type: at least let the user open it in a new tab.
  return (
    <div className="relative bg-gradient-to-br from-gray-50 to-gray-100 rounded-lg p-8 aspect-[3/4] border border-gray-200">
      <div className="absolute inset-4 bg-white rounded shadow-sm border border-gray-200 flex items-center justify-center">
        <div className="text-center">
          <FileText className="w-16 h-16 text-gray-200 mx-auto mb-3" />
          <p className="text-xs text-gray-400 font-medium">{doc.fileName}</p>
          <a
            href={doc.previewUrl}
            target="_blank"
            rel="noreferrer"
            className="mt-3 inline-block text-xs text-[#6C5CE7] hover:underline"
          >
            Open file
          </a>
        </div>
      </div>
    </div>
  );
}
