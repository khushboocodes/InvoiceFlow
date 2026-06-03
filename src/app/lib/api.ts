/**
 * Tiny client for the local FastAPI bridge in `demo/server.py`.
 *
 * Talks to `http://127.0.0.1:8000` — the bridge runs on the same machine as
 * the React dev server. When the bridge is unreachable, the upload flow
 * falls back to its built-in simulated pipeline so the UI stays demoable
 * even when the backend isn't running.
 */

// Auto-detect the bridge host so the same app works on a different machine
// or browser on the LAN. Default: same hostname as the page, port 8000.
function inferBridgeBase(): string {
  if (typeof window === "undefined") return "http://127.0.0.1:8000";
  // Allow an env override or a runtime override stashed on window.
  const override = (window as any).__BRIDGE_URL__;
  if (typeof override === "string" && override.length > 0) return override;
  const host = window.location.hostname || "127.0.0.1";
  return `${window.location.protocol}//${host}:8000`;
}

export const BRIDGE_BASE = inferBridgeBase();

export interface ExtractionFields {
  dealer_name: { value: string | null; confidence: number };
  model_name: { value: string | null; confidence: number };
  horse_power: { value: number | null; confidence: number };
  asset_cost: { value: number | null; confidence: number };
  signature: {
    present: boolean;
    bbox: [number, number, number, number] | null;
    confidence: number;
  };
  stamp: {
    present: boolean;
    bbox: [number, number, number, number] | null;
    confidence: number;
  };
}

export interface ExtractionResult {
  doc_id: string;
  fields: ExtractionFields;
  confidence: number;
  processing_time_sec: number;
  cost_estimate_usd: number;
  error: string | null;
}

export interface HealthResponse {
  status: string;
  device: string;
}

export async function checkBridgeHealth(): Promise<HealthResponse | null> {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const resp = await fetch(`${BRIDGE_BASE}/api/health`, { signal: controller.signal });
    clearTimeout(timeout);
    if (!resp.ok) return null;
    return (await resp.json()) as HealthResponse;
  } catch {
    return null;
  }
}

export async function extractDocument(file: File): Promise<ExtractionResult> {
  const form = new FormData();
  form.append("file", file);

  const resp = await fetch(`${BRIDGE_BASE}/api/extract`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`Bridge returned ${resp.status}: ${detail}`);
  }
  return (await resp.json()) as ExtractionResult;
}

/** Format an Indian currency amount (e.g. 525000 → "₹ 5,25,000"). */
export function formatINR(amount: number | null | undefined): string {
  if (amount === null || amount === undefined) return "—";
  // Indian numbering: last 3 digits grouped, then groups of 2.
  const s = String(Math.abs(amount));
  const last3 = s.slice(-3);
  const rest = s.slice(0, -3);
  const grouped =
    rest.length > 0 ? rest.replace(/\B(?=(\d{2})+(?!\d))/g, ",") + "," + last3 : last3;
  return `₹ ${grouped}`;
}


// --------------------------------------------------------------------------
// Server-side document store
//
// The bridge persists every successful extraction to disk so all clients
// (different browsers, machines on the LAN, fresh tabs) see the same list.
// The shape returned here matches the React `Document` interface.
// --------------------------------------------------------------------------

export interface ServerDocument {
  id: string;
  fileName: string;
  fileSize: string;
  fileType: string;
  mimeType?: string;
  previewUrl?: string;
  status: "completed" | "processing" | "failed";
  confidence?: number;
  processingTime?: string;
  language?: string;
  documentType?: string;
  extractedFields?: {
    dealerName?: string;
    modelName?: string;
    horsePower?: string;
    assetCost?: string;
    signatureDetected?: boolean;
    stampDetected?: boolean;
    signatureBbox?: [number, number, number, number] | null;
    stampBbox?: [number, number, number, number] | null;
  };
  uploadedAt: string;
  /** Optional: full ExtractionResult JSON returned by the pipeline. */
  extraction?: ExtractionResult;
}

/** Fetch all documents the bridge has persisted. */
export async function fetchDocuments(): Promise<ServerDocument[]> {
  const resp = await fetch(`${BRIDGE_BASE}/api/documents`);
  if (!resp.ok) {
    throw new Error(`Bridge returned ${resp.status}`);
  }
  return (await resp.json()) as ServerDocument[];
}

/** Permanently delete a document on the bridge. */
export async function deleteServerDocument(id: string): Promise<void> {
  const resp = await fetch(`${BRIDGE_BASE}/api/documents/${id}`, {
    method: "DELETE",
  });
  if (!resp.ok && resp.status !== 404) {
    throw new Error(`Bridge returned ${resp.status}`);
  }
}

/**
 * Upload a file. Returns a Document record already shaped for the React
 * store, plus the raw ExtractionResult under `extraction` for callers that
 * want it.
 */
export async function uploadAndExtract(file: File): Promise<ServerDocument> {
  const form = new FormData();
  form.append("file", file);

  const resp = await fetch(`${BRIDGE_BASE}/api/extract`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    const detail = await resp.text();
    throw new Error(`Bridge returned ${resp.status}: ${detail}`);
  }
  return (await resp.json()) as ServerDocument;
}
