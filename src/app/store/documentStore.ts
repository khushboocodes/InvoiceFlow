import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";

export interface Document {
  id: string;
  fileName: string;
  fileSize: string;
  fileType: string;
  /**
   * `data:` URL of the original uploaded file. Persisted to localStorage so
   * the preview survives a page reload. Files larger than ~4 MB are saved
   * without the data URL (we can't fit them under the localStorage quota);
   * those documents fall back to the placeholder preview after a reload.
   */
  previewUrl?: string;
  /**
   * Mime type of the uploaded file. Used by the result page to decide
   * whether to render an `<img>` or an `<embed type="application/pdf">`.
   */
  mimeType?: string;
  status: "processing" | "completed" | "failed";
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
}

interface DocumentStore {
  documents: Document[];
  currentDocument: Document | null;
  /** True after the first server hydrate finishes (success OR failure). */
  hydrated: boolean;
  addDocument: (doc: Document) => void;
  updateDocument: (id: string, updates: Partial<Document>) => void;
  setCurrentDocument: (doc: Document | null) => void;
  removeDocument: (id: string) => void;
  clearDocuments: () => void;
  /** Replace the in-memory list with a fresh copy from the server. */
  setDocuments: (docs: Document[]) => void;
  setHydrated: (v: boolean) => void;
}

export const useDocumentStore = create<DocumentStore>()(
  persist(
    (set) => ({
      documents: [],
      currentDocument: null,
      hydrated: false,
      addDocument: (doc) =>
        set((state) => ({
          documents: [doc, ...state.documents.filter((d) => d.id !== doc.id)],
        })),
      updateDocument: (id, updates) =>
        set((state) => ({
          documents: state.documents.map((doc) =>
            doc.id === id ? { ...doc, ...updates } : doc,
          ),
          currentDocument:
            state.currentDocument?.id === id
              ? { ...state.currentDocument, ...updates }
              : state.currentDocument,
        })),
      setCurrentDocument: (doc) => set({ currentDocument: doc }),
      removeDocument: (id) =>
        set((state) => ({
          documents: state.documents.filter((d) => d.id !== id),
          currentDocument:
            state.currentDocument?.id === id ? null : state.currentDocument,
        })),
      clearDocuments: () => set({ documents: [], currentDocument: null }),
      setDocuments: (docs) =>
        set((state) => {
          // Merge server docs with any local-only "processing" placeholders
          // so an in-flight upload doesn't disappear from the dashboard
          // every time the periodic hydrate runs.
          const inFlight = state.documents.filter(
            (d) => d.status === "processing" && !docs.some((sd) => sd.id === d.id),
          );
          return { documents: [...inFlight, ...docs] };
        }),
      setHydrated: (v) => set({ hydrated: v }),
    }),
    {
      name: "invoiceflow.documents.v1",
      storage: createJSONStorage(() => localStorage),
      // Persist the documents list (used as a fast offline cache while we
      // wait for the server hydrate). currentDocument is per-session.
      partialize: (state) => ({ documents: state.documents }),
      version: 1,
    },
  ),
);
