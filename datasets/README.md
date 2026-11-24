# INVOICEFLOW DATASET DIRECTORY

Paste your dataset files according to the following layout:

```
datasets/
├── raw/             <-- PASTE YOUR RAW INVOICES HERE (~500 files: PDFs, JPGs, PNGs)
├── holdout_test/    <-- (Optional) Paste unseen evaluation test files here (~100 files)
├── processed/       <-- (Auto-generated) Normalized 300 DPI deskewed images
└── manifests/       <-- (Auto-generated) Train/Val splits and ground-truth metadata
```

### Supported Input Formats
* Digital & Scanned PDFs (`.pdf`)
* Images (`.jpg`, `.jpeg`, `.png`, `.webp`, `.tiff`)
