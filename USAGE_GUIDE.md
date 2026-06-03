# InvoiceFlow - Usage Guide

## 📖 How to Use the Dashboard

### 1️⃣ Getting Started

When you first open the application, you'll see the **Dashboard** with empty states. This is your command center for document processing.

**Empty State Features:**
- Zero statistics across all metrics
- Empty charts with "No data yet" placeholders
- Large CTA button to upload your first document

---

### 2️⃣ Uploading a Document

#### Navigate to Upload Page
Click **"Upload"** in the sidebar or the **"Process Document"** button.

#### Three States of Upload:

**STATE 1: BEFORE (Empty)**
- Drag & drop zone with dashed border
- Supported formats displayed: PDF, PNG, JPG, JPEG
- Click "Browse Files" to select from your computer

**STATE 2: READY (File Selected)**
- File card shows with green border
- File metadata displayed (name, size, type)
- "Ready to process" confirmation message
- Large "Process Document with AI" button appears

**STATE 3: PROCESSING (Active)**
- Animated loading spinner with purple glow
- Processing status message
- Technology badges: OCR, NLP, Vision, Structuring
- AI Pipeline steps highlight in real-time:
  1. Document Ingestion (Purple → Green when complete)
  2. OCR Extraction
  3. Field Detection
  4. Vision Analysis
  5. Structured Output

**Processing Time:** ~5 seconds (simulated)

---

### 3️⃣ Viewing Results

After processing completes, you're automatically redirected to the **Results Page**.

#### Results Page Sections:

**Header:**
- Document filename
- Completion status badge (green)
- Action buttons: "Explain AI" and "View File"

**Statistics Cards:**
- **Confidence Score**: 95% (AI confidence in extraction)
- **Processing Time**: 17.4s
- **Language**: English (detected)
- **Document Type**: Tractor Quotation

**Document Preview (Left):**
- Visual representation of the document
- Color-coded bounding boxes:
  - 🟢 Green: Signature detection
  - 🔵 Blue: Stamp detection
- Coordinates displayed on hover

**Extracted Fields (Right):**
- **Dealer Name**: MADHU PAVAN AUTOMOBILES
- **Model Name**: NH 3032 TX DRIVE
- **Horse Power**: 35 HP
- **Asset Cost**: ₹6,80,000
- **Signature Detected**: ✓ with coordinates
- **Stamp Detected**: ✓ with coordinates

**JSON Output:**
- Full structured data export
- Copy to clipboard functionality
- Formatted with syntax highlighting

---

### 4️⃣ Managing Documents

Navigate to the **Documents** page to see all processed files.

#### Features:

**Search & Filter:**
- Full-text search bar
- Filter chips: All, Completed, Processing, Failed
- Results counter

**Document Table Columns:**
- **Document**: File icon + name + size
- **Status**: Color-coded badge (Green/Purple/Red)
- **Confidence**: Progress bar + percentage
- **Fields Extracted**: Purple dots indicator + count (e.g., "4/4")
- **Language**: Detected language
- **Date**: Upload date
- **Action**: View button (links to results)

**Visual Indicators:**
- 🟣 Purple file icons for consistency
- 🟢 Green confidence bars
- 6 purple dots = all fields detected
- Hover effects on rows

---

### 5️⃣ Analytics Dashboard

View comprehensive analytics on the **Analytics** page.

#### Summary Bar (Top):
- Total documents processed
- Completed count
- Average confidence score
- Average processing time

#### Charts:

**1. Document Volume (14 days)**
- Line chart showing daily processing trends
- Helps identify peak usage periods

**2. Field Extraction Success**
- Horizontal bar chart
- Shows success rate for each field type:
  - Dealer Name
  - Model Name
  - Horse Power
  - Asset Cost
  - Signature
  - Stamp

**3. Language Distribution**
- Donut chart with legend
- Visualizes document language breakdown
- Currently shows English (100%)

---

### 6️⃣ Navigation

#### Sidebar Menu:
- **Dashboard**: Overview and statistics
- **Documents**: All processed files
- **Upload**: New document processing
- **Analytics**: Performance insights
- **Sign Out**: Exit application

#### Top Bar:
- Page title
- Search bar (works globally)
- System status: "System Online" 🟢

---

## 🎨 Visual Features

### Color Coding:
- **Purple (#6C5CE7)**: Primary brand, active states
- **Green**: Success, completion, high confidence
- **Red**: Errors, failed processing
- **Orange**: Warnings, medium confidence
- **Gray**: Neutral, inactive states

### Animations:
- Smooth page transitions
- Hover effects on buttons and cards
- Processing pulse animation
- Chart loading states
- Fade-in for new content

### Glassmorphism:
- Subtle backdrop blur on cards
- Semi-transparent backgrounds
- Layered depth with shadows

---

## 🔄 Workflow Example

1. **Start**: Open dashboard → See empty state
2. **Upload**: Click "Upload" → Drag PDF file
3. **Process**: Click "Process with AI" → Watch pipeline
4. **Results**: Auto-redirect → View extracted data
5. **Review**: Check confidence scores and fields
6. **Copy**: Export JSON for integration
7. **Dashboard**: Return to see updated stats
8. **Analytics**: View processing trends

---

## 💡 Pro Tips

1. **Batch Processing**: Upload multiple documents sequentially
2. **Confidence Threshold**: Focus on documents with 90%+ confidence
3. **Field Validation**: Always verify critical fields manually
4. **Export Data**: Use JSON output for API integration
5. **Monitor Analytics**: Track success rates over time

---

## 🆘 Troubleshooting

**Issue**: File won't upload
- **Solution**: Check file format (PDF, PNG, JPG only)

**Issue**: Processing stuck
- **Solution**: Refresh page and try again

**Issue**: Low confidence score
- **Solution**: Try higher quality scan/image

---

## 📊 Understanding Metrics

### Confidence Score
- **90-100%**: Excellent, reliable extraction
- **80-89%**: Good, minor review needed
- **70-79%**: Fair, verification recommended
- **<70%**: Poor, manual entry suggested

### Processing Time
- **Normal**: 10-20 seconds per document
- **Fast**: <10 seconds (simple documents)
- **Slow**: >20 seconds (complex layouts)

### Field Detection
- **4/4**: All standard fields found ✓
- **3/4**: One field missing ⚠️
- **<3/4**: Multiple fields missing ✗

---

**Built with ❤️ for seamless document processing**
