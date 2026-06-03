# InvoiceFlow - AI-Powered Document Processing SaaS Dashboard

A premium, pixel-perfect SaaS dashboard for AI-powered document processing and field extraction. Built with React, TypeScript, and Tailwind CSS.

## 🎨 Design System

### Colors
- **Primary**: `#6C5CE7` (Purple)
- **Secondary**: `#4F46E5` (Indigo)
- **Sidebar**: `#0F172A` (Dark Navy)
- **Background**: `#F8FAFC` (Light Gray)
- **Cards**: `#FFFFFF` (White)
- **Border**: `#E5E7EB` (Gray 200)

### Status Colors
- **Success**: `#22C55E` (Green)
- **Processing**: `#8B5CF6` (Purple)
- **Warning**: `#F59E0B` (Orange)
- **Error**: `#EF4444` (Red)

### Design Features
- ✨ Glassmorphism effects on cards
- 🎨 Gradient glow buttons with hover animations
- 🌊 Smooth transitions and micro-interactions
- 📐 8px grid spacing system
- 💎 Premium Stripe/Linear-inspired UI

## 📱 Features

### Dashboard
- Real-time statistics (Total Documents, Accuracy Rate, Avg Confidence, Avg Processing Time)
- Interactive charts (Line, Bar, Donut)
- Recent documents table
- Empty state with CTA

### Documents Page
- Advanced search and filtering
- Filter by status (All, Completed, Processing, Failed)
- Document table with:
  - File icons and metadata
  - Status badges
  - Confidence progress bars
  - Field extraction indicators
  - Quick actions

### Upload Page
Three-state upload flow:
1. **BEFORE**: Drag & drop zone with file type indicators
2. **READY**: File uploaded card with process button
3. **PROCESSING**: Animated loader with 5-step AI pipeline visualization

### Analytics
- Summary statistics bar
- 14-day document volume chart
- Field extraction success rates
- Language distribution
- Processing performance metrics

### Extraction Results
- Document metadata and status
- Performance statistics
- Document preview with AI-detected bounding boxes
- Extracted fields display
- JSON output viewer with copy functionality

## 🛠️ Tech Stack

- **Framework**: React 18.3 + TypeScript
- **Routing**: React Router 7
- **Styling**: Tailwind CSS v4
- **State Management**: Zustand
- **Charts**: Recharts
- **Icons**: Lucide React
- **Animations**: Motion (Framer Motion)

## 🚀 Getting Started

### Installation

```bash
# Install dependencies
pnpm install

# Start development server
pnpm dev

# Build for production
pnpm build
```

## 📂 Project Structure

```
src/
├── app/
│   ├── components/
│   │   ├── Layout.tsx           # Main layout with sidebar & topbar
│   │   └── figma/               # Protected components
│   ├── pages/
│   │   ├── Dashboard.tsx        # Main dashboard
│   │   ├── Documents.tsx        # Document list
│   │   ├── Upload.tsx           # Upload interface
│   │   ├── Analytics.tsx        # Analytics page
│   │   └── ExtractionResult.tsx # Results viewer
│   ├── store/
│   │   └── documentStore.ts     # Zustand state management
│   ├── routes.tsx               # React Router configuration
│   └── App.tsx                  # Root component
├── styles/
│   ├── custom.css               # Custom styles (glassmorphism, gradients)
│   ├── theme.css                # Design tokens
│   └── tailwind.css             # Tailwind imports
```

## 🎯 Key Components

### Layout Component
- Fixed dark navy sidebar with gradient
- Logo and navigation menu
- Top bar with search and system status
- Active route highlighting

### Document Store (Zustand)
```typescript
interface Document {
  id: string;
  fileName: string;
  status: 'processing' | 'completed' | 'failed';
  confidence?: number;
  extractedFields?: {...};
  // ... more fields
}
```

### AI Pipeline Simulation
The upload flow simulates a 5-step AI processing pipeline:
1. Document Ingestion
2. OCR Extraction
3. Field Detection
4. Vision Analysis
5. Structured Output

## 💅 Custom Styles

### Glassmorphism Cards
```css
.glass-card {
  background: rgba(255, 255, 255, 0.7);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(229, 231, 235, 0.3);
}
```

### Gradient Buttons
```css
.gradient-button {
  background: linear-gradient(135deg, #6C5CE7 0%, #4F46E5 100%);
  box-shadow: 0 4px 14px 0 rgba(108, 92, 231, 0.25);
}
```

## 🎨 Design Principles

1. **Consistency**: Uniform spacing (8px grid), typography, and colors
2. **Clarity**: Clear hierarchy and visual feedback
3. **Elegance**: Subtle animations and premium aesthetics
4. **Usability**: Intuitive navigation and workflows

## 📊 State Management

The app uses Zustand for lightweight, performant state management:
- Document collection
- Current document selection
- Processing state tracking

## 🔄 Workflow

1. **Upload** → User drops/selects document
2. **Processing** → Simulated AI pipeline runs through 5 steps
3. **Results** → View extracted fields, bounding boxes, and JSON
4. **Analytics** → Track performance across all documents

## 🎯 Future Enhancements

- Real backend integration
- Multi-language support
- Batch processing
- Export functionality
- User authentication
- Document templates
- Advanced filtering
- Real-time collaboration

## 📄 License

This is a demonstration project created for Figma Make.

---

**Built with ❤️ using Figma Make**
