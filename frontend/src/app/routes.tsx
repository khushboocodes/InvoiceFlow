import { createBrowserRouter } from "react-router";
import { Layout } from "./components/Layout";
import { Landing } from "./pages/Landing";
import { Dashboard } from "./pages/Dashboard";
import { Documents } from "./pages/Documents";
import { Upload } from "./pages/Upload";
import { Analytics } from "./pages/Analytics";
import { ExtractionResult } from "./pages/ExtractionResult";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Landing,
  },
  {
    path: "/",
    Component: Layout,
    children: [
      { path: "dashboard", Component: Dashboard },
      { path: "documents", Component: Documents },
      { path: "upload", Component: Upload },
      { path: "analytics", Component: Analytics },
      { path: "results/:id", Component: ExtractionResult },
    ],
  },
]);
