import { Navigate, Route, Routes } from "react-router";
import type { AuthInfo } from "../types";
import { AllThesesPage } from "./AllThesesPage";
import { AppShell } from "./AppShell";
import { RecentPage } from "./RecentPage";

// Route table. Every page renders inside AppShell (header + sidebar + search),
// so navigating between them never remounts the chrome.
//
// "/" and "/thesis/:jobId" are the same page: the route parameter simply says
// which thesis it should be showing, and its absence means an empty workspace.
export function App({ auth }: { auth: AuthInfo }) {
  return (
    <Routes>
      <Route element={<AppShell auth={auth} />}>
        <Route path="/" element={<RecentPage auth={auth} />} />
        <Route path="/thesis/:jobId" element={<RecentPage auth={auth} />} />
        <Route path="/theses" element={<AllThesesPage auth={auth} />} />
        {/* Unknown paths fall back to the workspace rather than a dead end. */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
