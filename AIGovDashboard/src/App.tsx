import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import { TooltipProvider } from "./components/ui/tooltip";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import HomePage from "./pages/HomePage";
import NewProjectPage from "./pages/NewProjectPage";
import ProjectOverviewPage from "./pages/ProjectOverviewPage";
import PerformancePage from "./pages/PerformancePage";
import AppSidebar from "./components/AppSidebar";
import FairnessPage from "./pages/FairnessPage";
import ExplainabilityPage from "./pages/ExplainabilityPage";
import { ReactNode } from "react";
import ReportPage from "./pages/ReportPage";
import AppLayout from "./components/AppLayout";
import BenchmarkingPage from "./pages/BenchmarkingPage";
import RedTeamingPage from "./pages/RedTeamingPage";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import AuthCallback from "./components/AuthCallback";
import ProfilePage from "./pages/ProfilePage";
import DriftAnalysisPage from "./pages/DriftAnalysisPage";
import RiskAssessmentPage from "./pages/RiskAssessmentPage";
// Import the correct OverviewPage if it exists elsewhere or remove if not needed
// import OverviewPage from './pages/OverviewPage';

// Layout component that handles which sidebar to show
interface SidebarManagerProps {
  children: ReactNode;
  path: string;
}

const SidebarManager: React.FC<SidebarManagerProps> = ({ children, path }) => {
  let Sidebar = null;

  if (path.startsWith("/projects/")) {
    Sidebar = AppSidebar;
  }

  return (
    <div className="flex">
      {Sidebar && <Sidebar />}
      <main className={`${!Sidebar ? "w-full" : "flex-1"}`}>{children}</main>
    </div>
  );
};

// Create a Protected Route component
interface ProtectedRouteProps {
  children: React.ReactNode;
}

const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children }) => {
  const { user, loading } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!loading && !user) {
      navigate("/signup");
    }
  }, [user, loading, navigate]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary"></div>
      </div>
    );
  }

  return user ? <>{children}</> : null;
};

function App() {
  return (
    <Router>
      <AuthProvider>
        <TooltipProvider>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />

            {/* Protected routes */}
            <Route
              path="/home"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/home">
                      <HomePage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            {/* Project routes */}
            <Route
              path="/projects/new"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/new">
                      <NewProjectPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id">
                      <ProjectOverviewPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/performance"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/performance">
                      <PerformancePage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/fairness"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/fairness">
                      <FairnessPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/explainability"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/explainability">
                      <ExplainabilityPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/reports"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/reports">
                      <ReportPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/benchmarking"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/benchmarking">
                      <BenchmarkingPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/drift"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/drift">
                      <DriftAnalysisPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/redteaming"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/redteaming">
                      <RedTeamingPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            <Route
              path="/projects/:id/risk-assessment"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={false}>
                    <SidebarManager path="/projects/:id/risk-assessment">
                      <RiskAssessmentPage />
                    </SidebarManager>
                  </AppLayout>
                </ProtectedRoute>
              }
            />

            {/* If OverviewPage exists, uncomment and use this route */}
            {/* <Route path="/projects/:id/overview" element={
              <AppLayout showSidebar={false}>
                <SidebarManager path="/projects/:id/overview">
                  <OverviewPage />
                </SidebarManager>
              </AppLayout>
            } /> */}

            <Route path="/auth/callback" element={<AuthCallback />} />

            <Route path="/" element={<SignupPage />} />
            <Route
              path="/profile"
              element={
                <ProtectedRoute>
                  <AppLayout showSidebar={true} showHeader={true}>
                    <ProfilePage />
                  </AppLayout>
                </ProtectedRoute>
              }
            />
          </Routes>
        </TooltipProvider>
      </AuthProvider>
    </Router>
  );
}

export default App;
