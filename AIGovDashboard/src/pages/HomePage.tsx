import { useState, useEffect } from "react";
import {
  Search,
  FileText,
  Clock,
  CheckCircle,
  Plus,
  HelpCircle,
  Trash2,
  AlertCircle,
  X,
  Shield,
  Home,
  ChevronRight,
  BarChart3,
  TrendingUp,
  Activity,
  XCircle,
  Settings,
  Filter,
  ArrowUpRight,
  Zap,
  Target,
  Calendar,
  Users,
  Award,
  Sparkles,
  ChevronUp,
  ChevronDown,
} from "lucide-react";
import { Link, useNavigate } from "react-router-dom";
import { supabase } from "../lib/supabase";
import { useAuth } from "../contexts/AuthContext";
import OnboardingModal from "../components/OnboardingModal";

interface Project {
  project_id: string;
  project_name: string;
  description: string;
  project_type: "llm" | "generic";
  user_id: string;
}

// Database response structure
interface DbProject {
  project_id: string;
  project_name: string;
  description: string;
  project_type: "llm" | "generic";
  user_uuid: string;
}

// Audit and Report interfaces
interface Audit {
  id: number;
  project_id: number;
  audit_type: "performance" | "fairness_bias" | "explainability" | "drift";
  status: "completed" | "failed" | "running";
  results: any;
  created_at: string;
}

interface Report {
  id: number;
  project_id: number;
  model_id: number;
  dataset_id: number;
  user_id: number;
  report_type: string;
  blockchain_hash: string;
  file_path: string;
  report_metadata: any;
  created_at: string;
  updated_at: string;
}

interface ProjectAuditSummary {
  project_id: number;
  project_name: string;
  total_audits: number;
  completed_audits: number;
  failed_audits: number;
  reports_generated: number;
  pending_reports: number;
  has_failures: boolean;
  audits: Audit[];
  reports: Report[];
}

interface AuditStats {
  total_projects: number;
  total_audits: number;
  total_completed: number;
  total_failed: number;
  total_reports: number;
  pending_reports: number;
  failed_audits: Audit[];
}

const HomePage: React.FC = () => {
  const navigate = useNavigate();
  const { user } = useAuth();
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedStatus, setSelectedStatus] = useState<string | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [showOnboardingModal, setShowOnboardingModal] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<string | null>(null);
  const [showSuccessToast, setShowSuccessToast] = useState(false);
  const [deletedProjectName, setDeletedProjectName] = useState("");

  // System alerts states
  const [showSystemAlerts, setShowSystemAlerts] = useState(true);
  const [isAlertsCollapsed, setIsAlertsCollapsed] = useState(false);

  // Audit states
  const [auditStats, setAuditStats] = useState<AuditStats>({
    total_projects: 0,
    total_audits: 0,
    total_completed: 0,
    total_failed: 0,
    total_reports: 0,
    pending_reports: 0,
    failed_audits: [],
  });
  const [projectAuditSummaries, setProjectAuditSummaries] = useState<
    ProjectAuditSummary[]
  >([]);
  const [auditsLoading, setAuditsLoading] = useState(true);

  useEffect(() => {
    fetchProjects();
    checkFirstTimeUserStatus();
  }, []);

  useEffect(() => {
    console.log("User loaded in HomePage:", user);
  }, [user]); // This effect depends on user being loaded

  const fetchProjects = async () => {
    try {
      const userId = localStorage.getItem("userId");
      if (!userId) {
        navigate("/login");
        return;
      }

      console.log("userId!!!!!!!!!!!!!!!!!!:", userId);
      console.log(
        "access_token!!!!!!!!!!!!!!!!!!:",
        localStorage.getItem("access_token")
      );

      const { data, error } = await supabase
        .from("projectdetails")
        .select(
          "project_id, project_name, description, project_type, user_uuid"
        )
        .eq("user_uuid", userId);

      if (error) {
        throw error;
      }

      if (data && data.length > 0) {
        console.log(data[0].project_id);
        localStorage.setItem("projectId", data[0].project_id);
      }

      // Properly type the data as DbProject[]
      const typedData = (data as DbProject[]) || [];

      // Map the database results to match our Project interface (user_uuid -> user_id)
      const mappedProjects: Project[] = typedData.map((project) => ({
        project_id: project.project_id,
        project_name: project.project_name,
        description: project.description,
        project_type: project.project_type,
        user_id: project.user_uuid, // Map user_uuid to user_id
      }));

      setProjects(mappedProjects);

      // Refresh audit data after projects are loaded
      fetchAudits();
    } catch (error) {
      console.error("Error fetching projects:", error);
    } finally {
      setLoading(false);
    }
  };

  const fetchAudits = async () => {
    try {
      const userId = localStorage.getItem("userId");
      if (!userId) {
        setAuditsLoading(false);
        return;
      }

      // First, get all project IDs for this user
      const { data: userProjects, error: projectError } = await supabase
        .from("projectdetails")
        .select("project_id, project_name")
        .eq("user_uuid", userId);

      if (projectError) {
        throw projectError;
      }

      if (!userProjects || userProjects.length === 0) {
        setAuditsLoading(false);
        return;
      }

      const projectIds = userProjects.map((p) => p.project_id);

      // Fetch all audits for these projects
      const { data: audits, error: auditError } = await supabase
        .from("audits")
        .select("*")
        .in("project_id", projectIds)
        .order("created_at", { ascending: false });

      if (auditError) {
        throw auditError;
      }

      // Fetch all reports for these projects
      const { data: reports, error: reportError } = await supabase
        .from("reports")
        .select("*")
        .in("project_id", projectIds)
        .order("created_at", { ascending: false });

      if (reportError) {
        throw reportError;
      }

      const typedAudits = (audits as Audit[]) || [];
      const typedReports = (reports as Report[]) || [];

      console.log("Fetched audits:", typedAudits.length);
      console.log("Fetched reports:", typedReports.length);
      console.log("User projects:", userProjects.length);

      // Group audits and reports by project
      const projectSummaries: ProjectAuditSummary[] = userProjects.map(
        (project) => {
          // Convert project_id to number for comparison since audits and reports use numbers
          const projectIdNum = parseInt(project.project_id.toString());

          const projectAudits = typedAudits.filter(
            (audit) => audit.project_id === projectIdNum
          );
          const projectReports = typedReports.filter(
            (report) => report.project_id === projectIdNum
          );

          const completed = projectAudits.filter(
            (audit) => audit.status === "completed"
          ).length;
          const failed = projectAudits.filter(
            (audit) => audit.status === "failed"
          ).length;
          const running = projectAudits.filter(
            (audit) => audit.status === "running"
          ).length;

          // Calculate pending reports (completed audits without reports)
          const completedAudits = projectAudits.filter(
            (audit) => audit.status === "completed"
          );
          const pendingReports = completedAudits.length - projectReports.length;

          console.log(
            `Project ${project.project_name} (ID: ${project.project_id}):`,
            {
              total_audits: projectAudits.length,
              completed_audits: completed,
              failed_audits: failed,
              running_audits: running,
              reports_generated: projectReports.length,
              pending_reports: Math.max(0, pendingReports),
            }
          );

          return {
            project_id: projectIdNum,
            project_name: project.project_name,
            total_audits: projectAudits.length,
            completed_audits: completed,
            failed_audits: failed,
            reports_generated: projectReports.length,
            pending_reports: Math.max(0, pendingReports),
            has_failures: failed > 0,
            audits: projectAudits,
            reports: projectReports,
          };
        }
      );

      // Calculate overall stats
      const totalCompleted = typedAudits.filter(
        (audit) => audit.status === "completed"
      ).length;
      const totalFailed = typedAudits.filter(
        (audit) => audit.status === "failed"
      ).length;
      const failedAudits = typedAudits.filter(
        (audit) => audit.status === "failed"
      );
      const totalPendingReports = projectSummaries.reduce(
        (sum, project) => sum + project.pending_reports,
        0
      );

      setAuditStats({
        total_projects: userProjects.length,
        total_audits: typedAudits.length,
        total_completed: totalCompleted,
        total_failed: totalFailed,
        total_reports: typedReports.length,
        pending_reports: totalPendingReports,
        failed_audits: failedAudits,
      });

      setProjectAuditSummaries(projectSummaries);
    } catch (error) {
      console.error("Error fetching audits and reports:", error);
    } finally {
      setAuditsLoading(false);
    }
  };

  const checkFirstTimeUserStatus = async () => {
    try {
      const userId = localStorage.getItem("userId");
      if (!userId) {
        console.log("No userId found in localStorage");
        return;
      }

      const { data, error } = await supabase
        .from("userData")
        .select("firsttimeuserstatus")
        .eq("user_id", userId)
        .single();

      if (error) {
        console.error("Error fetching user data:", error);
        return;
      }

      console.log("First time user status:", data?.firsttimeuserstatus);

      // Only show onboarding modal if FirstTimeUserStatus is true
      if (data?.firsttimeuserstatus === true) {
        console.log("Showing onboarding modal based on FirstTimeUserStatus");
        setShowOnboardingModal(true);
      }
    } catch (error) {
      console.error("Error checking first time user status:", error);
    }
  };

  const filteredProjects = projects.filter((project) => {
    const matchesSearch =
      project.project_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      project.description.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesSearch;
  });

  const handleCloseOnboarding = () => {
    console.log("Closing onboarding modal");
    setShowOnboardingModal(false);
    // Remove the flag from localStorage
    localStorage.removeItem("showOnboarding");
  };

  const confirmDeleteProject = (
    e: React.MouseEvent,
    projectId: string,
    projectName: string
  ) => {
    e.preventDefault(); // Prevent navigation from the Link component
    e.stopPropagation(); // Stop the click from propagating

    setProjectToDelete(projectId);
    setDeletedProjectName(projectName);
    setShowDeleteConfirmation(true);
  };

  const handleDeleteProject = async () => {
    if (!projectToDelete || isDeleting) return;

    try {
      setIsDeleting(true);

      const { error } = await supabase
        .from("projectdetails")
        .delete()
        .eq("project_id", projectToDelete);

      if (error) {
        throw error;
      }

      // Show success message
      setShowSuccessToast(true);

      // Hide success message after 3 seconds
      setTimeout(() => {
        setShowSuccessToast(false);
      }, 3000);

      // Refresh the projects list and audit data after deletion
      fetchProjects();
      fetchAudits();
    } catch (error) {
      console.error("Error deleting project:", error);
      alert("Failed to delete project. Please try again.");
    } finally {
      setIsDeleting(false);
      setProjectToDelete(null);
      setShowDeleteConfirmation(false);
    }
  };

  const cancelDelete = () => {
    setProjectToDelete(null);
    setShowDeleteConfirmation(false);
  };

  // For debugging
  console.log(
    "HomePage render, showOnboardingModal:",
    showOnboardingModal,
    "user:",
    user
  );

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex-shrink-0 shadow-sm">
        <div className="h-full flex flex-col">
          {/* Navigation */}
          <nav className="flex-1 p-6 space-y-2">
            <Link
              to="/"
              className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-gray-50 text-gray-900 font-semibold border border-gray-200"
            >
              <div className="h-8 w-8 bg-emerald-400 rounded-lg flex items-center justify-center">
                <Home className="h-4 w-4 text-white" />
              </div>
              <span>Dashboard</span>
            </Link>

            <Link
              to="/iso"
              className="flex items-center space-x-3 px-4 py-3 rounded-xl text-gray-700 hover:bg-gray-50 hover:text-gray-900 font-medium transition-colors"
            >
              <div className="h-8 w-8 bg-blue-100 rounded-lg flex items-center justify-center">
                <Award className="h-4 w-4 text-blue-600" />
              </div>
              <span>ISO 42001 Audit</span>
            </Link>

            {/* Quick Stats in Sidebar */}
          </nav>

          {/* User Section */}
        </div>
      </div>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-8 py-6 sticky top-0 z-20">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center space-x-3 mb-2">
                <h1 className="text-3xl font-medium text-gray-900">
                  PRISM Dashboard DTCC <b>X</b> Hackathon
                </h1>
              </div>
              <p className="text-gray-600 font-medium">
                Monitor your AI systems and ensure compliance across all
                projects
              </p>
              <div className="flex items-center space-x-4 mt-2"></div>
            </div>
            <button
              onClick={() => navigate("/projects/new")}
              className="inline-flex items-center bg-emerald-400 px-6 py-3  text-white rounded-xl font-semibold hover:bg-slate-800 transition-colors shadow-lg"
            >
              <Plus className="h-4 w-4 mr-2" />
              New Project
            </button>
          </div>
        </header>

        <div className="p-8 space-y-8">
          {/* Key Metrics */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">
                    Total Projects
                  </p>
                  <p className="text-3xl font-bold text-gray-900">
                    {auditsLoading ? "—" : auditStats.total_projects}
                  </p>
                </div>
                <div className="h-12 w-12 bg-emerald-50 rounded-xl flex items-center justify-center">
                  <BarChart3 className="h-6 w-6 text-emerald-600" />
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">
                    Evaluations Completed
                  </p>
                  <p className="text-3xl font-bold text-gray-900">
                    {auditsLoading ? "—" : auditStats.total_completed || 0}
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    {auditsLoading
                      ? ""
                      : `Total: ${auditStats.total_audits || 0} | Failed: ${
                          auditStats.total_failed || 0
                        }`}
                  </p>
                </div>
                <div className="h-12 w-12 bg-emerald-50 rounded-xl flex items-center justify-center">
                  <CheckCircle className="h-6 w-6 text-emerald-600" />
                </div>
              </div>
            </div>

            <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm hover:shadow-md transition-shadow">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-gray-600 mb-1">
                    Analysis Reports Generated
                  </p>
                  <p className="text-3xl font-bold text-gray-900">
                    {auditsLoading ? "—" : auditStats.total_reports || 0}
                  </p>
                </div>
                <div className="h-12 w-12 bg-purple-50 rounded-xl flex items-center justify-center">
                  <FileText className="h-6 w-6 text-purple-600" />
                </div>
              </div>
            </div>
          </div>

          {/* System Alerts */}
          {auditStats.failed_audits.length > 0 && showSystemAlerts && (
            <div className="bg-white rounded-2xl border border-red-200 shadow-sm overflow-hidden">
              <div className="px-6 py-4 bg-red-50 border-b border-red-100">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-3">
                    <div className="h-8 w-8 bg-red-500 rounded-lg flex items-center justify-center">
                      <AlertCircle className="h-4 w-4 text-white" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-red-900">
                        System Alerts
                      </h3>
                      <p className="text-sm text-red-700">
                        Issues requiring immediate attention
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center space-x-3">
                    <span className="inline-flex items-center px-3 py-1 rounded-lg text-sm font-medium bg-red-100 text-red-800 border border-red-200">
                      {auditStats.failed_audits.length} Active
                    </span>
                    <button
                      onClick={() => setIsAlertsCollapsed(!isAlertsCollapsed)}
                      className="p-1.5 rounded-lg text-red-700 hover:bg-red-100 transition-colors"
                      title={
                        isAlertsCollapsed ? "Expand alerts" : "Collapse alerts"
                      }
                    >
                      {isAlertsCollapsed ? (
                        <ChevronDown className="h-4 w-4" />
                      ) : (
                        <ChevronUp className="h-4 w-4" />
                      )}
                    </button>
                    <button
                      onClick={() => setShowSystemAlerts(false)}
                      className="p-1.5 rounded-lg text-red-700 hover:bg-red-100 transition-colors"
                      title="Dismiss alerts"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                </div>
              </div>
              {!isAlertsCollapsed && (
                <div className="p-6">
                  <div className="space-y-3">
                    {auditStats.failed_audits.slice(0, 5).map((audit) => {
                      const project = projectAuditSummaries.find(
                        (p) =>
                          p.project_id.toString() ===
                          audit.project_id.toString()
                      );
                      return (
                        <div
                          key={audit.id}
                          className="flex items-center space-x-3 p-4 bg-red-50 rounded-xl border border-red-100 hover:bg-red-100 transition-colors cursor-pointer"
                        >
                          <div className="h-10 w-10 bg-red-500 rounded-lg flex items-center justify-center">
                            <XCircle className="h-5 w-5 text-white" />
                          </div>
                          <div className="flex-1 min-w-0">
                            <p className="text-sm font-semibold text-red-900">
                              {project?.project_name ||
                                `Project ${audit.project_id}`}
                            </p>
                            <p className="text-xs text-red-700 capitalize">
                              {audit.audit_type.replace("_", " ")} audit failed
                            </p>
                          </div>
                          <div className="flex items-center space-x-2">
                            <span className="px-2 py-1 bg-red-100 text-red-700 rounded-md text-xs font-medium">
                              High Priority
                            </span>
                            <ArrowUpRight className="h-4 w-4 text-red-400" />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Projects Overview */}
          <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-6 py-5 border-b border-gray-200 bg-gray-50">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className="h-10 w-10 bg-emerald-400 rounded-xl flex items-center justify-center">
                    <BarChart3 className="h-5 w-5 text-white" />
                  </div>
                  <div>
                    <h3 className="text-xl font-bold text-gray-900">
                      Projects
                    </h3>
                    <p className="text-sm text-gray-600">
                      Manage and monitor your AI systems
                    </p>
                  </div>
                </div>
                <div className="flex items-center space-x-3">
                  {/* Search */}
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                    <input
                      type="text"
                      placeholder="Search projects..."
                      className="pl-10 pr-4 py-2.5 w-64 text-sm border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-gray-500 focus:border-transparent bg-white"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                  </div>

                  {/* Filter */}
                  <button className="inline-flex items-center px-4 py-2.5 border border-gray-300 rounded-xl text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors">
                    <Filter className="h-4 w-4 mr-2" />
                    Filter
                  </button>
                </div>
              </div>
            </div>

            {/* Project List */}
            <div className="divide-y divide-gray-100">
              {loading ? (
                <div className="flex items-center justify-center py-16">
                  <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-b-2 border-gray-400"></div>
                </div>
              ) : filteredProjects.length === 0 ? (
                <div className="text-center py-16">
                  <div className="h-16 w-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
                    <FileText className="h-8 w-8 text-gray-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-2">
                    No projects found
                  </h3>
                  <p className="text-gray-600 mb-6 max-w-md mx-auto">
                    {searchQuery
                      ? "No projects match your search criteria."
                      : "Get started by creating your first project to monitor your AI systems."}
                  </p>
                  {!searchQuery && (
                    <button
                      onClick={() => navigate("/projects/new")}
                      className="inline-flex items-center px-6 py-3 bg-emerald-400 text-white rounded-xl font-medium hover:bg-slate-800 transition-colors"
                    >
                      <Plus className="h-4 w-4 mr-2" />
                      Create Your First Project
                    </button>
                  )}
                </div>
              ) : (
                filteredProjects.map((project) => {
                  const projectSummary = projectAuditSummaries.find(
                    (p) =>
                      p.project_id.toString() === project.project_id.toString()
                  );
                  return (
                    <div
                      key={project.project_id}
                      className="hover:bg-gray-50 transition-colors"
                    >
                      <Link
                        to={
                          project.project_type === "llm"
                            ? `/projects/${project.project_id}/redteaming`
                            : `/projects/${project.project_id}`
                        }
                        className="block px-6 py-5"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4 flex-1">
                            {/* Project Icon */}
                            <div className="h-12 w-12 bg-gray-100 rounded-xl flex items-center justify-center flex-shrink-0">
                              {project.project_type === "generic" ? (
                                <svg
                                  className="h-6 w-6 text-gray-600"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                >
                                  <path
                                    d="M12 4.75V6.25"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <path
                                    d="M17.127 6.873L16.073 7.927"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <path
                                    d="M19.25 12H17.75"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <path
                                    d="M17.127 17.127L16.073 16.073"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <path
                                    d="M12 19.25V17.75"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <path
                                    d="M6.873 17.127L7.927 16.073"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <path
                                    d="M4.75 12H6.25"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <path
                                    d="M6.873 6.873L7.927 7.927"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                  />
                                  <circle
                                    cx="12"
                                    cy="12"
                                    r="3"
                                    stroke="currentColor"
                                    strokeWidth="1.5"
                                  />
                                </svg>
                              ) : (
                                <Zap className="h-6 w-6 text-gray-600" />
                              )}
                            </div>

                            {/* Project Details */}
                            <div className="flex-1 min-w-0">
                              <div className="flex items-center space-x-3 mb-1">
                                <h4 className="text-lg font-semibold text-gray-900 truncate">
                                  {project.project_name}
                                </h4>
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-lg text-xs font-medium bg-gray-100 text-gray-700 border border-gray-200">
                                  {project.project_type === "llm"
                                    ? "LLM"
                                    : "Generic AI"}
                                </span>
                              </div>
                              <p className="text-gray-600 truncate">
                                {project.description}
                              </p>
                              <div className="flex items-center space-x-3 mt-1"></div>
                            </div>
                          </div>

                          {/* Project Stats */}
                          <div className="flex items-center space-x-6 text-sm text-gray-500 mr-4">
                            {projectSummary ? (
                              <>
                                <div className="text-center">
                                  <div className="text-xl font-bold text-gray-900">
                                    {projectSummary.total_audits || 0}
                                  </div>
                                  <div className="text-xs text-gray-500">
                                    Audits
                                  </div>
                                </div>
                                <div className="text-center">
                                  <div className="text-xl font-bold text-gray-900">
                                    {projectSummary.reports_generated || 0}
                                  </div>
                                  <div className="text-xs text-gray-500">
                                    Reports
                                  </div>
                                </div>

                                {projectSummary.failed_audits > 0 && (
                                  <div className="text-center">
                                    <div className="text-sm font-medium text-red-600">
                                      {projectSummary.failed_audits}
                                    </div>
                                    <div className="text-xs text-gray-500">
                                      Failed
                                    </div>
                                  </div>
                                )}
                                <div className="text-center min-w-[80px]">
                                  <div
                                    className={`inline-flex items-center px-3 py-1 rounded-lg text-xs font-medium ${
                                      projectSummary.has_failures
                                        ? "bg-red-100 text-red-700 border border-red-200"
                                        : projectSummary.reports_generated === 2
                                        ? "bg-emerald-100 text-emerald-700 border border-emerald-200"
                                        : projectSummary.pending_reports > 0
                                        ? "bg-yellow-100 text-yellow-700 border border-yellow-200"
                                        : projectSummary.total_audits > 0
                                        ? "bg-emerald-100 text-emerald-700 border border-emerald-200"
                                        : "bg-gray-100 text-gray-600 border border-gray-200"
                                    }`}
                                  >
                                    {projectSummary.has_failures ? (
                                      <>
                                        <XCircle className="h-3 w-3 mr-1" />
                                        Issues
                                      </>
                                    ) : projectSummary.reports_generated ===
                                      2 ? (
                                      <>
                                        <CheckCircle className="h-3 w-3 mr-1" />
                                        Report generation complete
                                      </>
                                    ) : projectSummary.pending_reports > 0 ? (
                                      <>
                                        <Clock className="h-3 w-3 mr-1" />
                                        Report generation Pending
                                      </>
                                    ) : projectSummary.total_audits > 0 ? (
                                      <>
                                        <CheckCircle className="h-3 w-3 mr-1" />
                                        Complete
                                      </>
                                    ) : (
                                      "No Data"
                                    )}
                                  </div>
                                </div>
                              </>
                            ) : (
                              <div className="text-center">
                                <div className="text-lg font-medium text-gray-400">
                                  —
                                </div>
                                <div className="text-xs text-gray-400">
                                  No Data
                                </div>
                              </div>
                            )}
                          </div>

                          {/* Actions */}
                          <div className="flex items-center space-x-2">
                            <ChevronRight className="h-5 w-5 text-gray-400" />
                            <button
                              onClick={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                confirmDeleteProject(
                                  e,
                                  project.project_id,
                                  project.project_name
                                );
                              }}
                              className="p-2 rounded-lg text-gray-400 hover:text-red-600 hover:bg-red-50 transition-colors"
                              title="Delete project"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      </Link>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Delete Confirmation Modal */}
      {showDeleteConfirmation && (
        <div className="fixed inset-0 bg-gray-900/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white rounded-2xl p-6 max-w-md w-full mx-4 shadow-xl">
            <div className="flex items-center justify-between mb-6">
              <h3 className="text-xl font-semibold text-gray-900">
                Delete Project
              </h3>
              <button
                onClick={cancelDelete}
                className="text-gray-400 hover:text-gray-500"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            <div className="mb-6">
              <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-2xl bg-red-100 mb-4">
                <AlertCircle className="h-8 w-8 text-red-600" />
              </div>
              <p className="text-center text-gray-600">
                Are you sure you want to delete this project? This action cannot
                be undone and all associated data will be permanently removed.
              </p>
            </div>
            <div className="flex justify-end space-x-3">
              <button
                onClick={cancelDelete}
                className="px-6 py-2.5 border border-gray-300 rounded-xl text-sm font-medium text-gray-700 hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleDeleteProject}
                className="px-6 py-2.5 bg-red-600 border border-transparent rounded-xl text-sm font-medium text-white hover:bg-red-700 transition-colors"
                disabled={isDeleting}
              >
                {isDeleting ? "Deleting..." : "Delete Project"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Success Toast */}
      {showSuccessToast && (
        <div className="fixed bottom-6 right-6 bg-white border border-gray-200 rounded-xl shadow-lg p-4 flex items-center space-x-3 z-50">
          <div className="h-8 w-8 bg-emerald-500 rounded-lg flex items-center justify-center">
            <CheckCircle className="h-4 w-4 text-white" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900">Success!</p>
            <p className="text-xs text-gray-600">
              Project deleted successfully
            </p>
          </div>
        </div>
      )}

      {/* Onboarding Modal */}
      {user && showOnboardingModal && (
        <OnboardingModal
          isOpen={true}
          onClose={handleCloseOnboarding}
          userId={user.id}
        />
      )}
    </div>
  );
};

export default HomePage;
