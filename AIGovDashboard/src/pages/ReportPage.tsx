import React, { useState, useEffect } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { supabase } from "../lib/supabase";

interface Report {
  id: number;
  type: string;
  category: string;
  hash: string;
  size: string;
  date: string;
  metrics: { name: string; value: string }[];
  blockchainHash: string;
}

interface ModelData {
  model_id: string;
  model_version: string;
  project_id: string;
  dataset_id: string;
}

interface ProjectDetails {
  project_id: string;
  user_uuid: string;
  project_name: string;
  description: string;
  project_type: string;
  project_status: string;
}

interface AuditData {
  id: number;
  project_id: string;
  model_id: string;
  dataset_id: string;
  user_id: string;
  audit_type: string;
  status: string;
  created_at?: string;
}

// Report generation loading states
enum ReportGenerationStatus {
  IDLE = "idle",
  GENERATING = "generating",
  DOWNLOADING = "downloading",
  COMPLETED = "completed",
  ERROR = "error",
}

// Mock data for reports
const mockReports = [
  {
    id: 1,
    type: "Performance Analysis Report",
    category: "Performance",
    hash: "0x1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b",
    size: "2.4 MB",
    date: "about 1 year ago",
    metrics: [
      { name: "Accuracy", value: "95.0%" },
      { name: "F1 Score", value: "94.0%" },
      { name: "Auc Roc", value: "96.0%" },
    ],
    blockchainHash: "0x9a8b7c6d5e4f3a2b1c0d1e2f3a4b5c6d7e8f9a0b",
  },
  {
    id: 2,
    type: "Fairness Audit Report",
    category: "Fairness",
    hash: "0x2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1",
    size: "1.8 MB",
    date: "about 1 year ago",
    metrics: [
      { name: "Gender Disparity", value: "2.0%" },
      { name: "Age Disparity", value: "3.0%" },
      { name: "Demographic Parity", value: "98.0%" },
    ],
    blockchainHash: "0x8b7c6d5e4f3a2b1c0d1e2f3a4b5c6d7e8f9a0b1c",
  },
];

const ReportPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState("");
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const [isLoading, setIsLoading] = useState(false);
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [reports, setReports] = useState<Report[]>([]);
  const [projectDetails, setProjectDetails] = useState<ProjectDetails | null>(
    null
  );
  const [auditData, setAuditData] = useState<AuditData[]>([]);
  const [projectData, setProjectData] = useState<{
    project_id: string;
    user_id: string;
    subscription_tier?: string;
  } | null>(null);
  const [reportGeneration, setReportGeneration] = useState<{
    status: ReportGenerationStatus;
    message: string;
  }>({
    status: ReportGenerationStatus.IDLE,
    message: "",
  });
  const [hasRiskAssessment, setHasRiskAssessment] = useState(false);
  const [riskAssessmentTimestamp, setRiskAssessmentTimestamp] = useState<
    string | null
  >(null);
  const [isRedirecting, setIsRedirecting] = useState(false);

  // Check if this is a dummy project
  const isDummyProject = id === "dummy-1" || id === "dummy-2";
  const isFreeUser =
    projectData?.subscription_tier === "free" ||
    !projectData?.subscription_tier;

  const fetchProjectData = async () => {
    try {
      setIsLoading(true);

      // Fetch project data from Supabase
      const { data, error } = await supabase
        .from("projects")
        .select("project_id, user_id, subscription_tier")
        .eq("project_id", id)
        .single();

      if (error) {
        console.error("Error fetching project data:", error);
        return;
      }

      console.log(id);

      if (data) {
        setProjectData(data);
        // Here you could also fetch project-specific reports
        // For now, using empty array or could fetch from another table
        setReports([]);
      }
    } catch (error) {
      console.error("Failed to fetch project data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchModelData = async () => {
    try {
      setIsLoading(true);

      // Fetch model data from Supabase
      const { data, error } = await supabase
        .from("modeldetails")
        .select("model_id, project_id, dataset_id, model_version")
        .eq("project_id", id)
        .limit(1);

      if (error) {
        console.error("Error fetching model data:", error);
        return;
      }

      console.log(data);

      if (data && data.length > 0) {
        setModelData(data[0]);
      }
    } catch (error) {
      console.error("Failed to fetch model data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchProjectDetails = async () => {
    try {
      setIsLoading(true);

      // Fetch project details from Supabase
      const { data, error } = await supabase
        .from("projectdetails")
        .select(
          "project_id, user_uuid, project_name, description, project_type, project_status"
        )
        .eq("project_id", id)
        .single();

      if (error) {
        console.error("Error fetching project details:", error);
        return;
      }

      if (data) {
        setProjectDetails(data);
      }
    } catch (error) {
      console.error("Failed to fetch project details:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchAuditData = async () => {
    try {
      setIsLoading(true);

      // Fetch audit data from Supabase
      const { data, error } = await supabase
        .from("audits")
        .select(
          "id, project_id, model_id, dataset_id, user_id, audit_type, status, created_at"
        )
        .eq("project_id", id)
        .order("created_at", { ascending: false });

      if (error) {
        console.error("Error fetching audit data:", error);
        return;
      }

      if (data) {
        setAuditData(data);
      }
    } catch (error) {
      console.error("Failed to fetch audit data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    if (isDummyProject) {
      // Use mock data for dummy projects
      setReports(mockReports);
    } else if (id) {
      // Fetch real data for actual projects
      fetchProjectData();
      fetchProjectDetails();
      fetchModelData();
      fetchAuditData();
    }
  }, [id, isDummyProject]);

  useEffect(() => {
    // Check if risk assessment was generated for this project
    const wasGenerated = localStorage.getItem(`riskAssessmentGenerated_${id}`);
    const timestamp = localStorage.getItem(`riskAssessmentTimestamp_${id}`);
    setHasRiskAssessment(wasGenerated === "true");
    setRiskAssessmentTimestamp(timestamp);
  }, [id]);

  // Add Loading Overlay Component
  const LoadingOverlay = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-8 max-w-sm w-full mx-4 text-center">
        <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          Redirecting...
        </h3>
        <p className="text-gray-600">Taking you to the ISO 42001 Audit page</p>
      </div>
    </div>
  );

  const handleDownloadRiskAssessment = async () => {
    try {
      // 1. Set in sessionStorage for persistence through refresh
      const sectionsToAutoComplete = [
        "impact-assessment",
        "testing-framework",
        "monitoring-systems",
        "reporting-mechanisms",
      ];
      sessionStorage.setItem(
        "iso42001_auto_sections",
        JSON.stringify(sectionsToAutoComplete)
      );
      console.log("Set sections in sessionStorage:", sectionsToAutoComplete);

      // 2. Then start the download process
      setReportGeneration({
        status: ReportGenerationStatus.GENERATING,
        message: "Retrieving AI Risk Assessment analysis...",
      });

      const storedAnalysis = localStorage.getItem(`riskAssessment_${id}`);
      if (!storedAnalysis) {
        alert(
          "No AI Risk Assessment analysis found. Please complete the risk assessment first."
        );
        setReportGeneration({
          status: ReportGenerationStatus.IDLE,
          message: "",
        });
        return;
      }

      const analysisData = JSON.parse(storedAnalysis);
      const projectName = analysisData.projectName || "AI System";

      setReportGeneration({
        status: ReportGenerationStatus.DOWNLOADING,
        message: "Downloading stored PDF report...",
      });

      if (analysisData.pdfData) {
        // Convert base64 and download
        const pdfBase64 = analysisData.pdfData;
        const byteCharacters = atob(pdfBase64);
        const byteNumbers = new Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        const blob = new Blob([byteArray], { type: "application/pdf" });

        const url = window.URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `AI_Risk_Assessment_Report_${projectName.replace(
          /\s+/g,
          "_"
        )}.pdf`;
        document.body.appendChild(a);
        a.click();

        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        setReportGeneration({
          status: ReportGenerationStatus.COMPLETED,
          message: "AI Risk Assessment report downloaded successfully!",
        });

        // Show loading overlay
        setIsRedirecting(true);

        // Wait for 2 seconds before navigating
        await new Promise((resolve) => setTimeout(resolve, 2000));

        // Navigate with a URL parameter as backup
        console.log("Download complete, navigating to /iso");
        navigate("/iso?autoComplete=true");
      } else {
        // Handle PDF generation fallback
        // ... existing fallback code ...
      }
    } catch (error) {
      console.error("Error in handleDownloadRiskAssessment:", error);
      setReportGeneration({
        status: ReportGenerationStatus.ERROR,
        message:
          "Failed to download AI Risk Assessment report. Please try again.",
      });
      setIsRedirecting(false);
    }
  };

  const handleDownloadReport = async (report: Report) => {
    if (isDummyProject) {
      // Show a mock download for dummy projects
      alert("Downloading demo report...");
      return;
    }

    // Handle AI Risk Assessment report download
    if (report.type === "AI Risk Assessment") {
      await handleDownloadRiskAssessment();
      return;
    }

    try {
      const accessToken = localStorage.getItem("access_token");
      if (!accessToken) {
        alert("You must be logged in to download reports");
        navigate("/login");
        return;
      }

      if (!modelData) {
        throw new Error("Model data is not available");
      }
      console.log(modelData);

      // Set loading state for report generation
      setReportGeneration({
        status: ReportGenerationStatus.GENERATING,
        message: "Generating report. This may take a few moments...",
      });

      // First API call to generate the report
      const generateResponse = await fetch(
        `http://localhost:8000/ml/${modelData.project_id}/reports/generate?model_id=${modelData.model_id}&dataset_id=${modelData.dataset_id}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        }
      );

      if (!generateResponse.ok) {
        throw new Error(
          `Failed to generate report: ${generateResponse.statusText}`
        );
      }

      // Get the generation result data
      const generationResult = await generateResponse.json();
      console.log("Report generation successful:", generationResult);

      // Update loading state for report download
      setReportGeneration({
        status: ReportGenerationStatus.DOWNLOADING,
        message: "Report generated. Downloading file...",
      });

      // Second API call to download the report file after successful generation
      const downloadResponse = await fetch(
        `http://localhost:8000/ml/download/${modelData.project_id}/${modelData.model_id}/${modelData.model_version}`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${accessToken}`,
          },
        }
      );

      if (!downloadResponse.ok) {
        throw new Error(
          `Failed to download report: ${downloadResponse.statusText}`
        );
      }

      // Get filename from content-disposition header
      const contentDisposition = downloadResponse.headers.get(
        "content-disposition"
      );
      console.log("Content-Disposition header:", contentDisposition);
      let filename = "report.pdf"; // Default fallback

      // Check content type to use appropriate extension if needed
      const contentType = downloadResponse.headers.get("content-type");
      console.log("Content-Type:", contentType);

      // Extract filename from content-disposition if present
      if (contentDisposition) {
        // Try different patterns to extract filename
        // First, try with standard format
        let filenameRegex = /filename[^;=\n]*=((['"]).*?\2|[^;\n]*)/;
        let matches = filenameRegex.exec(contentDisposition);

        // If that doesn't work, try alternative format with filename*
        if (!matches || !matches[1]) {
          filenameRegex = /filename\*=UTF-8''([^;]*)/i;
          matches = filenameRegex.exec(contentDisposition);
        }

        // If that doesn't work, try another common format
        if (!matches || !matches[1]) {
          filenameRegex = /filename="([^"]*)"/i;
          matches = filenameRegex.exec(contentDisposition);
        }

        console.log("Filename matches:", matches);
        if (matches && matches[1]) {
          // Remove quotes if present and decode URI components
          filename = matches[1].replace(/['"]/g, "").trim();
          try {
            // Handle URL encoded characters
            if (filename.includes("%")) {
              filename = decodeURIComponent(filename);
            }
          } catch (e) {
            console.error("Error decoding filename:", e);
          }
        }
      }

      // If we couldn't get a proper filename with extension, use content-type
      if (filename === "report.pdf" || !filename.includes(".")) {
        const contentTypeExtensionMap: Record<string, string> = {
          "application/pdf": ".pdf",
          "application/json": ".json",
          "text/csv": ".csv",
          "application/vnd.ms-excel": ".xls",
          "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            ".xlsx",
          "text/plain": ".txt",
          "application/zip": ".zip",
        };

        if (contentType && contentTypeExtensionMap[contentType]) {
          const baseName = filename.split(".")[0] || "report";
          filename = baseName + contentTypeExtensionMap[contentType];
        }
      }

      console.log("Using filename:", filename);

      // Convert response to blob and download
      const blob = await downloadResponse.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();

      // Clean up
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);

      // Set completion state
      setReportGeneration({
        status: ReportGenerationStatus.COMPLETED,
        message: "Report downloaded successfully!",
      });

      // Reset status after a delay
      setTimeout(() => {
        setReportGeneration({
          status: ReportGenerationStatus.IDLE,
          message: "",
        });
      }, 3000);
    } catch (error) {
      console.error("Error downloading report:", error);
      setReportGeneration({
        status: ReportGenerationStatus.ERROR,
        message: "Failed to generate or download report. Please try again.",
      });
      alert("Failed to download report. Please try again later.");
    }
  };

  // Filter reports based on search term
  const filteredReports = reports.filter(
    (report) =>
      report.type.toLowerCase().includes(searchTerm.toLowerCase()) ||
      report.hash.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="bg-gray-50 min-h-screen">
      <div className="container mx-auto px-6 py-8 max-w-7xl">
        {/* Breadcrumb */}
        <div className="flex items-center text-sm text-gray-500 mb-4">
          <span
            className="cursor-pointer hover:text-teal-600 transition-colors"
            onClick={() => navigate("/projects")}
          >
            Projects
          </span>
          <span className="mx-2">/</span>
          <span
            className="cursor-pointer hover:text-teal-600 transition-colors"
            onClick={() => navigate(`/project/${id}`)}
          >
            Project Details
          </span>
          <span className="mx-2">/</span>
          <span className="font-medium">Reports</span>
        </div>

        {/* Page Header with gradient */}
        <div className="bg-gradient-to-r from-teal-500 to-blue-500 rounded-lg shadow-lg mb-8 p-8 text-white">
          <h1 className="text-3xl font-bold mb-2">Analysis Reports</h1>
          <p className="text-teal-50">
            View and download comprehensive model analysis reports
          </p>
        </div>

        {/* Enhanced Project Information and Download Card */}
        <div className="bg-white border border-gray-200 rounded-xl shadow-lg mb-8 overflow-hidden">
          {/* Header Section with Gradient */}
          <div className="bg-gradient-to-r from-teal-50 to-blue-50 border-b border-gray-200 px-8 py-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  {projectDetails?.project_name || `Project ${id}`}
                </h2>
                <p className="text-gray-600 text-sm max-w-2xl">
                  {projectDetails?.description ||
                    "Comprehensive AI model analysis and reporting"}
                </p>
              </div>
              <div className="flex items-center gap-2">
                {projectDetails?.project_status && (
                  <span
                    className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
                      projectDetails.project_status.toLowerCase() === "active"
                        ? "bg-green-100 text-green-800"
                        : projectDetails.project_status.toLowerCase() ===
                          "pending"
                        ? "bg-yellow-100 text-yellow-800"
                        : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {projectDetails.project_status}
                  </span>
                )}
                {projectDetails?.project_type && (
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                    {projectDetails.project_type}
                  </span>
                )}
                {isDummyProject && (
                  <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-purple-100 text-purple-800">
                    Demo
                  </span>
                )}
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="p-8">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Project Information */}
              <div className="lg:col-span-2 space-y-6">
                {/* Project Details */}
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                    <div className="w-8 h-8 bg-teal-100 rounded-lg flex items-center justify-center mr-3">
                      <svg
                        className="w-4 h-4 text-teal-600"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                        />
                      </svg>
                    </div>
                    Project Information
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500 mb-1">Project ID</p>
                      <p className="font-mono text-sm font-medium text-gray-900">
                        {projectDetails?.project_id || id}
                      </p>
                    </div>
                    <div className="bg-gray-50 rounded-lg p-4">
                      <p className="text-sm text-gray-500 mb-1">User UUID</p>
                      <p className="font-mono text-sm font-medium text-gray-900">
                        {projectDetails?.user_uuid ||
                          projectData?.user_id ||
                          "N/A"}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Model Information */}
                {modelData && (
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                      <div className="w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center mr-3">
                        <svg
                          className="w-4 h-4 text-blue-600"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
                          />
                        </svg>
                      </div>
                      Model Information
                    </h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-500 mb-1">Model ID</p>
                        <p className="font-mono text-sm font-medium text-gray-900">
                          {modelData.model_id}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-500 mb-1">
                          Model Version
                        </p>
                        <p className="font-mono text-sm font-medium text-gray-900">
                          {modelData.model_version}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-500 mb-1">Dataset ID</p>
                        <p className="font-mono text-sm font-medium text-gray-900">
                          {modelData.dataset_id}
                        </p>
                      </div>
                      <div className="bg-gray-50 rounded-lg p-4">
                        <p className="text-sm text-gray-500 mb-1">Project ID</p>
                        <p className="font-mono text-sm font-medium text-gray-900">
                          {modelData.project_id}
                        </p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Audit Information */}
                {auditData.length > 0 && (
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-4 flex items-center">
                      <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center mr-3">
                        <svg
                          className="w-4 h-4 text-purple-600"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                          />
                        </svg>
                      </div>
                      Recent Audits
                      <span className="ml-2 bg-purple-100 text-purple-800 text-xs font-medium px-2 py-1 rounded-full">
                        {auditData.length}
                      </span>
                    </h3>
                    <div className="space-y-3 max-h-48 overflow-y-auto">
                      {auditData.slice(0, 3).map((audit) => (
                        <div
                          key={audit.id}
                          className="flex items-center justify-between bg-gray-50 rounded-lg p-4"
                        >
                          <div className="flex items-center space-x-3">
                            <div
                              className={`w-3 h-3 rounded-full ${
                                audit.status === "completed"
                                  ? "bg-green-500"
                                  : audit.status === "pending"
                                  ? "bg-yellow-500"
                                  : audit.status === "failed"
                                  ? "bg-red-500"
                                  : "bg-gray-500"
                              }`}
                            ></div>
                            <div>
                              <p className="text-sm font-medium text-gray-900">
                                {audit.audit_type}
                              </p>
                              <p className="text-xs text-gray-500">
                                Model: {audit.model_id}
                              </p>
                            </div>
                          </div>
                          <span
                            className={`text-xs font-medium px-2 py-1 rounded-full ${
                              audit.status === "completed"
                                ? "bg-green-100 text-green-800"
                                : audit.status === "pending"
                                ? "bg-yellow-100 text-yellow-800"
                                : audit.status === "failed"
                                ? "bg-red-100 text-red-800"
                                : "bg-gray-100 text-gray-800"
                            }`}
                          >
                            {audit.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Report Cards Column */}
              <div className="lg:col-span-1">
                {/* AI Risk Assessment Report Card - Show only if generated */}
                {hasRiskAssessment && (
                  <div className="bg-gradient-to-br from-teal-50 to-blue-50 rounded-xl p-6 border border-teal-100 mb-6">
                    <div className="text-center mb-6">
                      <div className="w-16 h-16 bg-gradient-to-br from-teal-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                        <svg
                          className="w-8 h-8 text-white"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                          />
                        </svg>
                      </div>
                      <h3 className="text-xl font-bold text-gray-900 mb-2">
                        AI Risk Assessment Report
                      </h3>
                      <p className="text-sm text-gray-600 mb-2">
                        Generated on{" "}
                        {new Date(
                          riskAssessmentTimestamp || ""
                        ).toLocaleDateString()}
                      </p>
                      <p className="text-sm text-gray-600 mb-4">
                        Download your comprehensive AI risk assessment report
                      </p>
                    </div>

                    {/* Report Features */}
                    <div className="space-y-3 mb-6">
                      <div className="flex items-center text-sm text-gray-700">
                        <svg
                          className="w-4 h-4 text-green-500 mr-3"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                        Risk evaluation & scoring
                      </div>
                      <div className="flex items-center text-sm text-gray-700">
                        <svg
                          className="w-4 h-4 text-green-500 mr-3"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                        Compliance assessment
                      </div>
                      <div className="flex items-center text-sm text-gray-700">
                        <svg
                          className="w-4 h-4 text-green-500 mr-3"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                        Mitigation recommendations
                      </div>
                    </div>

                    {/* Download Button */}
                    <button
                      onClick={() =>
                        handleDownloadReport({
                          id: 999,
                          type: "AI Risk Assessment",
                          category: "Risk",
                          hash: "",
                          size: "",
                          date: riskAssessmentTimestamp || "",
                          metrics: [],
                          blockchainHash: "",
                        })
                      }
                      disabled={
                        reportGeneration.status !== ReportGenerationStatus.IDLE
                      }
                      className="w-full inline-flex items-center justify-center px-6 py-3 border border-transparent rounded-lg shadow-sm text-base font-medium text-white bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-600 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300"
                    >
                      <svg
                        className="w-5 h-5 mr-2"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                        />
                      </svg>
                      Download Risk Assessment
                    </button>
                  </div>
                )}

                {/* Complete Analysis Report Card */}
                <div className="bg-gradient-to-br from-teal-50 to-blue-50 rounded-xl p-6 border border-teal-100">
                  <div className="text-center mb-6">
                    <div className="w-16 h-16 bg-gradient-to-br from-teal-500 to-blue-500 rounded-full flex items-center justify-center mx-auto mb-4">
                      <svg
                        className="w-8 h-8 text-white"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                        />
                      </svg>
                    </div>
                    <h3 className="text-xl font-bold text-gray-900 mb-2">
                      Complete Analysis Report
                    </h3>
                    <p className="text-sm text-gray-600 mb-4">
                      Download comprehensive model performance, fairness, and
                      compliance analysis
                    </p>
                  </div>

                  {/* Report Features */}
                  <div className="space-y-3 mb-6">
                    <div className="flex items-center text-sm text-gray-700">
                      <svg
                        className="w-4 h-4 text-green-500 mr-3"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                      Performance metrics & analytics
                    </div>
                    <div className="flex items-center text-sm text-gray-700">
                      <svg
                        className="w-4 h-4 text-green-500 mr-3"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                      Bias & fairness assessment
                    </div>
                    <div className="flex items-center text-sm text-gray-700">
                      <svg
                        className="w-4 h-4 text-green-500 mr-3"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                      Regulatory compliance check
                    </div>
                    <div className="flex items-center text-sm text-gray-700">
                      <svg
                        className="w-4 h-4 text-green-500 mr-3"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path
                          fillRule="evenodd"
                          d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                          clipRule="evenodd"
                        />
                      </svg>
                      Blockchain verification
                    </div>
                  </div>

                  {/* Download Button */}
                  <button
                    onClick={() => {
                      if (isDummyProject) {
                        alert("Downloading demo report...");
                      } else if (modelData) {
                        const dummyReport = {
                          id: 999,
                          type: "Full Model Report",
                          category: "Performance",
                          hash: "",
                          size: "",
                          date: "",
                          metrics: [],
                          blockchainHash: "",
                        };
                        handleDownloadReport(reports[0] || dummyReport);
                      } else {
                        alert("No model data available to download report");
                      }
                    }}
                    disabled={
                      isLoading ||
                      (!isDummyProject && !modelData) ||
                      reportGeneration.status !== ReportGenerationStatus.IDLE
                    }
                    className="w-full inline-flex items-center justify-center px-6 py-3 border border-transparent rounded-lg shadow-sm text-base font-medium text-white bg-gradient-to-r from-teal-500 to-blue-500 hover:from-teal-600 hover:to-blue-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
                  >
                    {reportGeneration.status !== ReportGenerationStatus.IDLE ? (
                      <>
                        <svg
                          className="-ml-1 mr-3 h-5 w-5 animate-spin"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                        Processing...
                      </>
                    ) : (
                      <>
                        <svg
                          className="-ml-1 mr-3 h-5 w-5"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                          />
                        </svg>
                        Download Report
                      </>
                    )}
                  </button>

                  {/* Status Info */}
                  <div className="mt-4 text-center">
                    <p className="text-xs text-gray-500">
                      {auditData.length > 0 &&
                      auditData.some((a) => a.status === "completed") ? (
                        <span className="text-green-600 font-medium">
                          ✓ Latest audit completed
                        </span>
                      ) : auditData.length > 0 ? (
                        <span className="text-yellow-600 font-medium">
                          ⚠ Audit in progress
                        </span>
                      ) : (
                        <span className="text-gray-500">
                          Ready for download
                        </span>
                      )}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Report Generation Loading Modal */}
        {reportGeneration.status !== ReportGenerationStatus.IDLE && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white rounded-xl shadow-2xl p-8 max-w-md w-full mx-4 transform transition-all relative overflow-hidden">
              {/* Progress bar at the top */}
              <div className="absolute top-0 left-0 right-0 h-1.5 bg-gray-100">
                <div
                  className={`h-full ${
                    reportGeneration.status ===
                    ReportGenerationStatus.GENERATING
                      ? "w-1/2 bg-teal-500 animate-pulse"
                      : reportGeneration.status ===
                        ReportGenerationStatus.DOWNLOADING
                      ? "w-3/4 bg-blue-500 animate-pulse"
                      : reportGeneration.status ===
                        ReportGenerationStatus.COMPLETED
                      ? "w-full bg-green-500"
                      : reportGeneration.status === ReportGenerationStatus.ERROR
                      ? "w-full bg-red-500"
                      : ""
                  }`}
                ></div>
              </div>

              <div className="text-center">
                {/* Different icons based on status */}
                {reportGeneration.status ===
                  ReportGenerationStatus.GENERATING && (
                  <div className="mx-auto mb-6 relative w-24 h-24">
                    {/* Data visualization animation */}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-16 h-16 border-4 border-t-teal-500 border-r-blue-500 border-b-indigo-500 border-l-purple-500 rounded-full animate-spin"></div>
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div
                        className="w-8 h-8 border-4 border-t-teal-300 border-r-transparent border-b-transparent border-l-transparent rounded-full animate-spin"
                        style={{
                          animationDirection: "reverse",
                          animationDuration: "0.8s",
                        }}
                      ></div>
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <svg
                        className="w-10 h-10 text-teal-600 animate-pulse"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={1.5}
                          d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                        />
                      </svg>
                    </div>
                  </div>
                )}

                {reportGeneration.status ===
                  ReportGenerationStatus.DOWNLOADING && (
                  <div className="mx-auto mb-6 relative w-24 h-24">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-16 h-16 border-4 border-blue-500 border-opacity-25 rounded-full animate-pulse"></div>
                    </div>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <svg
                        className="w-10 h-10 text-blue-600"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                        />
                      </svg>
                    </div>

                    {/* Animated dots to show download progress */}
                    <div className="absolute -bottom-1 left-0 right-0 flex justify-center space-x-1">
                      <div
                        className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"
                        style={{ animationDelay: "0s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"
                        style={{ animationDelay: "0.2s" }}
                      ></div>
                      <div
                        className="w-2 h-2 bg-blue-600 rounded-full animate-bounce"
                        style={{ animationDelay: "0.4s" }}
                      ></div>
                    </div>
                  </div>
                )}

                {reportGeneration.status ===
                  ReportGenerationStatus.COMPLETED && (
                  <div className="mx-auto mb-6 relative w-24 h-24">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center animate-scale-in">
                        <svg
                          className="w-10 h-10 text-green-600"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M5 13l4 4L19 7"
                          />
                        </svg>
                      </div>
                    </div>
                  </div>
                )}

                {reportGeneration.status === ReportGenerationStatus.ERROR && (
                  <div className="mx-auto mb-6 relative w-24 h-24">
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center animate-scale-in">
                        <svg
                          className="w-10 h-10 text-red-600"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M6 18L18 6M6 6l12 12"
                          />
                        </svg>
                      </div>
                    </div>
                  </div>
                )}

                <h3 className="text-xl font-semibold text-gray-900 mb-2">
                  {reportGeneration.status === ReportGenerationStatus.GENERATING
                    ? "Generating Report"
                    : reportGeneration.status ===
                      ReportGenerationStatus.DOWNLOADING
                    ? "Downloading Report"
                    : reportGeneration.status ===
                      ReportGenerationStatus.COMPLETED
                    ? "Download Complete"
                    : "Error"}
                </h3>
                <p className="text-gray-600 mb-6">{reportGeneration.message}</p>

                {reportGeneration.status === ReportGenerationStatus.ERROR && (
                  <button
                    onClick={() =>
                      setReportGeneration({
                        status: ReportGenerationStatus.IDLE,
                        message: "",
                      })
                    }
                    className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-colors"
                  >
                    Close
                  </button>
                )}
              </div>

              {/* Small details showing during generation */}
              {reportGeneration.status ===
                ReportGenerationStatus.GENERATING && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-teal-500 rounded-full mr-2"></div>
                      <span>Preparing data</span>
                    </div>
                    <span>✓</span>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-blue-500 rounded-full mr-2 animate-pulse"></div>
                      <span>Calculating metrics</span>
                    </div>
                    <span className="animate-pulse">In progress...</span>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-gray-300 rounded-full mr-2"></div>
                      <span>Generating visualizations</span>
                    </div>
                    <span>Pending</span>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-gray-300 rounded-full mr-2"></div>
                      <span>Finalizing report</span>
                    </div>
                    <span>Pending</span>
                  </div>
                </div>
              )}

              {/* Small details showing during downloading */}
              {reportGeneration.status ===
                ReportGenerationStatus.DOWNLOADING && (
                <div className="mt-4 pt-4 border-t border-gray-200">
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-teal-500 rounded-full mr-2"></div>
                      <span>Report generated</span>
                    </div>
                    <span>✓</span>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-teal-500 rounded-full mr-2"></div>
                      <span>Data verified</span>
                    </div>
                    <span>✓</span>
                  </div>
                  <div className="flex justify-between text-xs text-gray-500 mb-1">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-blue-500 rounded-full mr-2 animate-pulse"></div>
                      <span>Downloading file</span>
                    </div>
                    <div className="flex items-center animate-pulse">
                      <span className="mr-1">Downloading</span>
                      <div className="flex space-x-0.5">
                        <div
                          className="w-1 h-1 bg-blue-500 rounded-full animate-bounce"
                          style={{ animationDelay: "0s" }}
                        ></div>
                        <div
                          className="w-1 h-1 bg-blue-500 rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                        <div
                          className="w-1 h-1 bg-blue-500 rounded-full animate-bounce"
                          style={{ animationDelay: "0.4s" }}
                        ></div>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Loading indicator for page load */}
        {isLoading && (
          <div className="flex justify-center my-8">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-teal-500"></div>
          </div>
        )}

        {/* Reports List or Premium Content Notice */}
        <div className="space-y-6">
          {!isLoading && !isDummyProject && isFreeUser ? null : !isLoading &&
            filteredReports.length > 0 ? (
            filteredReports.map((report) => (
              <div
                key={report.id}
                className="bg-white border border-gray-200 rounded-lg p-6 shadow-md transition-all duration-300 hover:shadow-lg hover:border-teal-200"
              >
                {/* Report Header */}
                <div className="flex flex-col md:flex-row justify-between md:items-start mb-6 gap-4">
                  <div className="flex gap-4 items-center">
                    <div className="text-teal-600 bg-teal-50 p-3 rounded-full">
                      <svg
                        className="h-6 w-6"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                        />
                      </svg>
                    </div>
                    <div>
                      <div className="flex flex-wrap items-center gap-2">
                        <h2 className="text-lg font-medium text-gray-900">
                          {report.type}
                        </h2>
                        <span
                          className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                            report.category === "Performance"
                              ? "bg-blue-100 text-blue-800"
                              : "bg-amber-100 text-amber-800"
                          }`}
                        >
                          {report.category}
                        </span>
                      </div>
                      <p className="text-sm text-gray-500 mt-1">
                        {report.hash} • {report.size} • {report.date}
                      </p>
                    </div>
                  </div>
                  <div className="flex gap-2 ml-auto">
                    <button
                      className="inline-flex items-center p-2 border border-transparent rounded-lg text-gray-500 hover:text-teal-600 hover:bg-teal-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 transition-colors duration-300"
                      title="View"
                    >
                      <svg
                        className="h-5 w-5"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"
                        />
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"
                        />
                      </svg>
                    </button>
                    <button
                      onClick={() => handleDownloadReport(report)}
                      className="inline-flex items-center p-2 border border-transparent rounded-lg text-gray-500 hover:text-teal-600 hover:bg-teal-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 transition-colors duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                      title="Download"
                      disabled={
                        reportGeneration.status !== ReportGenerationStatus.IDLE
                      }
                    >
                      {reportGeneration.status !==
                      ReportGenerationStatus.IDLE ? (
                        <svg
                          className="h-5 w-5 animate-spin"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                      ) : (
                        <svg
                          className="h-5 w-5"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                          />
                        </svg>
                      )}
                    </button>
                    <button
                      className="inline-flex items-center p-2 border border-transparent rounded-lg text-gray-500 hover:text-red-600 hover:bg-red-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 transition-colors duration-300"
                      title="Delete"
                    >
                      <svg
                        className="h-5 w-5"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                        />
                      </svg>
                    </button>
                  </div>
                </div>

                {/* Report Metrics */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-6">
                  {report.metrics.map((metric, i) => (
                    <div
                      key={i}
                      className="bg-gray-50 rounded-lg p-4 shadow-sm hover:shadow transition-shadow duration-300"
                    >
                      <p className="text-sm text-gray-500 mb-1">
                        {metric.name}
                      </p>
                      <p className="text-xl font-medium text-teal-600">
                        {metric.value}
                      </p>
                    </div>
                  ))}
                </div>

                {/* Blockchain Hash */}
                <div className="flex flex-col md:flex-row md:items-center justify-between pt-4 border-t border-gray-200 gap-4">
                  <div className="flex flex-col md:flex-row md:items-center gap-2">
                    <span className="text-sm text-gray-500">
                      Blockchain Hash:
                    </span>
                    <code className="text-sm font-mono bg-gray-50 px-2 py-1 rounded">
                      {report.blockchainHash}
                    </code>
                  </div>
                  <button className="inline-flex items-center text-sm font-medium text-teal-600 hover:text-teal-700 focus:outline-none transition-colors duration-300">
                    <svg
                      className="mr-1.5 h-4 w-4"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                      />
                    </svg>
                    Verify on Blockchain
                  </button>
                </div>
              </div>
            ))
          ) : !isLoading ? (
            <div className="text-center py-10 bg-white border border-gray-200 rounded-lg shadow-md">
              {isDummyProject ? (
                <>
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z"
                    />
                  </svg>
                  <p className="mt-4 text-gray-500">
                    This is a demo project. No additional reports available.
                  </p>
                </>
              ) : (
                <>
                  <svg
                    className="mx-auto h-12 w-12 text-gray-400"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                    />
                  </svg>
                  <p className="mt-4 text-gray-500">
                    No reports found for this project.
                  </p>
                </>
              )}
            </div>
          ) : null}
        </div>
      </div>
      {isRedirecting && <LoadingOverlay />}
    </div>
  );
};

export default ReportPage;
