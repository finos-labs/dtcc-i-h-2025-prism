import React, { useState, useEffect } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import {
  Home,
  BarChart,
  FileText,
  Lightbulb,
  ShieldCheck,
  FileQuestion,
  GitCompare,
  AlertTriangle,
} from "lucide-react";
import { supabase } from "../lib/supabase";

interface SidebarItemProps {
  icon: React.ReactNode;
  label: string;
  to: string;
  active?: boolean;
  hasSubMenu?: boolean;
}

const SidebarItem: React.FC<SidebarItemProps> = ({
  icon,
  label,
  to,
  active = false,
}) => {
  const navigate = useNavigate();

  return (
    <button
      onClick={() => navigate(to)}
      className={`w-full flex items-center px-4 py-3 rounded-lg mb-1 transition-colors ${
        active
          ? "bg-primary text-white font-medium"
          : "text-gray-600 hover:bg-gray-100"
      }`}
    >
      <div className="w-5 h-5 mr-3 flex-shrink-0">{icon}</div>
      <span className="text-sm">{label}</span>
    </button>
  );
};

const AppSidebar: React.FC = () => {
  const location = useLocation();
  const { id } = useParams<{ id?: string }>();
  const isProjectPage = location.pathname.includes("/projects/");
  const [projectType, setProjectType] = useState<"llm" | "generic" | null>(
    null
  );

  useEffect(() => {
    // Fetch project type when on a project page
    if (isProjectPage && id) {
      fetchProjectType(id);
    }
  }, [isProjectPage, id]);

  const fetchProjectType = async (projectId: string) => {
    // Make sure we have a valid project ID
    if (!projectId) {
      console.error("Project ID is undefined");
      return;
    }

    // For dummy projects with specific IDs
    if (projectId === "dummy-1") {
      setProjectType("generic");
      return;
    }
    if (projectId === "dummy-2") {
      setProjectType("llm");
      return;
    }

    try {
      const { data, error } = await supabase
        .from("projectdetails")
        .select("project_type")
        .eq("project_id", projectId)
        .single();

      if (error) throw error;
      if (data) {
        setProjectType(data.project_type);
      }
    } catch (error) {
      console.error("Error fetching project type:", error);
    }
  };

  return (
    <div className="w-56 bg-white border-r border-gray-200 shrink-0  overflow-y-auto">
      <div className="p-4">
        <SidebarItem
          icon={<Home className="w-full h-full" />}
          label="Projects"
          to="/home"
          active={location.pathname === "/home"}
        />

        {isProjectPage && (
          <>
            <div className="px-4 py-2 mt-2 text-xs font-medium text-gray-500 uppercase tracking-wider">
              Project
            </div>

            {/* LLM Project Navigation */}
            {projectType === "llm" && (
              <>
               
                <SidebarItem
                  icon={<ShieldCheck className="w-full h-full" />}
                  label="Redteaming"
                  to={`/projects/${id}/redteaming`}
                  active={location.pathname.includes("/redteaming")}
                />
               
              </>
            )}

            {/* Generic AI Project Navigation */}
            {projectType === "generic" && (
              <>
                <SidebarItem
                  icon={<FileText className="w-full h-full" />}
                  label="Overview"
                  to={`/projects/${id}`}
                  active={location.pathname === `/projects/${id}`}
                />
                
                <div className="px-4 py-2 mt-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Evaluations
                </div>
                
                <SidebarItem
                  icon={<BarChart className="w-full h-full" />}
                  label="Performance"
                  to={`/projects/${id}/performance`}
                  active={location.pathname.includes("/performance")}
                />
                <SidebarItem
                  icon={<ShieldCheck className="w-full h-full" />}
                  label="Fairness"
                  to={`/projects/${id}/fairness`}
                  active={location.pathname.includes("/fairness")}
                />
                <SidebarItem
                  icon={<Lightbulb className="w-full h-full" />}
                  label="Explainability"
                  to={`/projects/${id}/explainability`}
                  active={location.pathname.includes("/explainability")}
                />
                <SidebarItem
                  icon={<GitCompare className="w-full h-full" />}
                  label="Drift Analysis"
                  to={`/projects/${id}/drift`}
                  active={location.pathname.includes("/drift")}
                />
                
                <div className="px-4 py-2 mt-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Risk Assessment
                </div>
                
                <SidebarItem
                  icon={<AlertTriangle className="w-full h-full" />}
                  label="AI Risk Assessment"
                  to={`/projects/${id}/risk-assessment`}
                  active={location.pathname.includes("/risk-assessment")}
                />
                
                <div className="px-4 py-2 mt-4 text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Reports
                </div>
                
                <SidebarItem
                  icon={<FileQuestion className="w-full h-full" />}
                  label="Reports"
                  to={`/projects/${id}/reports`}
                  active={location.pathname.includes("/reports")}
                />
              </>
            )}

          </>
        )}
      </div>
    </div>
  );
};

export default AppSidebar;