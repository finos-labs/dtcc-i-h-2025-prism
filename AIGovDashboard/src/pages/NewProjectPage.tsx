import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { useNavigate } from "react-router-dom";
import { Card, CardContent } from "../components/ui/card";
import { Input } from "../components/ui/input";
import { Button } from "../components/ui/button";
import { Breadcrumb } from "../components/ui/breadcrumb";
import { Boxes, ArrowRight, Bot, Activity } from "lucide-react";
import { supabase } from "../lib/supabase";

const projectTypes = [
 
  {
    id: "generic",
    name: "Supervised machine learning model",
    icon: Activity,
  },
];

export default function NewProjectPage() {
  const navigate = useNavigate();
  const [projectType, setProjectType] = useState("");
  const [projectName, setProjectName] = useState("");
  const [description, setDescription] = useState("");
  const [isCreating, setIsCreating] = useState(false);
  const [error, setError] = useState("");

  // Check for user authentication
  useEffect(() => {
    const userId = localStorage.getItem("userId");
    if (!userId) {
      navigate("/login");
    }
  }, [navigate]);

  const breadcrumbSegments = [
    { title: "Projects", href: "/home" },
    { title: "New Project", href: "/projects/new" },
  ];

  const handleCreateProject = async () => {
    const userId = localStorage.getItem("userId");
    if (!userId) {
      setError("You must be logged in to create a project");
      navigate("/login");
      return;
    }

    setIsCreating(true);
    setError("");

    try {
      // Map project type to API expected format
      const mappedProjectType =
        projectType === "generic" ? "ML" : projectType.toUpperCase();
      console.log(localStorage.getItem("access_token"));
      // API call to localhost
      const response = await fetch(
        "http://localhost:8000/projects/create",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${localStorage.getItem("access_token")}`,
          },
          body: JSON.stringify({
            name: projectName,
            description: description,
            project_type: mappedProjectType,
            status: "NotStarted",
          }),
        }
      );

      if (!response.ok) {
        throw new Error("Failed to create project");
      }

      const data = await response.json();

      console.log(data.id);
      // Also create entry in Supabase
      const { error: supabaseError } = await supabase
        .from("projectdetails")
        .insert({
          project_id: data.id, // Assuming the API returns an id
          project_name: projectName,
          description,
          project_type: projectType,
          project_status: "active",
          user_uuid: userId,
        });

      if (supabaseError) throw supabaseError;

      // Increment the project count in userData
      // First, get the current userData
      const { data: userData, error: fetchError } = await supabase
        .from("userData")
        .select("project_count")
        .eq("user_id", userId)
        .single();

      if (fetchError && fetchError.code !== "PGRST116") {
        console.error("Error fetching userData:", fetchError);
      } else {
        // Calculate new count (default to 1 if no record exists)
        const currentCount = userData?.project_count || 0;
        const newCount = currentCount + 1;

        console.log(
          `Updating project count from ${currentCount} to ${newCount}`
        );

        // Update the project count
        const { error: updateError } = await supabase.from("userData").upsert(
          {
            user_id: userId,
            project_count: newCount,
            // Don't set other fields so we don't overwrite them
          },
          {
            onConflict: "user_id",
          }
        );

        if (updateError) {
          console.error("Error updating project count:", updateError);
        } else {
          console.log("Project count updated successfully");
        }
      }

      navigate(`/home`);
    } catch (error: unknown) {
      console.error("Error creating project:", error);
      setError(
        error instanceof Error
          ? error.message
          : "Failed to create project. Please try again."
      );
    } finally {
      setIsCreating(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="p-8 space-y-8"
    >
      <Breadcrumb segments={breadcrumbSegments} />

      <div>
        <h1 className="text-4xl font-bold tracking-tight">
          Create New Project
        </h1>
        <p className="mt-2 text-lg text-gray-500">
          Start a new project to test an AI Model and generate reports
        </p>
      </div>

      <Card className="bg-white shadow-lg w-full">
        <CardContent className="p-6 space-y-6">
          <div className="flex items-center space-x-3 mb-6">
            <div className="rounded-lg bg-[#59A9A9]/5 p-3">
              <Boxes className="h-6 w-6 text-[#59A9A9]" />
            </div>
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Project Details
              </h2>
              <p className="text-sm text-gray-500">
                Enter basic information about your project
              </p>
            </div>
          </div>

          <div className="space-y-6">
            <div className="space-y-4">
              <label className="text-sm font-medium text-gray-900">
                Project Type
              </label>
              <div className="grid grid-cols-2 gap-4">
                {projectTypes.map((type) => (
                  <div
                    key={type.id}
                    className={`flex cursor-pointer items-center space-x-3 rounded-lg border p-4 transition-all ${
                      projectType === type.id
                        ? "border-[#59A9A9] bg-[#59A9A9]/5 shadow-sm"
                        : "border-gray-100 hover:border-[#59A9A9]/30 hover:bg-[#59A9A9]/5"
                    }`}
                    onClick={() => setProjectType(type.id)}
                  >
                    <div className="rounded-full bg-[#59A9A9]/5 p-2.5">
                      <type.icon className="h-5 w-5 text-[#59A9A9]" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900 text-sm">
                        {type.name}
                      </h3>
                     
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="space-y-3 pt-2">
              <label className="text-sm font-medium text-gray-900">
                Project Name
              </label>
              <Input
                value={projectName}
                onChange={(e) => setProjectName(e.target.value)}
                placeholder="Enter project name"
                className="bg-white border-gray-200 p-6 h-12 rounded-lg"
              />
            </div>

            <div className="space-y-3 pt-2">
              <label className="text-sm font-medium text-gray-900">
                Description
              </label>
              <Input
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                placeholder="Brief description of your project"
                className="bg-white border-gray-200 p-6 h-12 rounded-lg"
              />
            </div>

            {error && <div className="text-red-500 text-sm mt-2">{error}</div>}
          </div>

          <div className="pt-4">
            <Button
              className="w-full bg-[#59A9A9] hover:bg-[#59A9A9]/90 text-white transition-colors py-4 text-base font-medium rounded-lg shadow-sm hover:shadow-md flex items-center justify-center"
              onClick={handleCreateProject}
              disabled={!projectType || !projectName || isCreating}
            >
              {isCreating ? (
                <span className="flex items-center">Creating Project...</span>
              ) : (
                <span className="flex items-center">
                  Continue to Setup
                  <ArrowRight className="ml-2 h-4 w-4" />
                </span>
              )}
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}