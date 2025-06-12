import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { Download } from "lucide-react";
import { Button } from "../components/ui/button";
import { Breadcrumb } from "../components/ui/breadcrumb";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  LabelList,
} from "recharts";
import { useState, useEffect } from "react";
import { InfoTooltip } from "../components/InfoTooltip";
import { supabase } from "../lib/supabase";
// Add API data interface
interface ExplainabilityApiResponse {
  project_id: number;
  model_name: string;
  model_version: string;
  timestamp: string;
  feature_importance: {
    importances: number[];
    feature_names: string[];
    method: string;
  };
  shap_values: number[][][];
  shap_importance: {
    importances: number[][];
    feature_names: string[];
    explainer_type: string;
  };
  lime_explanations: {
    [key: string]: {
      prediction: number;
      feature_importance: [string, number][];
      feature_names: string[];
    };
  };
}

// Add hook to fetch data
const useExplainabilityData = (
  projectId: string,
  modelId: string,
  version: string = "1.0.0"
) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ExplainabilityApiResponse | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);

      try {
        const accessToken = localStorage.getItem("access_token");
        const { data, error } = await supabase
          .from("modeldetails")
          .select("model_id, project_id, dataset_id, model_version")
          .eq("project_id", projectId);

        if (error) {
          throw error;
        }

        const modelId = data[0].model_id;
        const model_version = data[0].model_version;

        const apiUrl = `http://localhost:8000/ml/explainability/${projectId}/${modelId}/${model_version}`;

        const response = await fetch(apiUrl, {
          headers: {
            Authorization: `Bearer ${accessToken}`,
            "Content-Type": "application/json",
          },
        });

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const apiData = await response.json();
        setData(apiData);
      } catch (err) {
        console.error("Failed to fetch explainability data:", err);
        setError(err instanceof Error ? err.message : "Unknown error");
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [projectId, modelId, version]);

  return { data, loading, error };
};

// Adapted data processing functions for API response
const processFeatureImportance = (data: ExplainabilityApiResponse | null) => {
  if (!data) return [];

  // Handle both direct and metrics-wrapped data structures
  const featureData = data.feature_importance || 
                     data.shap_importance || 
                     (data as any).metrics?.feature_importance || 
                     (data as any).metrics?.shap_importance;
  
  // Add safety check for featureData
  if (!featureData || !featureData.importances || !featureData.feature_names) {
    return [];
  }

  const { importances, feature_names } = featureData;

  // Check the structure of importances and handle accordingly
  const meanImportances = feature_names.map((feature: string, idx: number) => {
    // Check if importances[idx] is an array before calling reduce
    const imp = importances[idx];
    const mean = Array.isArray(imp)
      ? imp.reduce((sum, val) => sum + Math.abs(val), 0) / imp.length
      : Math.abs(imp); // Handle if it's a single value

    return {
      feature: feature,
      importance: mean,
      color: "#3182CE",
    };
  });

  return meanImportances.sort((a: any, b: any) => b.importance - a.importance);
};

const processShapDependence = (data: ExplainabilityApiResponse | null) => {
  if (!data) return [];

  // Handle both direct and metrics-wrapped data structures
  const shap_values = data.shap_values || (data as any).metrics?.shap_values;
  const shapImportance = data.shap_importance || (data as any).metrics?.shap_importance;
  
  if (!shap_values || !shapImportance || !shapImportance.feature_names) {
    return [];
  }

  const { feature_names } = shapImportance;

  // For simplicity, we'll just use the first few features
  const result = [];

  // Only process if we have data
  if (shap_values.length > 0 && shap_values[0].length > 0) {
    // Use first 3 features for demo
    const featuresToUse = feature_names.slice(
      0,
      Math.min(3, feature_names.length)
    );

    for (let featureIdx = 0; featureIdx < featuresToUse.length; featureIdx++) {
      for (let i = 0; i < shap_values.length; i++) {
        // Create a normalized feature value (just for demonstration)
        const featureValue = (i / shap_values.length) * 100;

        // Check if shap_values[i][featureIdx] is an array before calling reduce
        const shapValues = shap_values[i][featureIdx];
        let shapValue;

        if (Array.isArray(shapValues)) {
          // Average SHAP values across classes for this feature
          shapValue = shapValues.reduce((a, b) => a + b, 0) / shapValues.length;
        } else {
          // Handle if it's a single value
          shapValue = Number(shapValues);
        }

        result.push({
          featureValue,
          shapValue,
          feature: feature_names[featureIdx],
        });
      }
    }
  }

  return result;
};

const processShapFeatureImportance = (
  data: ExplainabilityApiResponse | null
) => {
  if (!data) return [];

  // Handle both direct and metrics-wrapped data structures
  const shapImportance = data.shap_importance || (data as any).metrics?.shap_importance;
  
  if (!shapImportance || !shapImportance.importances || !shapImportance.feature_names) {
    return [];
  }

  const { importances, feature_names } = shapImportance;

  return feature_names
    .map((feature: string, idx: number) => {
      const imp = importances[idx];

      // Check if imp is an array before calling reduce
      const mean = Array.isArray(imp)
        ? imp.reduce((sum, val) => sum + Math.abs(val), 0) / imp.length
        : Math.abs(imp);

      const stdev = Array.isArray(imp)
        ? Math.sqrt(
            imp.reduce(
              (sum, val) => sum + Math.pow(Math.abs(val) - mean, 2),
              0
            ) / imp.length
          ) * 0.5
        : 0.1; // Default value for single values

      return {
        feature: feature,
        mean,
        stdev,
        color: "#3182CE",
      };
    })
    .sort((a: any, b: any) => b.mean - a.mean);
};

// Demo data for the explainability analysis
const explainabilityData = {
  metrics: {
    interpretability: 87,
    robustness: 92,
    stability: 89,
    status: {
      interpretability: "Good",
      robustness: "Excellent",
      stability: "Good",
    },
  },
  featureImportance: [
    { feature: "Market Volatility", importance: 0.28, color: "#3182CE" },
    { feature: "Debt-Equity Ratio", importance: 0.22, color: "#3182CE" },
    { feature: "Return on Assets", importance: 0.17, color: "#3182CE" },
    { feature: "Earnings Growth", importance: 0.14, color: "#3182CE" },
    { feature: "Price-Earnings Ratio", importance: 0.08, color: "#3182CE" },
    { feature: "Current Ratio", importance: 0.05, color: "#3182CE" },
    { feature: "Dividend Yield", importance: 0.04, color: "#3182CE" },
    { feature: "Market Cap", importance: 0.02, color: "#3182CE" },
  ].sort((a, b) => b.importance - a.importance),
  shapDependence: [
    // Market Volatility
    ...Array.from({ length: 50 }, (_, i) => ({
      featureValue: i * 2,
      shapValue: i * 0.013 - 0.3 + (Math.random() * 0.1 - 0.05),
      feature: "Market Volatility",
    })),
    // Debt-Equity Ratio
    ...Array.from({ length: 50 }, (_, i) => ({
      featureValue: i * 0.1 + 0.5,
      shapValue: -0.25 + i * 0.01 + (Math.random() * 0.1 - 0.05),
      feature: "Debt-Equity Ratio",
    })),
    // Return on Assets
    ...Array.from({ length: 50 }, (_, i) => ({
      featureValue: i * 0.2 + 1,
      shapValue: 0.1 + i * 0.004 + (Math.random() * 0.1 - 0.05),
      feature: "Return on Assets",
    })),
  ],
  shapFeatureImportance: [
    { feature: "Market Volatility", mean: 0.32, stdev: 0.15, color: "#3182CE" },
    { feature: "Debt-Equity Ratio", mean: 0.26, stdev: 0.12, color: "#3182CE" },
    { feature: "Return on Assets", mean: 0.18, stdev: 0.08, color: "#3182CE" },
    { feature: "Earnings Growth", mean: 0.13, stdev: 0.06, color: "#3182CE" },
    {
      feature: "Price-Earnings Ratio",
      mean: 0.06,
      stdev: 0.03,
      color: "#3182CE",
    },
    { feature: "Current Ratio", mean: 0.03, stdev: 0.02, color: "#3182CE" },
    { feature: "Dividend Yield", mean: 0.02, stdev: 0.01, color: "#3182CE" },
  ].sort((a, b) => b.mean - a.mean),
};

const MetricCard = ({
  title,
  value,
  status,
  description,
}: {
  title: string;
  value: number;
  status: string;
  description: string;
}) => (
  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
    <div className="flex justify-between items-center">
      <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
      <InfoTooltip title={title} entityType="metric" entityName={title} />
    </div>
    <div className="flex items-baseline space-x-2">
      <span className="text-4xl font-bold text-gray-900">{value}%</span>
      <span
        className={`text-sm font-medium px-2.5 py-0.5 rounded-full ${
          status === "Excellent"
            ? "bg-green-100 text-green-800"
            : status === "Good"
            ? "bg-blue-100 text-blue-800"
            : "bg-yellow-100 text-yellow-800"
        }`}
      >
        {status}
      </span>
    </div>
    <p className="mt-2 text-sm text-gray-500">{description}</p>
  </div>
);

// Feature Importance Chart
const FeatureImportanceChart = ({ data }: { data: any[] }) => (
  <div className="w-full h-[300px]">
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        layout="vertical"
        data={data}
        margin={{ top: 10, right: 45, left: 100, bottom: 10 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          opacity={0.3}
          horizontal={true}
          vertical={false}
        />
        <XAxis
          type="number"
          domain={[0, 'dataMax']}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fontSize: 12 }}
          width={90}
        />
        <Tooltip
          formatter={(value: number) => [
            `${(value * 100).toFixed(0)}%`,
            "Importance",
          ]}
          contentStyle={{
            backgroundColor: "rgba(255, 255, 255, 0.95)",
            borderRadius: "8px",
            boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
            border: "1px solid #E5E7EB",
          }}
        />
        <Bar
          dataKey="importance"
          fill="#3182CE"
          animationDuration={1500}
          radius={[0, 4, 4, 0]}
        >
          <LabelList
            dataKey="importance"
            position="right"
            formatter={(value: number) => `${(value * 100).toFixed(0)}%`}
            style={{ fontSize: '11px' }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>
);

// SHAP Dependence Plot
const ShapDependencePlot = ({ data }: { data: any[] }) => {
  // Get unique feature names from the data
  const uniqueFeatures = Array.from(new Set(data.map((d) => d.feature)));
  const feature1 = uniqueFeatures[0] || "Market Volatility";
  const feature2 = uniqueFeatures[1] || "Debt-Equity Ratio";
  const feature3 = uniqueFeatures[2] || "Return on Assets";

  const feature1Data = data.filter((d) => d.feature === feature1);
  const feature2Data = data.filter((d) => d.feature === feature2);
  const feature3Data = data.filter((d) => d.feature === feature3);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="bg-white rounded-lg p-3 h-[250px]">
        <div className="flex items-center mb-1">
          <h3 className="text-sm font-medium text-gray-700">{feature1}</h3>
        </div>
        <ResponsiveContainer width="100%" height="90%">
          <ScatterChart margin={{ top: 5, right: 10, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              type="number"
              dataKey="featureValue"
              name={feature1}
              label={{ value: "Value", position: "insideBottom", offset: -5, fontSize: 11 }}
              tick={{ fontSize: 10 }}
            />
            <YAxis
              type="number"
              dataKey="shapValue"
              name="SHAP Value"
              label={{
                value: "SHAP Impact",
                angle: -90,
                position: "insideLeft",
                fontSize: 11,
                offset: 5
              }}
              tick={{ fontSize: 10 }}
            />
            <Tooltip
              formatter={(value: number) => [value.toFixed(3), ""]}
              contentStyle={{
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                borderRadius: "8px",
                boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
                border: "1px solid #E5E7EB",
                fontSize: "11px"
              }}
            />
            <Scatter
              data={feature1Data}
              fill="#3182CE"
              animationDuration={1500}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white rounded-lg p-3 h-[250px]">
        <div className="flex items-center mb-1">
          <h3 className="text-sm font-medium text-gray-700">{feature2}</h3>
        </div>
        <ResponsiveContainer width="100%" height="90%">
          <ScatterChart margin={{ top: 5, right: 10, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              type="number"
              dataKey="featureValue"
              name={feature2}
              label={{ value: "Value", position: "insideBottom", offset: -5, fontSize: 11 }}
              tick={{ fontSize: 10 }}
            />
            <YAxis
              type="number"
              dataKey="shapValue"
              name="SHAP Value"
              label={{
                value: "SHAP Impact",
                angle: -90,
                position: "insideLeft",
                fontSize: 11,
                offset: 5
              }}
              tick={{ fontSize: 10 }}
            />
            <Tooltip
              formatter={(value: number) => [value.toFixed(3), ""]}
              contentStyle={{
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                borderRadius: "8px",
                boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
                border: "1px solid #E5E7EB",
                fontSize: "11px"
              }}
            />
            <Scatter
              data={feature2Data}
              fill="#3182CE"
              animationDuration={1500}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-white rounded-lg p-3 h-[250px]">
        <div className="flex items-center mb-1">
          <h3 className="text-sm font-medium text-gray-700">{feature3}</h3>
        </div>
        <ResponsiveContainer width="100%" height="90%">
          <ScatterChart margin={{ top: 5, right: 10, bottom: 30, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              type="number"
              dataKey="featureValue"
              name={feature3}
              label={{ value: "Value", position: "insideBottom", offset: -5, fontSize: 11 }}
              tick={{ fontSize: 10 }}
            />
            <YAxis
              type="number"
              dataKey="shapValue"
              name="SHAP Value"
              label={{
                value: "SHAP Impact",
                angle: -90,
                position: "insideLeft",
                fontSize: 11,
                offset: 5
              }}
              tick={{ fontSize: 10 }}
            />
            <Tooltip
              formatter={(value: number) => [value.toFixed(3), ""]}
              contentStyle={{
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                borderRadius: "8px",
                boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
                border: "1px solid #E5E7EB",
                fontSize: "11px"
              }}
            />
            <Scatter
              data={feature3Data}
              fill="#3182CE"
              animationDuration={1500}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// SHAP Feature Importance
const ShapFeatureImportanceChart = ({ data }: { data: any[] }) => (
  <div className="w-full h-[300px]">
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        layout="vertical"
        data={data}
        margin={{ top: 10, right: 45, left: 100, bottom: 10 }}
      >
        <CartesianGrid
          strokeDasharray="3 3"
          opacity={0.3}
          horizontal={true}
          vertical={false}
        />
        <XAxis
          type="number"
          domain={[0, 'dataMax']}
          tickFormatter={(value) => `${(value * 100).toFixed(0)}%`}
        />
        <YAxis
          type="category"
          dataKey="feature"
          tick={{ fontSize: 12 }}
          width={90}
        />
        <Tooltip
          formatter={(value: number) => [
            `${(value * 100).toFixed(0)}%`,
            "SHAP Value",
          ]}
          contentStyle={{
            backgroundColor: "rgba(255, 255, 255, 0.95)",
            borderRadius: "8px",
            boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
            border: "1px solid #E5E7EB",
          }}
        />
        <Bar
          dataKey="mean"
          fill="#3182CE"
          animationDuration={1500}
          radius={[0, 4, 4, 0]}
        >
          <LabelList
            dataKey="mean"
            position="right"
            formatter={(value: number) => `${(value * 100).toFixed(0)}%`}
            style={{ fontSize: '11px' }}
          />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>
);

const ExplainabilityPage: React.FC = () => {
  const { id, modelId } = useParams<{ id: string; modelId: string }>();
  const isDummyProject = id === "dummy-1" || id === "dummy-2";

  // Set default modelId for dummy project
  const effectiveModelId = isDummyProject ? "demo-model" : modelId;

  // Always create the hook with empty values for dummy projects
  const { data, loading, error } = useExplainabilityData(
    id || "",
    effectiveModelId || ""
  );

  // Always use demo data for dummy projects, otherwise fallback to demo data if API fails
  const featureImportanceData = isDummyProject
    ? explainabilityData.featureImportance
    : data
    ? processFeatureImportance(data)
    : explainabilityData.featureImportance;

  const shapDependenceData = isDummyProject
    ? explainabilityData.shapDependence
    : data
    ? processShapDependence(data)
    : explainabilityData.shapDependence;

  const shapFeatureImportanceData = isDummyProject
    ? explainabilityData.shapFeatureImportance
    : data
    ? processShapFeatureImportance(data)
    : explainabilityData.shapFeatureImportance;

  const breadcrumbSegments = [
    { title: "Projects", href: "/home" },
    { title: "Investment Portfolio Analysis", href: `/projects/${id}` },
    { title: "Explainability", href: `/projects/${id}/explainability` },
  ];

  // If loading, show loading state
  if (loading) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-700">
            Loading explainability data...
          </h2>
        </div>
      </div>
    );
  }

  // If error or no data for non-dummy project, show premium empty state
  if ((error || !data) && !isDummyProject) {
    return (
      <div className="flex-1 p-8">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="mb-10">
            <h1 className="text-3xl font-bold text-gray-900">
              Model Explainability
            </h1>
            <p className="mt-2 text-gray-600">
              Understand how your model makes decisions and interprets data
            </p>
          </div>

          {/* Premium Empty State */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-md overflow-hidden">
            <div className="p-8 text-center">
              <div className="mx-auto w-20 h-20 bg-indigo-50 rounded-full flex items-center justify-center mb-6">
                <svg
                  className="h-10 w-10 text-indigo-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </div>

              <h2 className="text-2xl font-bold text-gray-900 mb-3">
                Explainability Analysis Not Available
              </h2>
              <p className="text-gray-600 max-w-2xl mx-auto mb-6">
                {error
                  ? `Error loading data: ${error}`
                  : "No model has been uploaded yet for this project. Please upload a model in the Project Overview page to begin analyzing how your model makes decisions."}
              </p>

              <button
                onClick={() => window.history.back()}
                className="inline-flex items-center px-6 py-3 border border-transparent rounded-lg shadow-sm text-base font-medium text-white bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-700 hover:to-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 transition-colors duration-200"
              >
                <svg
                  className="mr-3 h-5 w-5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M11 15l-3-3m0 0l3-3m-3 3h8M3 12a9 9 0 1118 0 9 9 0 01-18 0z"
                  />
                </svg>
                Return to Project Overview
              </button>
            </div>

            {/* Feature explanations section */}
            <div className="bg-gradient-to-r from-gray-50 to-indigo-50 p-6 border-t border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Explainability Features
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-white p-5 rounded-lg border border-gray-100 shadow-sm">
                  <div className="w-12 h-12 bg-orange-50 rounded-lg flex items-center justify-center mb-4">
                    <svg
                      className="h-6 w-6 text-orange-600"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6"
                      />
                    </svg>
                  </div>
                  <h4 className="text-base font-medium text-gray-900 mb-2">
                    Feature Importance
                  </h4>
                  <p className="text-gray-600 text-sm">
                    Identify which features have the greatest impact on your
                    model's predictions.
                  </p>
                </div>

                <div className="bg-white p-5 rounded-lg border border-gray-100 shadow-sm">
                  <div className="w-12 h-12 bg-green-50 rounded-lg flex items-center justify-center mb-4">
                    <svg
                      className="h-6 w-6 text-green-600"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                      />
                    </svg>
                  </div>
                  <h4 className="text-base font-medium text-gray-900 mb-2">
                    Local Explanations
                  </h4>
                  <p className="text-gray-600 text-sm">
                    Understand specific predictions with instance-level
                    explanation techniques.
                  </p>
                </div>

                <div className="bg-white p-5 rounded-lg border border-gray-100 shadow-sm">
                  <div className="w-12 h-12 bg-red-50 rounded-lg flex items-center justify-center mb-4">
                    <svg
                      className="h-6 w-6 text-red-600"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
                      />
                    </svg>
                  </div>
                  <h4 className="text-base font-medium text-gray-900 mb-2">
                    Concept Visualization
                  </h4>
                  <p className="text-gray-600 text-sm">
                    Visualize how your model interprets abstract concepts and
                    represents knowledge.
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Additional information section */}
          <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-indigo-100 rounded-full flex items-center justify-center mr-3">
                  <svg
                    className="h-5 w-5 text-indigo-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-900">
                  Why Explainability Matters
                </h3>
              </div>
              <p className="text-gray-700 mb-4">
                AI explainability is becoming increasingly important for
                regulatory compliance, building user trust, and debugging model
                behavior. Explainable AI helps you:
              </p>
              <ul className="space-y-2">
                <li className="flex items-start">
                  <svg
                    className="h-5 w-5 text-indigo-500 mr-2 mt-0.5"
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
                  <span className="text-gray-600">
                    Meet regulatory requirements for high-risk AI systems
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="h-5 w-5 text-indigo-500 mr-2 mt-0.5"
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
                  <span className="text-gray-600">
                    Identify and mitigate unintended behaviors
                  </span>
                </li>
                <li className="flex items-start">
                  <svg
                    className="h-5 w-5 text-indigo-500 mr-2 mt-0.5"
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
                  <span className="text-gray-600">
                    Build stakeholder trust and confidence in your AI
                  </span>
                </li>
              </ul>
            </div>

            <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
              <div className="flex items-center mb-4">
                <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center mr-3">
                  <svg
                    className="h-5 w-5 text-purple-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
                    />
                  </svg>
                </div>
                <h3 className="text-lg font-semibold text-gray-900">
                  Explainability Methods
                </h3>
              </div>
              <div className="space-y-3">
                <div className="bg-gray-50 p-3 rounded border border-gray-100">
                  <h4 className="text-sm font-medium text-gray-900 mb-1">
                    SHAP (SHapley Additive exPlanations)
                  </h4>
                  <p className="text-xs text-gray-600">
                    Provides feature importance values for each prediction using
                    game theory concepts.
                  </p>
                </div>
                <div className="bg-gray-50 p-3 rounded border border-gray-100">
                  <h4 className="text-sm font-medium text-gray-900 mb-1">
                    LIME (Local Interpretable Model-agnostic Explanations)
                  </h4>
                  <p className="text-xs text-gray-600">
                    Creates locally faithful approximations to explain
                    individual predictions.
                  </p>
                </div>
                <div className="bg-gray-50 p-3 rounded border border-gray-100">
                  <h4 className="text-sm font-medium text-gray-900 mb-1">
                    Integrated Gradients
                  </h4>
                  <p className="text-xs text-gray-600">
                    Assigns importance scores to features by integrating
                    gradients along a path.
                  </p>
                </div>
                <div className="mt-4 text-center">
                  <span className="text-sm text-indigo-600 font-medium">
                    Upload your model to access these explainability tools
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // If we have data or it's a dummy project, show the visualization
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="p-8 space-y-8 bg-gray-50 min-h-screen"
    >
      
      <div>
        <h1 className="text-3xl font-bold text-gray-900">
          Explainability Analysis
        </h1>
        <p className="text-gray-500 mt-1">
          {data
            ? `Model: ${data.model_name} (v${data.model_version})`
            : isDummyProject
            ? "Demo Explainability Analysis"
            : "Understanding model decisions and feature importance"}
        </p>
        {error && (
          <p className="mt-2 text-sm text-red-500">
          
          </p>
        )}
      </div>



      {/* Feature Importance */}
      <motion.div
        whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
        transition={{ duration: 0.3 }}
        className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
      >
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <h2 className="text-xl font-semibold text-gray-900">
              Feature Importance
            </h2>
          </div>
          <InfoTooltip
            title="About Feature Importance"
            entityType="chart"
            entityName="Feature Importance"
          />
        </div>
        <FeatureImportanceChart data={featureImportanceData} />
      </motion.div>

      {/* SHAP Dependence Plot */}
      <motion.div
        whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
        transition={{ duration: 0.3 }}
        className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
      >
        <div className="flex justify-between items-center mb-3">
          <div className="flex items-center">
            <h2 className="text-xl font-semibold text-gray-900">
              SHAP Dependence Plot
            </h2>
          </div>
          <InfoTooltip
            title="About SHAP Dependence Plot"
            entityType="chart"
            entityName="SHAP Dependence Plot"
          />
        </div>
        <ShapDependencePlot data={shapDependenceData} />
      </motion.div>

      {/* SHAP Feature Importance */}
      <motion.div
        whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
        transition={{ duration: 0.3 }}
        className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
      >
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <h2 className="text-xl font-semibold text-gray-900">
              SHAP Feature Importance
            </h2>
          </div>
          <InfoTooltip
            title="About SHAP Feature Importance"
            entityType="chart"
            entityName="SHAP Feature Importance"
          />
        </div>
        <ShapFeatureImportanceChart data={shapFeatureImportanceData} />
      </motion.div>

      {/* LIME Explanations */}
      <motion.div
        whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
        transition={{ duration: 0.3 }}
        className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
      >
        <div className="flex justify-between items-center mb-4">
          <div className="flex items-center">
            <h2 className="text-xl font-semibold text-gray-900">
              LIME Instance Explanations
            </h2>
          </div>
          <InfoTooltip
            title="About LIME Explanations"
            entityType="chart"
            entityName="LIME Explanations"
          />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {data && (data.lime_explanations || (data as any).metrics?.lime_explanations) ? (
            Object.entries(data.lime_explanations || (data as any).metrics?.lime_explanations || {}).slice(0, 4).map(([instanceKey, instance]) => (
              <div key={instanceKey} className="bg-gray-50 p-4 rounded-lg h-[220px]">
                <h3 className="text-md font-medium mb-2">{instanceKey.replace('_', ' ')}</h3>
                <p className="text-sm text-gray-600 mb-3">
                  Prediction: {(instance.prediction * 100).toFixed(1)}%
                </p>
                <div className="space-y-3">
                  {instance.feature_importance.slice(0, 3).map(([feature, importance], idx) => {
                    const isPositive = importance > 0;
                    const absImportance = Math.abs(importance);
                    const maxWidth = 60; // Maximum width in pixels
                    const barWidth = Math.max(10, absImportance * maxWidth * 10); // Scale for visibility
                    
                    return (
                      <div key={idx} className="flex justify-between items-center">
                        <span className="text-sm truncate flex-1 mr-2">{feature}</span>
                        <div className="flex items-center">
                          <div
                            className={`h-2 rounded-full ${
                              isPositive ? "bg-green-500" : "bg-red-500"
                            }`}
                            style={{ width: `${Math.min(barWidth, maxWidth)}px` }}
                          ></div>
                          <span className={`text-xs ml-2 ${
                            isPositive ? "text-green-500" : "text-red-500"
                          }`}>
                            {importance.toFixed(3)}
                          </span>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
          ) : (
            // Fallback display for dummy projects or when no LIME data is available
            <>
              <div className="bg-gray-50 p-4 rounded-lg h-[220px]">
                <h3 className="text-md font-medium mb-2">Instance 0</h3>
                <p className="text-sm text-gray-600 mb-3">Prediction: 49.65%</p>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">age &lt;= -0.84</span>
                    <div className="flex items-center">
                      <div className="h-2 bg-red-500 rounded-full" style={{ width: "30px" }}></div>
                      <span className="text-xs ml-2 text-red-500">-0.01</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">-1.00 &lt; purchased &lt;= 1.00</span>
                    <div className="flex items-center">
                      <div className="h-2 bg-red-500 rounded-full" style={{ width: "15px" }}></div>
                      <span className="text-xs ml-2 text-red-500">-0.00</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">income &lt;= -0.78</span>
                    <div className="flex items-center">
                      <div className="h-2 bg-red-500 rounded-full" style={{ width: "10px" }}></div>
                      <span className="text-xs ml-2 text-red-500">-0.00</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gray-50 p-4 rounded-lg h-[220px]">
                <h3 className="text-md font-medium mb-2">Instance 1</h3>
                <p className="text-sm text-gray-600 mb-3">Prediction: 49.54%</p>
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-sm">age &lt;= -0.84</span>
                    <div className="flex items-center">
                      <div className="h-2 bg-red-500 rounded-full" style={{ width: "30px" }}></div>
                      <span className="text-xs ml-2 text-red-500">-0.01</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">purchased &lt;= -1.00</span>
                    <div className="flex items-center">
                      <div className="h-2 bg-green-500 rounded-full" style={{ width: "15px" }}></div>
                      <span className="text-xs ml-2 text-green-500">0.00</span>
                    </div>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-sm">0.13 &lt; income &lt;= 0.87</span>
                    <div className="flex items-center">
                      <div className="h-2 bg-green-500 rounded-full" style={{ width: "10px" }}></div>
                      <span className="text-xs ml-2 text-green-500">0.00</span>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </motion.div>

      {/* Only show the overview section if we don't have API data */}
      {!data && (
        <div className="mt-8">
          <h3 className="text-2xl font-bold text-gray-900 mb-4">
            Explainability Overview
          </h3>
          <p className="text-gray-600 mb-8">
            Identify which features have the greatest impact on your model's
            predictions and outcomes.
          </p>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
              <div className="w-12 h-12 bg-green-50 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="h-6 w-6 text-green-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4"
                  />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">
                Local Explanations
              </h4>
              <p className="text-gray-600">
                Understand specific predictions with instance-level explanation
                techniques like LIME and SHAP.
              </p>
            </div>

            <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
              <div className="w-12 h-12 bg-red-50 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="h-6 w-6 text-red-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4"
                  />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">
                Concept Visualization
              </h4>
              <p className="text-gray-600">
                Visualize how your model interprets abstract concepts and
                represents knowledge internally.
              </p>
            </div>

            <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
              <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
                <svg
                  className="h-6 w-6 text-blue-600"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={1.5}
                    d="M9 17v-2m3 2v-4m3 4v-6m2 10H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
              </div>
              <h4 className="text-lg font-medium text-gray-900 mb-2">
                Model Transparency
              </h4>
              <p className="text-gray-600">
                Gain insights into your model's decision-making process to build
                trust and meet regulatory requirements.
              </p>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default ExplainabilityPage;
