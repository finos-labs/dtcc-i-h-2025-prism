import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Button } from "../components/ui/button";
import { FileUp as FileUpload2 } from 'lucide-react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Cell
} from 'recharts';

// Define interface for actual API response
interface FairnessApiResponse {
  project_id: number;
  model_name: string;
  model_version: string;
  timestamp: string;
  metrics: {
    dataset_info: {
      available_features: string[];
      detected_sensitive_features: string[];
      sample_size: number;
      total_features: number;
    };
    interpretation: any;
    metrics: {
      [feature: string]: {
        demographic_parity: { [key: string]: number };
        equal_opportunity: { [key: string]: number };
        equalized_odds: { [key: string]: any };
        disparate_impact: { [key: string]: number };
        treatment_equality: { [key: string]: any };
        statistical_parity: { [key: string]: number };
        statistical_tests: { [key: string]: any };
        interpretation: {
          demographic_parity_threshold: number;
          equal_opportunity_threshold: number;
          disparate_impact_threshold: number;
          statistical_parity_threshold: number;
        };
      }
    };
    processing_timestamp: string;
    sensitive_features: string[];
    statistical_tests: any;
    status: string;
  };
}

const UploadModal = () => (
  <div className="bg-gradient-to-br from-white to-gray-50 rounded-xl p-12 shadow-xl border border-gray-100">
    <div className="text-center max-w-2xl mx-auto">
      <div className="bg-primary/5 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6">
        <FileUpload2 className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-2xl font-bold text-gray-900 mb-3">Upload Your Model for Fairness Analysis</h2>
      <p className="mt-2 text-gray-600">
        Upload your trained model to assess fairness metrics across different demographic groups.
      </p>
      <Button className="bg-primary text-white hover:bg-primary/90 px-8 py-6 text-lg shadow-lg">
        Upload Model
      </Button>
      <p className="mt-4 text-sm text-gray-500">
        Supported formats: .h5, .pkl, .pt, .onnx
      </p>
    </div>
  </div>
);

const FairnessPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isDummyProject = id === "dummy-1" || id === "dummy-2";
  const [hasAnalysis, setHasAnalysis] = useState(isDummyProject);
  const [loading, setLoading] = useState(!isDummyProject);
  const [fairnessAPIData, setFairnessAPIData] = useState<FairnessApiResponse | null>(null);

  useEffect(() => {
    if (!isDummyProject) {
      setLoading(true);
      // Make API call to fetch fairness data
      const fetchFairnessData = async () => {
        const projectId = id;

        try {
          const { data: modelData, error: modelError } = await supabase
            .from("modeldetails")
            .select("model_id, project_id, dataset_id, model_version")
            .eq("project_id", projectId);

          if (modelError) {
            throw modelError;
          }

          if (!modelData || modelData.length === 0) {
            throw new Error("No model found for this project");
          }

          // Get authentication token
          const accessToken = localStorage.getItem("access_token");

          const modelId = modelData[0].model_id;
          const model_version = modelData[0].model_version;
            
          const response = await axios.get(`https://prism-backend-dtcc-dot-block-convey-p1.uc.r.appspot.com/ml/fairness/${id}/${modelId}/${model_version}`, {
            headers: {
              'Authorization': `Bearer ${accessToken}`,
              'Content-Type': 'application/json',
            }
          });

          console.log(response.data); 
          
          // Detailed metrics logging
          console.log('========== Fairness Metrics Analysis ==========');
          
          // Log metrics for each sensitive feature
          if (response.data.metrics) {
            Object.entries(response.data.metrics).forEach(([feature, metrics]: [string, any]) => {
              console.log(`\n=== Metrics for ${feature} ===`);
              console.log('Demographic Parity:', metrics.demographic_parity);
              console.log('Equal Opportunity:', metrics.equal_opportunity);
              console.log('Equalized Odds:', metrics.equalized_odds);
              console.log('Disparate Impact:', metrics.disparate_impact);
              console.log('Treatment Equality:', metrics.treatment_equality);
              console.log('Statistical Parity:', metrics.statistical_parity);
              
              if (metrics.interpretation) {
                console.log('\nInterpretation Thresholds:');
                console.log('Demographic Parity Threshold:', metrics.interpretation.demographic_parity_threshold);
                console.log('Equal Opportunity Threshold:', metrics.interpretation.equal_opportunity_threshold);
                console.log('Disparate Impact Threshold:', metrics.interpretation.disparate_impact_threshold);
                console.log('Statistical Parity Threshold:', metrics.interpretation.statistical_parity_threshold);
              }
              
              if (metrics.statistical_tests) {
                console.log('\nStatistical Tests:');
                console.log(metrics.statistical_tests);
              }
            });
          }
          
          if (response.data.interpretation) {
            console.log('\n=== Overall Interpretation ===');
            console.log(response.data.interpretation);
          }
          
          console.log('\n==========================================');
          
          setFairnessAPIData(response.data);
          setHasAnalysis(true);
          setLoading(false);
        } catch (error) {
          console.error("Error fetching fairness data:", error);
          setHasAnalysis(false);
          setLoading(false);
        }
      };

      fetchFairnessData();
    }
  }, [id, isDummyProject]);

  if (loading) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading fairness analysis...</p>
        </div>
      </div>
    );
  }

  if (!hasAnalysis) {
    return (
      <div className="flex-1 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-10">
            <h1 className="text-3xl font-bold text-gray-900">
              Fairness Assessment
            </h1>
            <p className="mt-2 text-gray-600">
              Evaluate your model for bias and discrimination across
              demographics
            </p>
          </div>

          <UploadModal />

          <div className="mt-12">
            <h3 className="text-xl font-semibold text-gray-900 mb-6">
              Fairness Assessment Features
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                <div className="w-12 h-12 bg-indigo-50 rounded-lg flex items-center justify-center mb-4">
                  <svg
                    className="h-6 w-6 text-indigo-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z"
                    />
                  </svg>
                </div>
                <h4 className="text-lg font-medium text-gray-900 mb-2">
                  Demographic Analysis
                </h4>
                <p className="text-gray-600">
                  Evaluate model fairness across gender, age, ethnicity, and
                  other sensitive attributes.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                <div className="w-12 h-12 bg-pink-50 rounded-lg flex items-center justify-center mb-4">
                  <svg
                    className="h-6 w-6 text-pink-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                    />
                  </svg>
                </div>
                <h4 className="text-lg font-medium text-gray-900 mb-2">
                  Bias Detection
                </h4>
                <p className="text-gray-600">
                  Identify and quantify potential biases in your model's
                  predictions and decision boundaries.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                <div className="w-12 h-12 bg-cyan-50 rounded-lg flex items-center justify-center mb-4">
                  <svg
                    className="h-6 w-6 text-cyan-600"
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
                  Mitigation Strategies
                </h4>
                <p className="text-gray-600">
                  Receive recommendations for addressing and mitigating
                  identified fairness issues in your model.
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Helper functions for data processing and visualization
  const processMetricDataForChart = (metricObj: { [key: string]: number } | {}, metricName: string) => {
    if (Object.keys(metricObj).length === 0) return [];
    
    return Object.entries(metricObj)
      .map(([threshold, value]) => ({
        threshold: parseFloat(threshold).toFixed(2),
        value: typeof value === 'number' ? value : 0,
        metric: metricName
      }))
      .sort((a, b) => parseFloat(a.threshold) - parseFloat(b.threshold));
  };

  const getMetricSummary = (metricObj: { [key: string]: number } | {}) => {
    if (Object.keys(metricObj).length === 0) return { avg: 0, min: 0, max: 0, count: 0 };
    
    const values = Object.values(metricObj).filter(v => typeof v === 'number') as number[];
    if (values.length === 0) return { avg: 0, min: 0, max: 0, count: 0 };
    
    return {
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      count: values.length
    };
  };

  const getMetricStatus = (summary: { avg: number; min: number; max: number; count: number }, metricName: string, thresholds: any): 'good' | 'warning' | 'poor' => {
    if (!thresholds || summary.count === 0) return 'good';
    
    const threshold = thresholds[`${metricName}_threshold`];
    if (!threshold) return 'good';
    
    // Different metrics have different ideal ranges
    if (metricName === 'disparate_impact') {
      if (summary.avg >= 0.8 && summary.avg <= 1.25) return 'good';
      if (summary.avg >= 0.7 && summary.avg <= 1.5) return 'warning';
      return 'poor';
    } else {
      const absValue = Math.abs(summary.avg);
      if (absValue <= threshold) return 'good';
      if (absValue <= threshold * 1.5) return 'warning';
      return 'poor';
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="p-8 space-y-8 bg-gray-50 min-h-screen"
    >
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Fairness Analysis</h1>
        <p className="text-gray-500 mt-1">
          Evaluating model fairness across demographic groups
        </p>
      </div>

      {fairnessAPIData && (
        <>
          {/* Dataset Overview */}
          <div className="bg-white rounded-xl p-6 shadow-md border border-gray-100">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Dataset Information</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
              <div className="text-center p-3 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{fairnessAPIData.metrics.dataset_info.sample_size}</div>
                <div className="text-sm text-gray-600">Sample Size</div>
              </div>
              <div className="text-center p-3 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{fairnessAPIData.metrics.dataset_info.total_features}</div>
                <div className="text-sm text-gray-600">Total Features</div>
              </div>
              <div className="text-center p-3 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{fairnessAPIData.metrics.sensitive_features.length}</div>
                <div className="text-sm text-gray-600">Sensitive Features</div>
              </div>
              <div className="text-center p-3 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="text-2xl font-bold text-gray-900">{fairnessAPIData.metrics.status}</div>
                <div className="text-sm text-gray-600">Status</div>
              </div>
            </div>
            
            <div className="mb-4">
              <h3 className="font-medium text-gray-900 mb-2">Sensitive Features Analyzed</h3>
              <div className="flex flex-wrap gap-2">
                {fairnessAPIData.metrics.sensitive_features?.map((feature: string, index: number) => (
                  <span 
                    key={index}
                    className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium"
                  >
                    {feature}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Fairness Metrics Visualizations */}
          {fairnessAPIData.metrics.metrics && Object.entries(fairnessAPIData.metrics.metrics).map(([feature, metrics]: [string, any]) => {
            const demographicParityData = processMetricDataForChart(metrics.demographic_parity, 'Demographic Parity');
            const equalOpportunityData = processMetricDataForChart(metrics.equal_opportunity, 'Equal Opportunity');
            const statisticalParityData = processMetricDataForChart(metrics.statistical_parity, 'Statistical Parity');
            
            // Combine data for multi-line chart
            const combinedData = demographicParityData.map((item, index) => ({
              threshold: item.threshold,
              demographicParity: item.value,
              equalOpportunity: equalOpportunityData[index]?.value || 0,
              statisticalParity: statisticalParityData[index]?.value || 0
            }));

            // Get metric summaries for overview cards
            const dpSummary = getMetricSummary(metrics.demographic_parity);
            const eoSummary = getMetricSummary(metrics.equal_opportunity);
            const diSummary = getMetricSummary(metrics.disparate_impact);
            const spSummary = getMetricSummary(metrics.statistical_parity);

            return (
              <div key={feature} className="bg-white rounded-xl p-6 shadow-md border border-gray-100 space-y-6">
                <h2 className="text-xl font-semibold text-gray-900 mb-4">
                  Fairness Analysis for: <span >{feature}</span>
                </h2>

                {/* Overview Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                  <div className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium text-gray-900 text-sm">Demographic Parity</h3>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        getMetricStatus(dpSummary, 'demographic_parity', metrics.interpretation) === 'good' 
                          ? 'bg-green-100 text-green-800' 
                          : getMetricStatus(dpSummary, 'demographic_parity', metrics.interpretation) === 'warning'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {getMetricStatus(dpSummary, 'demographic_parity', metrics.interpretation)}
                      </span>
                    </div>
                    <p className="text-lg font-bold text-gray-900">{dpSummary.avg.toFixed(4)}</p>
                    <p className="text-xs text-gray-500">Avg across {dpSummary.count} thresholds</p>
                  </div>

                  <div className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium text-gray-900 text-sm">Equal Opportunity</h3>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        getMetricStatus(eoSummary, 'equal_opportunity', metrics.interpretation) === 'good' 
                          ? 'bg-green-100 text-green-800' 
                          : getMetricStatus(eoSummary, 'equal_opportunity', metrics.interpretation) === 'warning'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {getMetricStatus(eoSummary, 'equal_opportunity', metrics.interpretation)}
                      </span>
                    </div>
                    <p className="text-lg font-bold text-gray-900">{eoSummary.avg.toFixed(4)}</p>
                    <p className="text-xs text-gray-500">Avg across {eoSummary.count} thresholds</p>
                  </div>

                  <div className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium text-gray-900 text-sm">Disparate Impact</h3>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        getMetricStatus(diSummary, 'disparate_impact', metrics.interpretation) === 'good' 
                          ? 'bg-green-100 text-green-800' 
                          : getMetricStatus(diSummary, 'disparate_impact', metrics.interpretation) === 'warning'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {getMetricStatus(diSummary, 'disparate_impact', metrics.interpretation)}
                      </span>
                    </div>
                    <p className="text-lg font-bold text-gray-900">{diSummary.count > 0 ? diSummary.avg.toFixed(4) : 'No data'}</p>
                    <p className="text-xs text-gray-500">{diSummary.count > 0 ? `Avg across ${diSummary.count} thresholds` : 'No threshold data'}</p>
                  </div>

                  <div className="p-4 border border-gray-200 rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium text-gray-900 text-sm">Statistical Parity</h3>
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        getMetricStatus(spSummary, 'statistical_parity', metrics.interpretation) === 'good' 
                          ? 'bg-green-100 text-green-800' 
                          : getMetricStatus(spSummary, 'statistical_parity', metrics.interpretation) === 'warning'
                          ? 'bg-yellow-100 text-yellow-800'
                          : 'bg-red-100 text-red-800'
                      }`}>
                        {getMetricStatus(spSummary, 'statistical_parity', metrics.interpretation)}
                      </span>
                    </div>
                    <p className="text-lg font-bold text-gray-900">{spSummary.avg.toFixed(4)}</p>
                    <p className="text-xs text-gray-500">Avg across {spSummary.count} thresholds</p>
                  </div>
                </div>

                {/* Line Chart - Metrics across Thresholds */}
                {combinedData.length > 0 && (
                  <div className="mb-6">
                    <h3 className="text-lg font-semibold text-gray-900 mb-4">Fairness Metrics Across Thresholds</h3>
                    <div className="h-80">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={combinedData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis 
                            dataKey="threshold" 
                            label={{ value: 'Threshold', position: 'insideBottom', offset: -5 }}
                          />
                          <YAxis label={{ value: 'Metric Value', angle: -90, position: 'insideLeft' }} />
                          <Tooltip />
                          <Legend />
                          <Line 
                            type="monotone" 
                            dataKey="demographicParity" 
                            stroke="#3B82F6" 
                            name="Demographic Parity"
                            strokeWidth={2}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="equalOpportunity" 
                            stroke="#EF4444" 
                            name="Equal Opportunity"
                            strokeWidth={2}
                          />
                          <Line 
                            type="monotone" 
                            dataKey="statisticalParity" 
                            stroke="#10B981" 
                            name="Statistical Parity"
                            strokeWidth={2}
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                    <p className="text-sm text-gray-600 mt-2">
                      This chart shows how fairness metrics vary across different threshold values. 
                      Values closer to 0 generally indicate better fairness.
                    </p>
                  </div>
                )}

                {/* Bar Chart - Metric Summary Comparison */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-900 mb-4">Metric Summary Comparison</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={[
                        { metric: 'Demographic Parity', average: dpSummary.avg, min: dpSummary.min, max: dpSummary.max },
                        { metric: 'Equal Opportunity', average: eoSummary.avg, min: eoSummary.min, max: eoSummary.max },
                        { metric: 'Statistical Parity', average: spSummary.avg, min: spSummary.min, max: spSummary.max }
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="metric" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="average" fill="#3B82F6" name="Average" />
                        <Bar dataKey="min" fill="#93C5FD" name="Minimum" />
                        <Bar dataKey="max" fill="#1E40AF" name="Maximum" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Thresholds Information */}
                {metrics.interpretation && (
                  <div className="p-4 bg-gray-50 rounded-lg">
                    <h4 className="font-medium text-gray-900 mb-3">Fairness Thresholds</h4>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                      <div>
                        <span className="text-gray-600">Demographic Parity:</span>
                        <span className="ml-2 font-medium">{metrics.interpretation.demographic_parity_threshold}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Equal Opportunity:</span>
                        <span className="ml-2 font-medium">{metrics.interpretation.equal_opportunity_threshold}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Disparate Impact:</span>
                        <span className="ml-2 font-medium">{metrics.interpretation.disparate_impact_threshold}</span>
                      </div>
                      <div>
                        <span className="text-gray-600">Statistical Parity:</span>
                        <span className="ml-2 font-medium">{metrics.interpretation.statistical_parity_threshold}</span>
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">
                      Values within these thresholds are considered acceptable for fairness.
                    </p>
                  </div>
                )}
              </div>
            );
          })}

          {/* Overall Interpretation */}
          

          {/* Processing Information */}
          <div className="bg-white rounded-xl p-6 shadow-md border border-gray-100">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Processing Details</h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Model:</span>
                <span className="ml-2 font-medium">{fairnessAPIData.model_name} v{fairnessAPIData.model_version}</span>
              </div>
              <div>
                <span className="text-gray-600">Project ID:</span>
                <span className="ml-2 font-medium">{fairnessAPIData.project_id}</span>
              </div>
              <div>
                <span className="text-gray-600">Processed:</span>
                <span className="ml-2 font-medium">{new Date(fairnessAPIData.metrics.processing_timestamp).toLocaleString()}</span>
              </div>
            </div>
          </div>
        </>
      )}
    </motion.div>
  );
};

export default FairnessPage;
