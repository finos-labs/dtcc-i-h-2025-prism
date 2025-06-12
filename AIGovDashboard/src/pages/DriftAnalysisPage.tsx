import { useParams } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Download, AlertTriangle, AlertCircle, ArrowUpDown, CheckCircle, Database } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Breadcrumb } from '../components/ui/breadcrumb';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, LineChart, Line, Legend,
  Cell, PieChart, Pie, Scatter, ScatterChart
} from 'recharts';
import { useState, useEffect } from 'react';
import { InfoTooltip } from '../components/InfoTooltip';
import { supabase } from '../lib/supabase';

// Define interfaces for the API response
interface DriftApiResponse {
  project_id: number;
  model_name: string;
  model_version: string;
  timestamp: string;
  metrics: {
    model_info: {
      type: string;
      name: string;
      version: string;
    };
    data_info: {
      total_samples: number;
      feature_count: number;
      feature_names: string[];
    };
    feature_drift: {
      [feature: string]: {
        type: string;
        ks_statistic: number;
        p_value: number;
      };
    };
    label_drift: {
      train_distribution: {
        [label: string]: number;
      };
      test_distribution: {
        [label: string]: number;
      };
      chi2_statistic: number;
      p_value: number;
    };
    covariate_drift: {
      mean_score: number;
      std_score: number;
      anomaly_rate: number;
      feature_types: {
        numerical: string[];
        categorical: string[];
      };
    };
  };
}

// Custom hook to fetch drift data
const useDriftAnalysisData = (projectId: string) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<DriftApiResponse | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      
      try {
        // Fetch model information from Supabase
        const { data: modelData, error: modelError } = await supabase
          .from('modeldetails')
          .select('model_id, project_id, dataset_id, model_version')
          .eq('project_id', projectId);
          
        if (modelError) {
          throw modelError;
        }
        
        if (!modelData || modelData.length === 0) {
          throw new Error('No model found for this project');
        }
        
        const modelId = modelData[0].model_id;
        const model_version = modelData[0].model_version;
        // Fetch drift analysis from API
        const accessToken = localStorage.getItem('access_token');
        
        const apiUrl = `http://localhost:8000/ml/drift/${projectId}/${modelId}/${model_version}`;
        
        const response = await fetch(
          apiUrl,
          {
            headers: {
              'Authorization': `Bearer ${accessToken}`,
              'Content-Type': 'application/json',
            },
          }
        );

        if (!response.ok) {
          throw new Error(`API error: ${response.status}`);
        }

        const apiData = await response.json();
        setData(apiData);
      } catch (err) {
        console.error('Failed to fetch drift analysis data:', err);
        setError(err instanceof Error ? err.message : 'Unknown error');
      } finally {
        setLoading(false);
      }
    };

    if (projectId) {
      fetchData();
    } else {
      setLoading(false);
      setError('Missing project ID');
    }
  }, [projectId]);

  return { data, loading, error };
};

// Helper function to convert feature drift data to chart format
const prepareFeatureDriftData = (featureDrift: DriftApiResponse['metrics']['feature_drift']) => {
  return Object.entries(featureDrift).map(([feature, stats]) => ({
    feature,
    ks_statistic: stats.ks_statistic,
    p_value: stats.p_value,
    isDrifting: stats.p_value < 0.05, // Common threshold for statistical significance
  }));
};

// Status indicator component
const DriftStatusIndicator = ({ value, threshold, label }: { value: number, threshold: number, label: string }) => {
  const isDrifting = value < threshold;
  
  return (
    <div className="flex items-center">
      {isDrifting ? (
        <AlertCircle className="h-5 w-5 text-red-500 mr-2" />
      ) : (
        <CheckCircle className="h-5 w-5 text-green-500 mr-2" />
      )}
      <span className={`font-medium ${isDrifting ? 'text-red-500' : 'text-green-500'}`}>
        {isDrifting ? 'Drift Detected' : 'No Significant Drift'} - {label}
      </span>
    </div>
  );
};

// Metric Card component
const MetricCard = ({ title, value, description, icon, status = 'normal' }: { 
  title: string;
  value: string | number;
  description: string;
  icon: React.ReactNode;
  status?: 'normal' | 'warning' | 'alert';
}) => {
  const statusColors = {
    normal: 'bg-blue-100 text-blue-800',
    warning: 'bg-yellow-100 text-yellow-800',
    alert: 'bg-red-100 text-red-800',
  };

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
        <InfoTooltip 
          title={title}
          entityType="metric"
          entityName={title}
        />
      </div>
      <div className="flex items-baseline space-x-2">
        <span className="text-3xl font-bold text-gray-900">{value}</span>
        <span className={`text-sm font-medium px-2.5 py-0.5 rounded-full ${statusColors[status]}`}>
          {status === 'normal' ? 'Stable' : status === 'warning' ? 'Warning' : 'Alert'}
        </span>
      </div>
      <div className="mt-4 flex items-center">
        <div className="p-2 bg-gray-100 rounded-full mr-3">
          {icon}
        </div>
        <p className="text-sm text-gray-500">{description}</p>
      </div>
    </div>
  );
};

// Feature Drift Chart
const FeatureDriftChart = ({ data }: { data: ReturnType<typeof prepareFeatureDriftData> }) => (
  <div className="h-[400px] w-full">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-sm font-medium text-gray-700">Feature Drift (KS Statistic)</h3>
    </div>
    <ResponsiveContainer width="100%" height="100%">
      <BarChart
        layout="vertical"
        data={data}
        margin={{ top: 20, right: 30, left: 120, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" opacity={0.3} horizontal={true} vertical={false} />
        <XAxis 
          type="number" 
          domain={[0, 1]}
          tickFormatter={(value) => value.toFixed(2)}
        />
        <YAxis 
          type="category" 
          dataKey="feature" 
          tick={{ fontSize: 12 }}
          width={120}
        />
        <Tooltip 
          formatter={(value: number) => [value.toFixed(3), 'KS Statistic']}
          contentStyle={{ 
            backgroundColor: 'rgba(255, 255, 255, 0.95)', 
            borderRadius: '8px',
            boxShadow: '0px 4px 12px rgba(0, 0, 0, 0.1)',
            border: '1px solid #E5E7EB'
          }}
        />
        <Bar 
          dataKey="ks_statistic" 
          animationDuration={1500}
          radius={[0, 4, 4, 0]}
        >
          {data.map((entry, index) => (
            <Cell 
              key={`cell-${index}`} 
              fill={entry.isDrifting ? '#EF4444' : '#3B82F6'} 
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  </div>
);

// Empty State Component for when there's no drift data
const EmptyState = () => (
  <div className="bg-white rounded-xl border border-gray-200 shadow-md overflow-hidden p-8">
    <div className="max-w-3xl mx-auto text-center">
      <div className="mx-auto w-20 h-20 bg-blue-50 rounded-full flex items-center justify-center mb-6">
        <ArrowUpDown className="h-10 w-10 text-blue-600" />
      </div>
      
      <h2 className="text-2xl font-bold text-gray-900 mb-3">Drift Analysis Not Available</h2>
      <p className="text-gray-600 max-w-2xl mx-auto mb-8">
        No drift analysis data is available yet. Drift analysis helps you monitor changes in your model's input data and output predictions over time.
      </p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="bg-blue-50 p-6 rounded-lg text-left">
          <h3 className="text-lg font-semibold text-blue-700 mb-2">What is Data Drift?</h3>
          <p className="text-blue-600">
            Data drift occurs when the statistical properties of your model's inputs change over time, potentially affecting the model's performance. Monitoring data drift helps identify when your model needs retraining.
          </p>
        </div>
        
        <div className="bg-purple-50 p-6 rounded-lg text-left">
          <h3 className="text-lg font-semibold text-purple-700 mb-2">What is Concept Drift?</h3>
          <p className="text-purple-600">
            Concept drift happens when the relationship between input and output variables changes over time. This can cause your model's predictions to become less accurate, even if the input distribution remains stable.
          </p>
        </div>
      </div>
      
      <div className="bg-gray-50 p-6 rounded-lg border border-gray-100">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Benefits of Drift Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="flex flex-col items-center p-4">
            <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center mb-3">
              <CheckCircle className="h-6 w-6 text-green-600" />
            </div>
            <h4 className="font-medium text-gray-800 mb-1">Improved Reliability</h4>
            <p className="text-sm text-gray-600 text-center">Detect when models need retraining to maintain accuracy</p>
          </div>
          
          <div className="flex flex-col items-center p-4">
            <div className="w-12 h-12 bg-blue-100 rounded-full flex items-center justify-center mb-3">
              <AlertCircle className="h-6 w-6 text-blue-600" />
            </div>
            <h4 className="font-medium text-gray-800 mb-1">Early Warning</h4>
            <p className="text-sm text-gray-600 text-center">Get notified of changes before they impact business outcomes</p>
          </div>
          
          <div className="flex flex-col items-center p-4">
            <div className="w-12 h-12 bg-purple-100 rounded-full flex items-center justify-center mb-3">
              <svg className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h4 className="font-medium text-gray-800 mb-1">Performance Tracking</h4>
            <p className="text-sm text-gray-600 text-center">Monitor how your model performs against changing data patterns</p>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// Add mock data for drift analysis
const mockDriftData: DriftApiResponse = {
  project_id: 123,
  model_name: "RandomForest Classifier",
  model_version: "1.0.0",
  timestamp: new Date().toISOString(),
  metrics: {
    model_info: {
      type: "classification",
      name: "Demo Model",
      version: "1.0.0"
    },
    data_info: {
      total_samples: 1200,
      feature_count: 8,
      feature_names: ["Market Volatility", "Debt-Equity Ratio", "Return on Assets", "Earnings Growth", 
                      "Price-Earnings Ratio", "Current Ratio", "Dividend Yield", "Market Cap"]
    },
    feature_drift: {
      "Market Volatility": {
        type: "numerical",
        ks_statistic: 0.18,
        p_value: 0.02
      },
      "Debt-Equity Ratio": {
        type: "numerical",
        ks_statistic: 0.12,
        p_value: 0.09
      },
      "Return on Assets": {
        type: "numerical",
        ks_statistic: 0.15,
        p_value: 0.04
      },
      "Earnings Growth": {
        type: "numerical",
        ks_statistic: 0.07,
        p_value: 0.32
      },
      "Price-Earnings Ratio": {
        type: "numerical",
        ks_statistic: 0.09,
        p_value: 0.21
      },
      "Current Ratio": {
        type: "numerical",
        ks_statistic: 0.06,
        p_value: 0.48
      },
      "Dividend Yield": {
        type: "numerical",
        ks_statistic: 0.11,
        p_value: 0.11
      },
      "Market Cap": {
        type: "numerical",
        ks_statistic: 0.05,
        p_value: 0.62
      }
    },
    label_drift: {
      train_distribution: {
        "Buy": 0.35,
        "Hold": 0.45,
        "Sell": 0.20
      },
      test_distribution: {
        "Buy": 0.28,
        "Hold": 0.42,
        "Sell": 0.30
      },
      chi2_statistic: 7.82,
      p_value: 0.02
    },
    covariate_drift: {
      mean_score: 0.14,
      std_score: 0.08,
      anomaly_rate: 0.12,
      feature_types: {
        numerical: ["Market Volatility", "Debt-Equity Ratio", "Return on Assets", "Earnings Growth", 
                    "Price-Earnings Ratio", "Current Ratio", "Dividend Yield", "Market Cap"],
        categorical: []
      }
    }
  }
};

const DriftAnalysisPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const isDummyProject = id === 'dummy-1' || id === 'dummy-2';
  
  // Use mock data for dummy projects
  const { data, loading, error } = isDummyProject 
    ? { data: mockDriftData, loading: false, error: null }
    : useDriftAnalysisData(id || '');
  
  const breadcrumbSegments = [
    { title: "Projects", href: "/home" },
    { title: "Project Details", href: `/projects/${id}` },
    { title: "Drift Analysis", href: `/projects/${id}/drift` },
  ];

  // Loading state
  if (loading) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <h2 className="text-xl font-semibold text-gray-700">Loading drift analysis data...</h2>
        </div>
      </div>
    );
  }

  // Error or no data state
  if (error || !data) {
    return (
      <div className="flex-1 p-8">
        <div className="max-w-7xl mx-auto">
          <div className="mb-6">
            <Breadcrumb segments={breadcrumbSegments} />
          </div>
          
          <div className="mb-10">
            <h1 className="text-3xl font-bold text-gray-900">Drift Analysis</h1>
            <p className="mt-2 text-gray-600">Monitor changes in your model's data and prediction patterns over time</p>
          </div>

          <EmptyState />
        </div>
      </div>
    );
  }

  // Prepare data for visualizations
  const featureDriftData = data.metrics.feature_drift ? prepareFeatureDriftData(data.metrics.feature_drift) : [];
  
  // Calculate overall drift status
  const hasDrift = data.metrics.label_drift.p_value < 0.05 || 
                  Object.values(data.metrics.feature_drift).some(f => f.p_value < 0.05);

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="p-8 space-y-8 bg-gray-50 min-h-screen"
    >
    

      <div>
        <h1 className="text-3xl font-bold text-gray-900">Drift Analysis</h1>
        <p className="text-gray-500 mt-1">
          Model: {data.model_name} (v{data.model_version})
        </p>
      </div>

      {/* Overall Drift Status */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Drift Status Summary</h2>
        <div className="flex flex-col space-y-3">
          <DriftStatusIndicator 
            value={data.metrics.label_drift.p_value} 
            threshold={0.05} 
            label="Label Distribution"
          />
          <DriftStatusIndicator 
            value={Math.min(...Object.values(data.metrics.feature_drift).map(f => f.p_value))}
            threshold={0.05} 
            label="Feature Distributions" 
          />
          
          <div className="mt-4 p-4 rounded-lg bg-blue-50 border border-blue-100">
            <h3 className="font-medium text-blue-800 mb-2">Analysis Summary</h3>
            <p className="text-blue-700 text-sm">
              {hasDrift ? 
                "Significant drift detected. Consider retraining your model with more recent data to maintain performance." : 
                "No significant drift detected. Your model is performing consistently with its training data."}
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <MetricCard
          title="Data Size"
          value={data.metrics.data_info.total_samples}
          description="Number of samples in current dataset"
          icon={<Database className="h-5 w-5 text-blue-500" />}
          status="normal"
        />
        <MetricCard
          title="Feature Count"
          value={data.metrics.data_info.feature_count}
          description="Number of features analyzed for drift"
          icon={<BarChart className="h-5 w-5 text-blue-500" />}
          status="normal"
        />
        <MetricCard
          title="Label Drift"
          value={data.metrics.label_drift.p_value < 0.05 ? "Detected" : "Stable"}
          description={`Chi-square test p-value: ${data.metrics.label_drift.p_value !== undefined ? data.metrics.label_drift.p_value.toFixed(4) : 'N/A'}`}
          icon={<AlertTriangle className="h-5 w-5 text-blue-500" />}
          status={data.metrics.label_drift.p_value < 0.05 ? "alert" : "normal"}
        />
      </div>

      {/* Feature Drift Chart */}
      <motion.div 
        whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
        transition={{ duration: 0.3 }}
        className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
      >
        <div className="flex justify-between items-center mb-6">
          <div className="flex items-center">
            <h2 className="text-xl font-semibold text-gray-900">Feature Drift Analysis</h2>
          
          </div>
          <InfoTooltip 
              title="Feature Drift Analysis"
              entityType="chart"
              entityName="Feature Drift Analysis"
              data={{ chartData: featureDriftData }}
            />
        </div>
        <FeatureDriftChart data={featureDriftData} />
      </motion.div>

      {/* Recommendations */}
      <div className="bg-white p-6 rounded-xl shadow-sm border border-gray-100">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">Recommendations</h2>
        <div className="space-y-4">
          {hasDrift ? (
            <>
              <div className="flex items-start">
                <div className="mt-0.5 mr-3 text-red-500">
                  <AlertCircle className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">Retrain Your Model</h3>
                  <p className="text-gray-600 text-sm">Consider retraining your model with more recent data that includes the shifted patterns.</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="mt-0.5 mr-3 text-yellow-500">
                  <AlertTriangle className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">Investigate Drift Sources</h3>
                  <p className="text-gray-600 text-sm">Examine the features with highest drift to understand what's changing in your data.</p>
                </div>
              </div>
            </>
          ) : (
            <>
              <div className="flex items-start">
                <div className="mt-0.5 mr-3 text-green-500">
                  <CheckCircle className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">Continue Monitoring</h3>
                  <p className="text-gray-600 text-sm">Your model is performing well, but continue regular drift monitoring to catch any future changes.</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="mt-0.5 mr-3 text-blue-500">
                  <ArrowUpDown className="h-5 w-5" />
                </div>
                <div>
                  <h3 className="font-medium text-gray-900">Set Up Alerts</h3>
                  <p className="text-gray-600 text-sm">Consider setting up automated alerts for early detection of potential drift.</p>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default DriftAnalysisPage; 