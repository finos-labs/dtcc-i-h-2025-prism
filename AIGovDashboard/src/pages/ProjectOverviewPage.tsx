import { useEffect, useState, useRef } from 'react';
import { useParams } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import { Clock, Upload, Info } from 'lucide-react';
import { Breadcrumb } from '../components/ui/breadcrumb';
import UploadModal from '../components/UploadModal';
import axios from 'axios';

interface Project {
  project_id: string;
  project_name: string;
  description: string;
  project_type: 'llm' | 'generic';
  project_status: 'Completed' | 'No Report Yet' | 'Running Test';
}

// Dummy projects data
const dummyProjects: Record<string, Project> = {
  'dummy-1': {
    project_id: 'dummy-1',
    project_name: 'Investment Portfolio Analysis',
    description: 'AI-powered investment strategy and risk assessment system',
    project_type: 'generic',
    project_status: 'Completed'
  },
  'dummy-2': {
    project_id: 'dummy-2',
    project_name: 'Customer Enquiry LLM',
    description: 'Advanced LLM for customer support and inquiry handling',
    project_type: 'llm',
    project_status: 'No Report Yet'
  }
};

interface ModelMetrics {
  version: string;
  health: string;
  performance: {
    accuracy: number;
    f1_score: number;
    auc_roc: number;
  };
  fairness: {
    gender_disparity: number;
    age_disparity: number;
    demographic_parity: number;
  };
  explainability: {
    feature_importance: number;
    local_fidelity: number;
    global_fidelity: number;
  };
  technical: {
    framework: string;
    model_size: string;
    parameters: string;
    input_shape: string;
    output_shape: string;
  };
  training: {
    optimizer: string;
    learning_rate: number;
    batch_size: number;
    epochs: number;
    loss_function: string;
  };
  dataset: {
    type: string;
    samples: number;
    features: number;
    distribution: string;
  };
  benchmarks: {
    industry_rank: string;
    performance_percentile: string;
    fairness_score: string;
  };
}

// Add these interfaces
interface ModelData {
  name: string;
  description: string;
  model_type: string;
  version: string;
  file_path: string;
  metadata: Record<string, any>;
  project_id: number;
  user_id: number;
  id: number;
  created_at: string;
  updated_at: string;
}

interface DatasetData {
  // Similar structure to ModelData, adjust as needed
  name: string;
  description: string;
  file_path: string;
  metadata: Record<string, any>;
  project_id: number;
  id: number;
  created_at: string;
  updated_at: string;
}

const ProjectOverviewPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [project, setProject] = useState<Project | null>(null);
  const [loading, setLoading] = useState(true);
  const [metrics] = useState<ModelMetrics>({
    version: 'v2.1.0',
    health: 'healthy',
    performance: {
      accuracy: 94.5,
      f1_score: 94.0,
      auc_roc: 96.0
    },
    fairness: {
      gender_disparity: 2.0,
      age_disparity: 3.0,
      demographic_parity: 98.0
    },
    explainability: {
      feature_importance: 85.0,
      local_fidelity: 92.0,
      global_fidelity: 90.0
    },
    technical: {
      framework: 'PyTorch',
      model_size: '256 MB',
      parameters: '12.5M',
      input_shape: '[1, 3, 224, 224]',
      output_shape: '[1, 1000]'
    },
    training: {
      optimizer: 'Adam',
      learning_rate: 0.001,
      batch_size: 32,
      epochs: 100,
      loss_function: 'CrossEntropyLoss'
    },
    dataset: {
      type: 'Synthetic',
      samples: 10000,
      features: 20,
      distribution: 'Normal'
    },
    benchmarks: {
      industry_rank: '92/100',
      performance_percentile: '95%',
      fairness_score: '98/100'
    }
  });

  const [isUploadModalVisible, setIsUploadModalVisible] = useState(false);

  // Check if this is a dummy project
  const isDummyProject = id === 'dummy-1' || id === 'dummy-2';
  
  // New state to track if a model is uploaded
  const [modelUploaded, setModelUploaded] = useState(isDummyProject);

  // State for processing animation
  const [showProcessingAnimation, setShowProcessingAnimation] = useState(false);
  const [processingStep, setProcessingStep] = useState(0);
  const [processingModel, setProcessingModel] = useState({
    name: '',
    version: '',
    status: 'processing',
    message: ''
  });
  const processingIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const dataPointsIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [dataPoints, setDataPoints] = useState<number[]>([]);
  
  // State to track if error message should be dismissed
  const [errorDismissed, setErrorDismissed] = useState(false);
  
  // Functions for button actions
  const handleContactSupport = () => {
    // Open email client with support email
    window.location.href = "mailto:support@aigovdashboard.com?subject=Resource%20Limit%20Exceeded&body=Project%20ID:%20" + id;
  };
  
  const handleUpgradePlan = () => {
    // Redirect to a hypothetical upgrade page
    window.location.href = "/upgrade-plan?from=project&id=" + id;
  };
  
  const dismissError = () => {
    setErrorDismissed(true);
  };

  const processingSteps = [
    "Initializing model pipeline...",
    "Processing model parameters...",
    "Validating model structure...",
    "Running performance tests...",
    "Evaluating metrics...",
    "Generating visualizations..."
  ];

  // Generate random data points for the animated chart
  useEffect(() => {
    if (showProcessingAnimation) {
      const points = Array.from({ length: 20 }, () => Math.floor(Math.random() * 60) + 20);
      setDataPoints(points);
      
      dataPointsIntervalRef.current = setInterval(() => {
        setDataPoints(prev => {
          const newPoints = [...prev];
          newPoints.shift();
          newPoints.push(Math.floor(Math.random() * 60) + 20);
          return newPoints;
        });
      }, 1000);
    }
    
    return () => {
      if (dataPointsIntervalRef.current) {
        clearInterval(dataPointsIntervalRef.current);
      }
    };
  }, [showProcessingAnimation]);

  // Listen for model upload events from UploadModal
  useEffect(() => {
    // Handler for when model processing starts
    const handleModelProcessingStart = (event: CustomEvent) => {
      const { modelName, modelVersion } = event.detail;
      
      // Start the processing animation
      setProcessingModel({
        name: modelName,
        version: modelVersion,
        status: 'processing',
        message: `Processing ${modelName} v${modelVersion}...`
      });
      setShowProcessingAnimation(true);
      
      // Cycle through processing steps
      let step = 0;
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
      }
      
      processingIntervalRef.current = setInterval(() => {
        step = (step + 1) % processingSteps.length;
        setProcessingStep(step);
      }, 3000);
      
      // Set minimum display time
      setTimeout(() => {
        // If we haven't received completion/error event yet, keep showing
      }, 60000); // 1 minute minimum
    };
    
    // Handler for when model processing completes successfully
    const handleModelProcessingSuccess = (event: CustomEvent) => {
      const { modelName, modelVersion } = event.detail;
      
      // Update the model status
      setProcessingModel({
        name: modelName,
        version: modelVersion,
        status: 'success',
        message: `${modelName} v${modelVersion} processed successfully!`
      });
      
      // Stop the step cycling
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
      }
      
      // Set model as uploaded
      setModelUploaded(true);
      
      // Hide the animation after a delay and reload the page
      setTimeout(() => {
        setShowProcessingAnimation(false);
        // Add page reload here
        window.location.reload();
      }, 10000);
    };
    
    // Handler for when model processing fails
    const handleModelProcessingError = (event: CustomEvent) => {
      const { modelName, modelVersion } = event.detail;
      
      // Update the model status
      setProcessingModel({
        name: modelName,
        version: modelVersion, 
        status: 'error',
        message: `Model exceeds free plan limits due to high compute requirements. Please upload a simplified version or contact our team to upgrade`
      });
      
      // Stop the step cycling
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
      }
      
      // Hide the animation after a delay and reload the page
      setTimeout(() => {
        setShowProcessingAnimation(false);
        // Add page reload here, even on error to refresh UI state
        window.location.reload();
      }, 10000);
    };
    
    // Add event listeners
    window.addEventListener('modelProcessingStart', handleModelProcessingStart as EventListener);
    window.addEventListener('modelProcessingSuccess', handleModelProcessingSuccess as EventListener);
    window.addEventListener('modelProcessingError', handleModelProcessingError as EventListener);
    
    // Clean up on unmount
    return () => {
      window.removeEventListener('modelProcessingStart', handleModelProcessingStart as EventListener);
      window.removeEventListener('modelProcessingSuccess', handleModelProcessingSuccess as EventListener);
      window.removeEventListener('modelProcessingError', handleModelProcessingError as EventListener);
      
      if (processingIntervalRef.current) {
        clearInterval(processingIntervalRef.current);
      }
      
      if (dataPointsIntervalRef.current) {
        clearInterval(dataPointsIntervalRef.current);
      }
    };
  }, []);

  const [models, setModels] = useState<ModelData[]>([]);
  const [datasets, setDatasets] = useState<DatasetData[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Add this function to fetch both models and datasets
  const fetchProjectData = async () => {
    if (!id) return;
    
    setIsLoading(true);
    setError(null);
    
    // Get the access token from localStorage
    const accessToken = localStorage.getItem('access_token');
    
    // Configure headers with the auth token
    const config = {
      headers: {
        'Authorization': `Bearer ${accessToken}`,
        'Content-Type': 'application/json'
      }
    };
    
    try {
      // Only fetch models for now since the dataset endpoint seems to be missing
      const modelsResponse = await axios.get(`http://localhost:8000/ml/${id}/models/list`, config);
      
      setModels(modelsResponse.data);
      
      // If we have models, set modelUploaded to true
      if (modelsResponse.data && modelsResponse.data.length > 0) {
        setModelUploaded(true);
      }
      
      // Comment out the datasets fetch until the API endpoint is available
      // Try to fetch datasets separately to avoid Promise.all failing completely
      try {
        const datasetsResponse = await axios.get(`http://localhost:8000/ml/${id}/datasets/list`, config);
        setDatasets(datasetsResponse.data);
      } catch (datasetErr) {
        console.log('Datasets API not available yet:', datasetErr);
        // Don't fail the whole operation if just datasets aren't available
        setDatasets([]);
      }
      
    } catch (err) {
      console.error('Error fetching project data:', err);
      // Instead of showing specific error messages, show the custom message
      setError('Model exceeds free plan limits due to high compute requirements. Please upload a simplified version or contact our team to upgrade');
    } finally {
      setIsLoading(false);
    }
  };

  // Update useEffect to include the new fetch
  useEffect(() => {
    fetchProject();
    fetchProjectData();
  }, [id]);

  useEffect(() => {
    fetchProject();
  }, [id]);

  const fetchProject = async () => {
    try {
      if (!id) return;

      // Check if it's a dummy project first
      if (id in dummyProjects) {
        setProject(dummyProjects[id]);
        setLoading(false);
        setModelUploaded(true);
        return;
      }

      // If not a dummy project, fetch from database
      const { data, error } = await supabase
        .from('projectdetails')
        .select('*')
        .eq('project_id', id)
        .single();

      if (error) throw error;
      setProject(data);
      setModelUploaded(true);
    } catch (error) {
      console.error('Error fetching project:', error);
      // Display the custom error message for any API errors
      setError('Model exceeds free plan limits due to high compute requirements. Please upload a simplified version or contact our team to upgrade');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="p-8">Loading...</div>;
  }

  if (!project) {
    return <div className="p-8">Project not found</div>;
  }

  // Show demo metrics only for the Investment Portfolio Analysis demo project
  const isPortfolioDemo = id === 'dummy-1';

  const breadcrumbSegments = [
    { title: "Projects", href: "/home" },
    { title: project.project_name, href: `/projects/${project.project_id}` },
  ];

  // Show regular upload modal
  const showUploadModal = () => {
    setIsUploadModalVisible(true);
  };
  
  // Handle close for regular upload modal
  const handleUploadModalClose = () => {
    setIsUploadModalVisible(false);
  };
  
  // General Project Overview Component
  const GeneralProjectOverview = () => (
    <div className="space-y-8">
  

      {/* Upload section */}
            {/* Benefits section */}
      <div>
        <h2 className="text-xl font-bold text-gray-900 mb-4">Benefits of Testing Your Model</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
              <svg className="h-6 w-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Comprehensive Metrics</h3>
            <p className="text-gray-600">
              Track accuracy, precision, recall, F1 score, and AUC-ROC across different thresholds and data segments.
            </p>
          </div>
          
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="w-12 h-12 bg-purple-50 rounded-lg flex items-center justify-center mb-4">
              <svg className="h-6 w-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Interactive Visualizations</h3>
            <p className="text-gray-600">
              Explore dynamic charts and plots to understand performance patterns and identify areas for improvement.
            </p>
          </div>
          
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <div className="w-12 h-12 bg-amber-50 rounded-lg flex items-center justify-center mb-4">
              <svg className="h-6 w-6 text-amber-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">Actionable Insights</h3>
            <p className="text-gray-600">
              Receive recommendations based on model performance to help optimize and improve your AI system.
            </p>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Why Performance Analysis Matters</h3>
            <p className="text-gray-600 mb-4">
              A comprehensive performance analysis helps you understand how well your model works across different scenarios and data distributions.
            </p>
            <ul className="space-y-2">
              <li className="flex items-start">
                <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-gray-600">Identify prediction strengths and weaknesses</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-gray-600">Compare model versions to track improvements</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-gray-600">Optimize decision thresholds for your use case</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-green-500 mr-2 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span className="text-gray-600">Ensure reliability before deployment</span>
              </li>
            </ul>
          </div>
          
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h3 className="text-lg font-medium text-gray-900 mb-4">Fairness and Explainability Benefits</h3>
            <div className="space-y-4">
              <div className="flex items-start">
                <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center mr-3 flex-shrink-0">
                  <svg className="h-5 w-5 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-base font-medium text-gray-900">Bias Detection</h4>
                  <p className="text-sm text-gray-600">Identify and mitigate bias across different demographic groups to ensure fair outcomes.</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="w-10 h-10 bg-purple-50 rounded-lg flex items-center justify-center mr-3 flex-shrink-0">
                  <svg className="h-5 w-5 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-base font-medium text-gray-900">Model Transparency</h4>
                  <p className="text-sm text-gray-600">Understand how your model makes decisions with feature importance and local explanations.</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="w-10 h-10 bg-teal-50 rounded-lg flex items-center justify-center mr-3 flex-shrink-0">
                  <svg className="h-5 w-5 text-teal-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-3 7h3m-3 4h3m-6-4h.01M9 16h.01" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-base font-medium text-gray-900">Regulatory Compliance</h4>
                  <p className="text-sm text-gray-600">Meet growing requirements for AI governance and documentation in regulated industries.</p>
                </div>
              </div>
              
              <div className="flex items-start">
                <div className="w-10 h-10 bg-red-50 rounded-lg flex items-center justify-center mr-3 flex-shrink-0">
                  <svg className="h-5 w-5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  </svg>
                </div>
                <div>
                  <h4 className="text-base font-medium text-gray-900">Risk Mitigation</h4>
                  <p className="text-sm text-gray-600">Identify and address potential issues before they affect your users or business.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Add this component to display models
  const ModelsSection = () => (
    <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
      <h2 className="text-xl font-bold mb-4">Models</h2>
      {models.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {models.map((model) => (
            <div key={model.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="flex justify-between items-start mb-2">
                <h3 className="font-semibold text-lg">{model.name}</h3>
                <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-sm">
                  v{model.version}
                </span>
              </div>
              <p className="text-gray-600 text-sm mb-2">{model.description}</p>
              <div className="flex flex-wrap gap-2 text-sm">
                <span className="px-2 py-1 bg-gray-100 rounded-full">
                  Type: {model.model_type}
                </span>
                <span className="px-2 py-1 bg-gray-100 rounded-full">
                  Created: {new Date(model.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500">No models uploaded yet.</p>
      )}
    </div>
  );

  // Add this component to display datasets
  const DatasetsSection = () => (
    <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
      <h2 className="text-xl font-bold mb-4">Datasets</h2>
      {datasets.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {datasets.map((dataset) => (
            <div key={dataset.id} className="border rounded-lg p-4 hover:shadow-md transition-shadow">
              <h3 className="font-semibold text-lg mb-2">{dataset.name}</h3>
              <p className="text-gray-600 text-sm mb-2">{dataset.description}</p>
              <div className="flex flex-wrap gap-2 text-sm">
                <span className="px-2 py-1 bg-gray-100 rounded-full">
                  Created: {new Date(dataset.created_at).toLocaleDateString()}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-gray-500">No datasets available.</p>
      )}
    </div>
  );

  return (
    <div className="p-8 space-y-8">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">Project Overview</h1>
        {/* Keep this upload button for all projects */}
          <button 
          className="inline-flex items-center px-5 py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 text-white rounded-lg hover:from-blue-700 hover:to-indigo-700 shadow-sm transition-all duration-200"
            onClick={showUploadModal}
          >
          <svg className="mr-2 h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
          {modelUploaded ? 'Upload New Version' : 'Upload Model'}
          </button>
      </div>
      
      <Breadcrumb segments={breadcrumbSegments} />

      {/* Display custom error message */}
      {error && !errorDismissed && !isDummyProject && (
        <div className="bg-gradient-to-r from-orange-50 to-amber-50 border border-orange-200 rounded-xl p-6 mb-6 shadow-sm">
          <div className="flex flex-col md:flex-row items-start md:items-center gap-4">
            <div className="flex-shrink-0 w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
              <svg className="h-6 w-6 text-orange-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </div>
            <div className="flex-1">
              <div className="flex justify-between items-start">
                <h3 className="text-lg font-semibold text-orange-800 mb-1">Resource Limit Reached</h3>
                <button 
                  onClick={dismissError}
                  className="text-orange-400 hover:text-orange-600"
                  aria-label="Dismiss"
                >
                  <svg className="h-5 w-5" viewBox="0 0 20 20" fill="currentColor">
                    <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
                  </svg>
                </button>
              </div>
              <p className="text-orange-700 mb-3">{error}</p>
              <div className="flex flex-wrap gap-3">
                <button 
                  onClick={handleContactSupport}
                  className="px-4 py-2 bg-white text-orange-700 rounded-lg border border-orange-300 hover:bg-orange-50 transition-colors duration-200 font-medium text-sm"
                >
                  Contact Support
                </button>
                <button 
                  onClick={handleUpgradePlan}
                  className="px-4 py-2 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-lg hover:from-orange-600 hover:to-amber-600 transition-colors duration-200 font-medium text-sm shadow-sm"
                >
                  Upgrade Plan
                </button>
              </div>
            </div>
            <div className="hidden md:block w-32 flex-shrink-0">
              <div className="relative h-24 w-24">
                <svg className="absolute inset-0" viewBox="0 0 100 100">
                  <circle 
                    cx="50" cy="50" r="45" 
                    fill="none" 
                    stroke="#FED7AA" 
                    strokeWidth="8" 
                  />
                  <circle 
                    cx="50" cy="50" r="45" 
                    fill="none" 
                    stroke="#F97316" 
                    strokeWidth="8" 
                    strokeDasharray="283" 
                    strokeDashoffset="100" 
                  />
                </svg>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-orange-700 font-bold text-lg">85%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header - Enhanced design */}
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6 mb-8">
        <div className="space-y-2">
          <div className="flex items-center">
        <h1 className="text-3xl font-bold tracking-tight text-gray-900">
          {project.project_name}
        </h1>
            <span className={`ml-4 inline-flex items-center px-3 py-1 rounded-full text-xs font-medium ${
            project.project_status === 'Completed' ? 'bg-green-100 text-green-800' : 
            project.project_status === 'Running Test' ? 'bg-blue-100 text-blue-800' :
            'bg-gray-100 text-gray-800'
          }`}>
            {project.project_status.toLowerCase()}
          </span>
          </div>
          <p className="text-gray-500 text-lg">{project.description}</p>
          <div className="flex items-center space-x-6 pt-2 text-sm text-gray-500 border-t border-gray-100 mt-4 pt-4">
           
            <div className="flex items-center">
              <svg className="h-4 w-4 mr-1 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z" />
              </svg>
              {project.project_type === 'llm' ? 'LLM' : 'Generic AI'}
        </div>
            
          </div>
        </div>
      </div>

      {/* Model Processing Animation */}
      {showProcessingAnimation && (
        <div className="mb-8 w-full">
          <div className={`w-full bg-white rounded-xl border shadow-md overflow-hidden ${
            processingModel.status === 'processing' ? 'border-blue-300' :
            processingModel.status === 'success' ? 'border-green-300' : 'border-red-300'
          }`}>
            <div className={`h-2 ${
              processingModel.status === 'processing' ? 'bg-blue-500' :
              processingModel.status === 'success' ? 'bg-green-500' : 'bg-red-500'
            }`}>
              {processingModel.status === 'processing' && (
                <div className="h-full bg-white/30 animate-pulse w-1/3"></div>
              )}
            </div>
            
            <div className="p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  {processingModel.status === 'processing' ? (
                    <div className="mr-4 relative w-12 h-12">
                      <svg className="animate-spin w-full h-full text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                    </div>
                  ) : processingModel.status === 'success' ? (
                    <div className="mr-4 w-12 h-12 bg-green-100 rounded-full flex items-center justify-center text-green-500 text-2xl">
                      ✓
                    </div>
                  ) : (
                    <div className="mr-4 w-12 h-12 bg-red-100 rounded-full flex items-center justify-center text-red-500 text-2xl">
                      ✗
                    </div>
                  )}
                  
                  <div>
                    <h3 className="text-lg font-semibold text-gray-800">
                      {processingModel.status === 'processing' ? 'Processing Model' :
                       processingModel.status === 'success' ? 'Processing Complete' : 'Processing Failed'}
                    </h3>
                    <p className="text-sm text-gray-600">{processingModel.message}</p>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-sm font-medium text-gray-500">Model: {processingModel.name}</div>
                  <div className="text-sm text-gray-400">Version: {processingModel.version}</div>
                </div>
              </div>
              
              {processingModel.status === 'processing' && (
                <>
                  {/* Progress visualization */}
                  <div className="mb-6 bg-blue-50 p-4 rounded-lg">
                    <div className="flex justify-between items-center mb-2">
                      <span className="text-sm font-medium text-blue-800">Processing Activity</span>
                      <span className="text-xs text-blue-600 bg-blue-100 px-2 py-1 rounded-full">Live</span>
                    </div>
                    <div className="h-32 flex items-end gap-1">
                      {dataPoints.map((point, i) => (
                        <div 
                          key={i}
                          className="bg-blue-500 rounded-t w-full"
                          style={{ 
                            height: `${point}%`,
                            opacity: 0.3 + (i / dataPoints.length) * 0.7
                          }}
                        ></div>
                      ))}
                    </div>
                  </div>
                  
                  {/* Processing steps */}
                  <div className="mt-6">
                    <div className="text-sm font-medium text-gray-700 mb-2">
                      {processingSteps[processingStep]}
                    </div>
                    <div className="grid grid-cols-6 gap-4 mt-3">
                      {processingSteps.map((step, i) => (
                        <div key={i} className="relative">
                          <div className={`h-1 rounded-full ${
                            i < processingStep ? 'bg-blue-500' : 
                            i === processingStep ? 'bg-blue-300 animate-pulse' : 'bg-gray-200'
                          }`}></div>
                          <div className="mt-1 text-xs text-gray-400 truncate">{step.split('...')[0]}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                  
                  <div className="flex items-center gap-4 text-sm text-gray-500 mt-6 justify-center">
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-blue-500 rounded-full mr-2 animate-pulse"></div>
                      <span>Processing</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                      <span>Completed</span>
                    </div>
                    <div className="flex items-center">
                      <div className="w-3 h-3 bg-gray-300 rounded-full mr-2"></div>
                      <span>Pending</span>
                    </div>
                  </div>
                </>
              )}
              
              {processingModel.status === 'success' && (
                <div className="mt-4 p-3 bg-green-50 rounded">
                  <p className="text-sm text-green-700">
                    Model processing completed successfully. The results are ready to view.
                  </p>
                </div>
              )}
              
              {processingModel.status === 'error' && (
                <div className="mt-4 p-5 bg-gradient-to-r from-orange-50 to-amber-50 rounded-lg border border-orange-200">
                  <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3">
                    <div className="flex-shrink-0 w-10 h-10 bg-orange-100 rounded-full flex items-center justify-center">
                      <svg className="h-5 w-5 text-orange-500" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                    </div>
                    <div className="flex-1">
                      <h3 className="text-base font-semibold text-orange-800 mb-1">Resource Limit Reached</h3>
                      <p className="text-sm text-orange-700 mb-3">{processingModel.message}</p>
                      <div className="flex flex-wrap gap-2">
                        <button 
                          onClick={handleContactSupport}
                          className="px-3 py-1.5 bg-white text-orange-700 rounded-lg border border-orange-300 hover:bg-orange-50 transition-colors duration-200 text-xs font-medium"
                        >
                          Contact Support
                        </button>
                        <button 
                          onClick={handleUpgradePlan}
                          className="px-3 py-1.5 bg-gradient-to-r from-orange-500 to-amber-500 text-white rounded-lg hover:from-orange-600 hover:to-amber-600 transition-colors duration-200 text-xs font-medium shadow-sm"
                        >
                          Upgrade Plan
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Display appropriate content based on project type and model upload status */}
      {isPortfolioDemo && modelUploaded ? (
        // Show the portfolio demo metrics for dummy-1 project
        <div className="bg-white rounded-lg border border-gray-200 my-6">
          <div className="p-6 pb-4 border-b border-gray-100">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <h2 className="text-xl font-semibold text-gray-900">
                  Portfolio Optimization Model
                </h2>
                <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                  {metrics.version.replace('v', '')}
                </span>
                <span className="px-2 py-0.5 text-xs font-medium bg-green-100 text-green-800 rounded-full">
                  {metrics.health}
                </span>
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 border-b border-gray-100">
            {/* Performance Metrics */}
            <div className="p-6 border-r border-gray-100">
              <h3 className="text-sm font-medium text-gray-500 uppercase flex items-center mb-4">
                Performance
                <Info className="h-4 ml-2 text-gray-400" />
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Accuracy</span>
                  <span className="font-medium">{metrics.performance.accuracy}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">F1 Score</span>
                  <span className="font-medium">{metrics.performance.f1_score}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">AUC-ROC</span>
                  <span className="font-medium">{metrics.performance.auc_roc}%</span>
                </div>
              </div>
            </div>

            {/* Fairness Metrics */}
            <div className="p-6 border-r border-gray-100">
              <h3 className="text-sm font-medium text-gray-500 uppercase flex items-center mb-4">
                Fairness
                <Info className="h-4 ml-2 text-gray-400" />
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Gender Disparity</span>
                  <span className="font-medium">{metrics.fairness.gender_disparity}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Age Disparity</span>
                  <span className="font-medium">{metrics.fairness.age_disparity}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Demographic Parity</span>
                  <span className="font-medium">{metrics.fairness.demographic_parity}%</span>
                </div>
              </div>
            </div>

            {/* Explainability Metrics */}
            <div className="p-6">
              <h3 className="text-sm font-medium text-gray-500 uppercase flex items-center mb-4">
                Explainability
                <Info className="h-4 ml-2 text-gray-400" />
              </h3>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Feature Importance</span>
                  <span className="font-medium">{metrics.explainability.feature_importance}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Local Fidelity</span>
                  <span className="font-medium">{metrics.explainability.local_fidelity}%</span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-gray-600">Global Fidelity</span>
                  <span className="font-medium">{metrics.explainability.global_fidelity}%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Technical Details and Training Configuration */}
          <div className="grid grid-cols-1 md:grid-cols-2 border-b border-gray-100">
            <div className="p-6 border-r border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-500 uppercase">Technical Details</h3>
                <Info className="h-4 w-4 text-gray-400" />
              </div>
              <div className="space-y-4">
                {Object.entries(metrics.technical).slice(0, 4).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center">
                    <span className="text-gray-600">{key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</span>
                    <span className="font-medium">{value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-500 uppercase">Training Configuration</h3>
                <Info className="h-4 w-4 text-gray-400" />
              </div>
              <div className="space-y-4">
                {Object.entries(metrics.training).slice(0, 4).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center">
                    <span className="text-gray-600">{key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</span>
                    <span className="font-medium">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Dataset Information and Benchmarks */}
          <div className="grid grid-cols-1 md:grid-cols-2">
            <div className="p-6 border-r border-gray-100">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-500 uppercase">Dataset Information</h3>
                <Info className="h-4 w-4 text-gray-400" />
              </div>
              <div className="space-y-4">
                {Object.entries(metrics.dataset).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center">
                    <span className="text-gray-600">{key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</span>
                    <span className="font-medium">{value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-sm font-medium text-gray-500 uppercase">Benchmarks</h3>
                <Info className="h-4 w-4 text-gray-400" />
              </div>
              <div className="space-y-4">
                {Object.entries(metrics.benchmarks).map(([key, value]) => (
                  <div key={key} className="flex justify-between items-center">
                    <span className="text-gray-600">{key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}</span>
                    <span className="font-medium">{value}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Show Models and Datasets sections if data exists */}
          {(models.length > 0 || datasets.length > 0) ? (
            <>
              <ModelsSection />
              <DatasetsSection />
            </>
          ) : (
            // Show the general overview with benefits section if no data
            <GeneralProjectOverview />
          )}
        </>
      )}

      {/* Upload Modals */}
      <UploadModal 
        isVisible={isUploadModalVisible} 
        onClose={handleUploadModalClose} 
        isNewVersion={modelUploaded}
        projectId={id}
      />
    </div>
  );
};

// Add these animation utilities to your CSS or tailwind config
const cssAdditions = `
@keyframes ping-slow {
  0% {
    transform: scale(1);
    opacity: 0.5;
  }
  50% {
    transform: scale(1.2);
    opacity: 0.3;
  }
  100% {
    transform: scale(1);
    opacity: 0.5;
  }
}

.animate-ping-slow {
  animation: ping-slow 3s cubic-bezier(0, 0, 0.2, 1) infinite;
}

.animation-delay-500 {
  animation-delay: 500ms;
}
`;

export default ProjectOverviewPage;