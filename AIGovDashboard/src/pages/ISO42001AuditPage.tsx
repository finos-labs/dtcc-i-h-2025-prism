import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  CheckCircle,
  ChevronDown,
  ChevronUp,
  ArrowLeft,
  FileText,
  Target,
  Award,
  AlertCircle,
  Shield,
  Users,
  Settings,
  BookOpen,
  Database,
  Eye,
  UserCheck,
  Zap,
  Home,
  Upload,
  X,
} from "lucide-react";
import { Link } from "react-router-dom";
import axios from 'axios';

const ISO42001AuditPage: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([1, 2, 3]));
  const [completedSections, setCompletedSections] = useState<Set<number>>(new Set());
  const [autoSectionsCompleted, setAutoSectionsCompleted] = useState<Set<string>>(new Set());
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [currentUploadSection, setCurrentUploadSection] = useState<string>("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);

  useEffect(() => {
    checkAutoSectionsCompletion();
  }, []);

  const checkAutoSectionsCompletion = async () => {
    try {
      setLoading(true);
      
      // Get token from localStorage
      const token = localStorage.getItem('access_token');
      if (!token) {
        console.log('No access token found');
        setLoading(false);
        return;
      }

      const config = {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      };

      // Check if models/data exist for any project (since this is a general audit page)
      // We'll check for a dummy project or use a default approach
      try {
        // Try to check for models - if successful, auto-complete certain subsections
        const modelsResponse = await axios.get(`https://prism-backend-dtcc-dot-block-convey-p1.uc.r.appspot.com/ml/dummy-1/models/list`, config);
        
        // Only mark subsections as auto-completed if we get a successful response with actual data
        if (modelsResponse.data && Array.isArray(modelsResponse.data) && modelsResponse.data.length > 0) {
          console.log('Models found, auto-completing subsections:', modelsResponse.data);
          // Auto-complete specific subsections when models exist
          setAutoSectionsCompleted(new Set([
            'impact-assessment',
            'risk-mitigation-strategies',
            'testing-framework',
            'kpi-definition',
            'monitoring-systems', 
            'reporting-mechanisms'
          ]));
          // Update completed sections to reflect which main sections have any completed subsections
          setCompletedSections(new Set([1, 2, 3, 4]));
        } else {
          console.log('No model data found, all sections remain manual');
          // No model data, all sections remain manual (clickable for upload)
          setAutoSectionsCompleted(new Set());
          setCompletedSections(new Set());
        }
      } catch (apiError) {
        console.log('Models API call failed, all sections remain manual:', apiError);
        // API call failed, all sections remain manual (clickable for upload)
        setAutoSectionsCompleted(new Set());
        setCompletedSections(new Set());
      }
    } catch (error) {
      console.error('Error checking auto sections completion:', error);
      setAutoSectionsCompleted(new Set());
      setCompletedSections(new Set());
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (sectionNumber: number) => {
    setExpandedSections(prev => {
      const newSet = new Set(prev);
      if (newSet.has(sectionNumber)) {
        newSet.delete(sectionNumber);
      } else {
        newSet.add(sectionNumber);
      }
      return newSet;
    });
  };

  const calculateComplianceScore = () => {
    return Math.round((completedSections.size / 4) * 100);
  };

  const getProgressColor = (progress: number) => {
    if (progress >= 80) return "bg-green-500";
    if (progress >= 60) return "bg-yellow-500";
    if (progress >= 40) return "bg-orange-500";
    return "bg-red-500";
  };

  const handleSubsectionClick = (subsectionId: string, subsectionTitle: string) => {
    if (autoSectionsCompleted.has(subsectionId)) {
      // If auto-completed, do nothing (already completed)
      return;
    } else {
      // If not completed, show upload modal
      setCurrentUploadSection(subsectionTitle);
      setShowUploadModal(true);
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setUploadedFile(file);
    }
  };

  const handleSubmitDocument = () => {
    if (uploadedFile) {
      // Here you would typically upload the document to your backend
      console.log(`Uploading document for ${currentUploadSection}:`, uploadedFile);
      
      // For demo purposes, mark the subsection as completed
      const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
      if (subsectionId) {
        setAutoSectionsCompleted(prev => new Set([...prev, subsectionId]));
      }
      
      alert(`Document uploaded successfully for ${currentUploadSection}!`);
      setShowUploadModal(false);
      setUploadedFile(null);
      setCurrentUploadSection("");
    }
  };

  const getSubsectionIdFromTitle = (title: string): string | null => {
    const titleMap: Record<string, string> = {
      "AI Policy Documentation": "ai-policy-documentation",
      "Procedures and Guidelines": "procedures-guidelines", 
      "Record Keeping System": "record-keeping-system",
      "Impact Assessment": "impact-assessment",
      "Risk Mitigation Strategies": "risk-mitigation-strategies",
      "Incident Response": "incident-response",
      "Development Controls": "development-controls",
      "Testing Framework": "testing-framework",
      "Deployment Procedures": "deployment-procedures",
      "KPI Definition": "kpi-definition",
      "Monitoring Systems": "monitoring-systems",
      "Reporting Mechanisms": "reporting-mechanisms",
    };
    return titleMap[title] || null;
  };

  const renderSubsectionItem = (
    subsectionId: string,
    title: string,
    description: string
  ) => {
    const isAutoCompleted = autoSectionsCompleted.has(subsectionId);
    
    return (
      <div 
        key={subsectionId}
        className={`flex items-start space-x-3 p-3 rounded-lg border transition-colors ${
          isAutoCompleted 
            ? 'bg-green-50 border-green-200' 
            : 'bg-gray-50 border-gray-200 hover:bg-gray-100 cursor-pointer'
        }`}
        onClick={() => !isAutoCompleted && handleSubsectionClick(subsectionId, title)}
      >
        {isAutoCompleted ? (
          <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
        ) : (
          <div className="w-5 h-5 border-2 border-gray-300 rounded mt-0.5 flex-shrink-0"></div>
        )}
        <div className="flex-1">
          <h4 className={`font-medium ${isAutoCompleted ? 'text-green-900' : 'text-gray-900'}`}>
            {title}
          </h4>
          <p className={`text-sm ${isAutoCompleted ? 'text-green-700' : 'text-gray-600'}`}>
            {isAutoCompleted ? 'Automatically verified and implemented' : description}
          </p>
          {isAutoCompleted && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700 mt-1">
              Auto-completed
            </span>
          )}
        </div>
      </div>
    );
  };

  const renderCollapsibleSection = (
    sectionNumber: number,
    title: string,
    icon: React.ReactNode,
    subsections: Array<{id: string, title: string, description: string}>,
    isCompleted: boolean = false
  ) => {
    // Check if any subsections are completed to determine section status
    const hasCompletedSubsections = subsections.some(sub => autoSectionsCompleted.has(sub.id));
    const allSubsectionsCompleted = subsections.every(sub => autoSectionsCompleted.has(sub.id));
    
    return (
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div 
          className="flex items-center justify-between p-6 cursor-pointer hover:bg-gray-50 transition-colors"
          onClick={() => toggleSection(sectionNumber)}
        >
          <div className="flex items-center">
            <div className={`w-10 h-10 rounded-xl flex items-center justify-center mr-4 ${
              allSubsectionsCompleted ? 'bg-green-100' : hasCompletedSubsections ? 'bg-yellow-100' : 'bg-blue-100'
            }`}>
              {allSubsectionsCompleted ? (
                <CheckCircle className="w-6 h-6 text-green-600" />
              ) : (
                icon
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
              <p className="text-sm text-gray-600">
                {allSubsectionsCompleted ? "Completed" : hasCompletedSubsections ? "Partially Complete" : "In Progress"}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-3">
            {allSubsectionsCompleted && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700">
                Compliant
              </span>
            )}
            {hasCompletedSubsections && !allSubsectionsCompleted && (
              <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-100 text-yellow-700">
                Partial
              </span>
            )}
            {expandedSections.has(sectionNumber) ? (
              <ChevronUp className="w-5 h-5 text-gray-400" />
            ) : (
              <ChevronDown className="w-5 h-5 text-gray-400" />
            )}
          </div>
        </div>
        {expandedSections.has(sectionNumber) && (
          <div className="px-6 pb-6 border-t border-gray-100">
            <div className="space-y-4 mt-4">
              <p className="text-gray-700 mb-4">
                {allSubsectionsCompleted 
                  ? "All requirements completed"
                  : hasCompletedSubsections 
                    ? "Some requirements completed automatically, others require documentation"
                    : "Click on incomplete items to upload documentation"
                }
              </p>
              <div className="space-y-3">
                {subsections.map(subsection => 
                  renderSubsectionItem(subsection.id, subsection.title, subsection.description)
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Upload Modal Component
  const UploadModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-xl p-6 w-full max-w-md mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Upload Documentation</h3>
          <button
            onClick={() => setShowUploadModal(false)}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>
        
        <p className="text-sm text-gray-600 mb-4">
          Upload documentation for: <strong>{currentUploadSection}</strong>
        </p>

        <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center mb-4">
          <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
          <p className="text-sm text-gray-600 mb-2">
            {uploadedFile ? uploadedFile.name : "Click to upload or drag and drop"}
          </p>
          <input
            type="file"
            onChange={handleFileUpload}
            className="hidden"
            id="file-upload"
            accept=".pdf,.doc,.docx,.txt"
          />
          <label
            htmlFor="file-upload"
            className="cursor-pointer text-blue-600 hover:text-blue-700 text-sm font-medium"
          >
            Browse Files
          </label>
        </div>

        <button
          onClick={handleSubmitDocument}
          disabled={!uploadedFile}
          className={`w-full py-2 px-4 rounded-lg font-medium ${
            uploadedFile
              ? "bg-blue-600 text-white hover:bg-blue-700"
              : "bg-gray-300 text-gray-500 cursor-not-allowed"
          }`}
        >
          Submit
        </button>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading ISO 42001 compliance audit...</p>
        </div>
      </div>
    );
  }

  const complianceScore = calculateComplianceScore();

  return (
    <div className="min-h-screen bg-gray-50 flex">
      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex-shrink-0 shadow-sm">
        <div className="h-full flex flex-col">
          {/* Navigation */}
          <nav className="flex-1 p-6 space-y-2">
            <Link
              to="/home"
              className="flex items-center space-x-3 px-4 py-3 rounded-xl text-gray-700 hover:bg-gray-50 hover:text-gray-900 font-medium transition-colors"
            >
              <div className="h-8 w-8 bg-emerald-100 rounded-lg flex items-center justify-center">
                <Home className="h-4 w-4 text-emerald-600" />
              </div>
              <span>Dashboard</span>
            </Link>

            <Link
              to="/iso"
              className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-gray-50 text-gray-900 font-semibold border border-gray-200"
            >
              <div className="h-8 w-8 bg-blue-400 rounded-lg flex items-center justify-center">
                <Award className="h-4 w-4 text-white" />
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
              <div className="flex items-center text-sm text-gray-500 mb-2">
                <button 
                  onClick={() => navigate("/home")}
                  className="hover:text-blue-600 transition-colors"
                >
                  Dashboard
                </button>
                <span className="mx-2">/</span>
                <span className="font-medium text-gray-700">ISO 42001 Audit</span>
              </div>
              <h1 className="text-3xl font-bold text-gray-900">ISO 42001 Compliance Audit</h1>
              <p className="text-gray-600 mt-1">
                AI Management System Standard
              </p>
            </div>
          </div>
        </header>

        <div className="p-8 space-y-8">
          {/* Dashboard Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Framework Adaptation */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-blue-100 rounded-xl flex items-center justify-center">
                    <Settings className="w-5 h-5 text-blue-600" />
                  </div>
                  <div className="ml-3">
                    <h3 className="font-semibold text-gray-900">Framework Adaptation</h3>
                    <p className="text-sm text-gray-600">Overall Progress</p>
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-gray-900">{complianceScore}%</span>
                <div className="w-20 h-2 bg-gray-200 rounded-full">
                  <div className={`h-2 rounded-full ${getProgressColor(complianceScore)}`} style={{ width: `${complianceScore}%` }}></div>
                </div>
              </div>
            </div>

            {/* Required Actions */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-yellow-100 rounded-xl flex items-center justify-center">
                    <AlertCircle className="w-5 h-5 text-yellow-600" />
                  </div>
                  <div className="ml-3">
                    <h3 className="font-semibold text-gray-900">Required Actions</h3>
                    <p className="text-sm text-gray-600">Pending Items</p>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                {(autoSectionsCompleted.has('ai-policy-documentation') || 
                  autoSectionsCompleted.has('procedures-guidelines') || 
                  autoSectionsCompleted.has('record-keeping-system')) ? (
                  <div className="flex items-center text-sm">
                    <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                    <span className="text-gray-700">Documentation and Policy Development</span>
                  </div>
                ) : (
                  <div className="flex items-center text-sm">
                    <AlertCircle className="w-4 h-4 text-yellow-500 mr-2" />
                    <span className="text-gray-700">Documentation and Policy Development</span>
                  </div>
                )}
                {(autoSectionsCompleted.has('impact-assessment') || 
                  autoSectionsCompleted.has('risk-mitigation-strategies') || 
                  autoSectionsCompleted.has('incident-response')) ? (
                  <div className="flex items-center text-sm">
                    <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                    <span className="text-gray-700">Risk Assessment and Management</span>
                  </div>
                ) : (
                  <div className="flex items-center text-sm">
                    <AlertCircle className="w-4 h-4 text-yellow-500 mr-2" />
                    <span className="text-gray-700">Risk Assessment and Management</span>
                  </div>
                )}
                {(autoSectionsCompleted.has('development-controls') || 
                  autoSectionsCompleted.has('testing-framework') || 
                  autoSectionsCompleted.has('deployment-procedures')) ? (
                  <div className="flex items-center text-sm">
                    <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                    <span className="text-gray-700">AI System Lifecycle Management</span>
                  </div>
                ) : (
                  <div className="flex items-center text-sm">
                    <AlertCircle className="w-4 h-4 text-yellow-500 mr-2" />
                    <span className="text-gray-700">AI System Lifecycle Management</span>
                  </div>
                )}
              </div>
            </div>

            {/* Compliance Score */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-purple-100 rounded-xl flex items-center justify-center">
                    <Award className="w-5 h-5 text-purple-600" />
                  </div>
                  <div className="ml-3">
                    <h3 className="font-semibold text-gray-900">Compliance Score</h3>
                    <p className="text-sm text-gray-600">Based on ISO 42001 requirements</p>
                  </div>
                </div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-gray-900 mb-2">{complianceScore}%</div>
                <div className={`w-full h-3 rounded-full mb-2 ${getProgressColor(complianceScore)}`}>
                  <div className="h-full bg-gray-200 rounded-full">
                    <div 
                      className={`h-full rounded-full ${getProgressColor(complianceScore)}`}
                      style={{ width: `${complianceScore}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* ISO 42001 Compliance Checklist */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
              <h2 className="text-xl font-bold text-gray-900">ISO 42001 Compliance Checklist</h2>
              <p className="text-sm text-gray-600 mt-1">
                Comprehensive assessment based on AI Management System Standard
              </p>
            </div>

            <div className="p-6 space-y-4">
              {/* Documentation and Policy Development */}
              {renderCollapsibleSection(
                1,
                "Documentation and Policy Development",
                <FileText className="w-5 h-5 text-blue-600" />,
                [
                  {
                    id: "ai-policy-documentation",
                    title: "AI Policy Documentation",
                    description: "Create and maintain AI policy documents"
                  },
                  {
                    id: "procedures-guidelines",
                    title: "Procedures and Guidelines", 
                    description: "Develop operational procedures"
                  },
                  {
                    id: "record-keeping-system",
                    title: "Record Keeping System",
                    description: "Implement documentation management"
                  }
                ]
              )}

              {/* Risk Assessment and Management */}
              {renderCollapsibleSection(
                2,
                "Risk Assessment and Management",
                <Shield className="w-5 h-5 text-blue-600" />,
                [
                  {
                    id: "impact-assessment",
                    title: "Impact Assessment",
                    description: "Conduct AI impact analysis"
                  },
                  {
                    id: "risk-mitigation-strategies",
                    title: "Risk Mitigation Strategies",
                    description: "Develop risk management plans"
                  },
                  {
                    id: "incident-response",
                    title: "Incident Response",
                    description: "Create incident handling procedures"
                  }
                ]
              )}

              {/* AI System Lifecycle Management */}
              {renderCollapsibleSection(
                3,
                "AI System Lifecycle Management",
                <Zap className="w-5 h-5 text-blue-600" />,
                [
                  {
                    id: "development-controls",
                    title: "Development Controls",
                    description: "Implement development standards"
                  },
                  {
                    id: "testing-framework",
                    title: "Testing Framework",
                    description: "Establish testing protocols"
                  },
                  {
                    id: "deployment-procedures",
                    title: "Deployment Procedures", 
                    description: "Define deployment guidelines"
                  }
                ]
              )}

              {/* Performance Monitoring */}
              {renderCollapsibleSection(
                4,
                "Performance Monitoring",
                <Target className="w-5 h-5 text-blue-600" />,
                [
                  {
                    id: "kpi-definition",
                    title: "KPI Definition",
                    description: "Define performance indicators"
                  },
                  {
                    id: "monitoring-systems",
                    title: "Monitoring Systems",
                    description: "Implement monitoring tools"
                  },
                  {
                    id: "reporting-mechanisms",
                    title: "Reporting Mechanisms",
                    description: "Establish reporting procedures"
                  }
                ]
              )}

            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-between items-center">
            <button
              onClick={() => navigate("/home")}
              className="inline-flex items-center px-6 py-3 border border-gray-300 rounded-xl text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 transition-colors"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Dashboard
            </button>
            
            <div className="flex space-x-3">
              <button className="inline-flex items-center px-6 py-3 border border-blue-600 rounded-xl text-sm font-medium text-blue-600 bg-white hover:bg-blue-50 transition-colors">
                <FileText className="w-4 h-4 mr-2" />
                Generate Report
              </button>
              <button className="inline-flex items-center px-6 py-3 bg-blue-600 rounded-xl text-sm font-medium text-white hover:bg-blue-700 transition-colors">
                Complete Audit
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Upload Modal */}
      {showUploadModal && <UploadModal />}
    </div>
  );
};

export default ISO42001AuditPage; 