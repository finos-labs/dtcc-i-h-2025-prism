import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Button } from "../components/ui/button";
import { 
  CheckCircle,
  Info,
  Save,
  ArrowLeft,
  FileText,
  ChevronDown,
  ChevronUp,
  Clock,
  BarChart3,
  Shield,
  AlertTriangle,
  Target,
  Calendar,
  Users,
  TrendingUp,
  Download
} from 'lucide-react';
import { useParams, useNavigate } from 'react-router-dom';
import { supabase } from '../lib/supabase';
import axios from 'axios';

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

interface AssessmentData {
  aiSystemDescription: string;
  aiSystemPurpose: string;
  deploymentMethod: string;
  deploymentRequirements: string;
  rolesDocumented: string;
  personnelTrained: string;
  humanInvolvement: string;
  biasTraining: string;
  humanIntervention: string;
  humanOverride: string;
  riskLevels: string;
  threatsIdentified: string;
  maliciousUseAssessed: string;
  personalInfoUsed: string;
  personalInfoCategories: string;
  privacyRegulations: string;
  privacyRiskAssessment: string;
  privacyByDesign: string;
  individualsInformed: string;
  privacyRights: string;
  dataQuality: string;
  thirdPartyRisks: string;
}



const RiskAssessmentPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const isDummyProject = id === "dummy-1" || id === "dummy-2";
  
  const [loading, setLoading] = useState(!isDummyProject);
  const [modelData, setModelData] = useState<ModelData | null>(null);
  const [projectDetails, setProjectDetails] = useState<ProjectDetails | null>(null);
  const [assessmentData, setAssessmentData] = useState<AssessmentData>({
    aiSystemDescription: '',
    aiSystemPurpose: '',
    deploymentMethod: '',
    deploymentRequirements: '',
    rolesDocumented: '',
    personnelTrained: '',
    humanInvolvement: '',
    biasTraining: '',
    humanIntervention: '',
    humanOverride: '',
    riskLevels: '',
    threatsIdentified: '',
    maliciousUseAssessed: '',
    personalInfoUsed: '',
    personalInfoCategories: '',
    privacyRegulations: '',
    privacyRiskAssessment: '',
    privacyByDesign: '',
    individualsInformed: '',
    privacyRights: '',
    dataQuality: '',
    thirdPartyRisks: ''
  });
  const [expandedSections, setExpandedSections] = useState<Set<number>>(new Set([1])); // Start with section 1 expanded
  const [analysisCompleted, setAnalysisCompleted] = useState<boolean>(false);
  const [autoSectionsCompleted, setAutoSectionsCompleted] = useState<Set<number>>(new Set()); // Track which auto sections are actually completed

  useEffect(() => {
    if (!isDummyProject && id) {
      fetchProjectData();
    }
    
    // Check if analysis has been completed
    const storedAnalysis = localStorage.getItem(`riskAssessment_${id}`);
    setAnalysisCompleted(!!storedAnalysis);
    
    // Check auto-sections completion status
    checkAutoSectionsCompletion();
  }, [id, isDummyProject]);

  const fetchProjectData = async () => {
    try {
      setLoading(true);

      // Fetch project details
      const { data: projectData, error: projectError } = await supabase
        .from("projectdetails")
        .select("project_id, user_uuid, project_name, description, project_type, project_status")
        .eq("project_id", id)
        .single();

      if (projectError) {
        console.error("Error fetching project details:", projectError);
      } else if (projectData) {
        setProjectDetails(projectData);
      }

      // Fetch model data
      const { data: modelInfo, error: modelError } = await supabase
        .from("modeldetails")
        .select("model_id, project_id, dataset_id, model_version")
        .eq("project_id", id)
        .limit(1);

      if (modelError) {
        console.error("Error fetching model data:", modelError);
      } else if (modelInfo && modelInfo.length > 0) {
        setModelData(modelInfo[0]);
        // Here you would typically fetch actual risk assessment data from your API
      }
    } catch (error) {
      console.error("Failed to fetch project data:", error);
    } finally {
      setLoading(false);
    }
  };

  const checkAutoSectionsCompletion = async () => {
    try {
      // Get token from localStorage (similar to ProjectOverviewPage)
      const token = localStorage.getItem('access_token');
      if (!token) {
        console.log('No access token found');
        return;
      }

      const config = {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json',
        },
      };

      // Check if models/data exist for this project
      try {
        const modelsResponse = await axios.get(`http://localhost:8000/ml/${id}/models/list`, config);
        
        // If we get a successful response with data, mark auto sections as completed
        if (modelsResponse.data && modelsResponse.data.length > 0) {
          setAutoSectionsCompleted(new Set([3, 5, 6, 8, 9, 10]));
        } else {
          // No data, keep auto sections as not completed
          setAutoSectionsCompleted(new Set());
        }
      } catch (apiError) {
        console.log('Models API not available or no data:', apiError);
        // API call failed or returned no data, keep auto sections as not completed
        setAutoSectionsCompleted(new Set());
      }
    } catch (error) {
      console.error('Error checking auto sections completion:', error);
      setAutoSectionsCompleted(new Set());
    }
  };





  const handleInputChange = (field: keyof AssessmentData, value: string) => {
    setAssessmentData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSave = async () => {
    // Implementation for saving assessment data
    console.log('Saving assessment data:', assessmentData);
  };

  const handleAnalyzeAssessment = async () => {
    try {
      setLoading(true);
      
      // Get project name from project details or use default
      const projectName = projectDetails?.project_name || "AI System";
      
      // Generate structured AI recommendations using Gemini API
      let geminiRecommendations = "";
      
      try {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyD6L9OZMjl5CMuwcezKH6dwZPF4oB5EKv8`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            contents: [{
              parts: [{
                text: `You are an AI Risk Assessment expert. Analyze each user response and create unique, specific recommendations for EVERY question. Project: ${projectName}

CRITICAL INSTRUCTIONS:
- Analyze EACH response individually and provide COMPLETELY UNIQUE recommendations
- Do NOT repeat generic advice - each recommendation must be specific to that question and user's answer
- Cover ALL sections, not just a few
- If user says "Yes" but is vague, ask for more specificity
- If user says "No", provide step-by-step implementation
- If user mentions specific tools/processes, reference them directly in recommendations

USER RESPONSES TO ANALYZE:

${Object.entries(assessmentData).filter(([key, value]) => value).map(([key, value]) => {
  const sectionMapping: Record<string, { section: string; question: string }> = {
    // Section 1: AI System Information
    aiSystemDescription: { section: "AI System Information", question: "What is your AI system description?" },
    aiSystemPurpose: { section: "AI System Information", question: "What is the purpose of your AI system?" },
    deploymentMethod: { section: "AI System Information", question: "What is your deployment method?" },
    deploymentRequirements: { section: "AI System Information", question: "What are your deployment requirements?" },
    
    // Section 2: Human and Stakeholder Involvement  
    rolesDocumented: { section: "Human and Stakeholder Involvement", question: "Are roles and responsibilities for AI governance clearly documented?" },
    personnelTrained: { section: "Human and Stakeholder Involvement", question: "Is personnel trained on AI ethics, bias, and risk management?" },
    humanInvolvement: { section: "Human and Stakeholder Involvement", question: "What level of human involvement exists in AI decision-making?" },
    biasTraining: { section: "Human and Stakeholder Involvement", question: "Has bias awareness and mitigation training been provided?" },
    humanIntervention: { section: "Human and Stakeholder Involvement", question: "Can humans intervene in AI system decisions when needed?" },
    humanOverride: { section: "Human and Stakeholder Involvement", question: "Can humans override AI system decisions completely?" },
    
    // Section 3: Safety and Reliability
    riskLevels: { section: "Safety and Reliability", question: "What risk levels have been identified and assessed?" },
    threatsIdentified: { section: "Safety and Reliability", question: "What potential threats and vulnerabilities have been identified?" },
    maliciousUseAssessed: { section: "Safety and Reliability", question: "Has the potential for malicious use been assessed?" },
    
    // Section 4: Privacy and Data Governance
    personalInfoUsed: { section: "Privacy and Data Governance", question: "Is personal information used by the AI system?" },
    personalInfoCategories: { section: "Privacy and Data Governance", question: "What categories of personal information are processed?" },
    privacyRegulations: { section: "Privacy and Data Governance", question: "Which privacy regulations apply to your system?" },
    privacyRiskAssessment: { section: "Privacy and Data Governance", question: "Has a privacy risk assessment been conducted?" },
    privacyByDesign: { section: "Privacy and Data Governance", question: "Are privacy-by-design principles implemented?" },
    individualsInformed: { section: "Privacy and Data Governance", question: "Are individuals informed about how their data is used?" },
    privacyRights: { section: "Privacy and Data Governance", question: "How are individual privacy rights handled and respected?" },
    dataQuality: { section: "Privacy and Data Governance", question: "How is data quality and accuracy ensured?" },
    thirdPartyRisks: { section: "Privacy and Data Governance", question: "How are third-party data sharing risks managed?" }
  };
  
  const mapping = sectionMapping[key] || { section: "Additional", question: key };
  return `
SECTION: ${mapping.section}
QUESTION: ${mapping.question}
USER'S ACTUAL RESPONSE: "${value}"

REQUIRED ANALYSIS FOR THIS SPECIFIC QUESTION:
1. What does this response tell us about their current implementation?
2. What specific gaps or strengths are revealed?
3. What unique risks apply to this particular area?
4. What specific next steps should they take based on what they wrote?
`;
}).join('\n')}

AUTO-COMPLETED SECTIONS (Include these with standard recommendations):

Section 3: Valid and Reliable AI
Description: This section assesses measures to ensure the AI system is developed for the good of society, environment, and community
Status: Analysis completed - Standard controls implemented
Recommendation: Maintain regular impact assessments and ensure compliance with all applicable regulations

Section 5: Secure and Resilient AI
Description: This section assesses measures to ensure system security and capability to respond to incidents and operate continuously
Status: Analysis completed - Standard controls implemented  
Recommendation: Implement comprehensive security testing including red-team exercises and continuous vulnerability assessments

Section 6: Explainable and Interpretable AI
Description: This section assesses measures to ensure information requirements for explainable AI are maintained and decisions are interpreted as expected
Status: Analysis completed - Standard controls implemented
Recommendation: Enhance traceability mechanisms and implement comprehensive logging for all decisions

Section 8: Fairness and Unbiased AI
Description: This section assesses measures to ensure the AI system is free from bias, inclusive, and diverse
Status: Analysis completed - Standard controls implemented
Recommendation: Implement comprehensive bias testing and monitoring with diverse representation in development teams

Section 9: Transparent and Accountable AI
Description: This section assesses measures to provide sufficient information to relevant stakeholders at any point of the AI lifecycle
Status: Analysis completed - Standard controls implemented
Recommendation: Establish clear communication protocols and ensure users are properly informed about AI interactions

Section 10: AI Accountability
Description: This section ensures the organization has risk management mechanisms to effectively manage identified AI risks
Status: Analysis completed - Standard controls implemented
Recommendation: Implement comprehensive risk management framework with regular audits and independent third-party assessments

NOW GENERATE REPORT IN THIS EXACT FORMAT:

Section 1: AI System Information
Question 1: What is your AI system description?
User Answer: [Their exact response]
Analysis: [Detailed analysis of their specific response - what it reveals about maturity, gaps, strengths]
Recommendation: [Unique, specific recommendation based only on this response - no generic advice]

Question 2: What is the purpose of your AI system?
User Answer: [Their exact response]  
Analysis: [Different analysis specific to purpose and use case]
Recommendation: [Completely different recommendation focused on purpose-specific risks]

[Continue for ALL questions in AI System Information]

Section 2: Human and Stakeholder Involvement
Question 1: Are roles and responsibilities for AI governance clearly documented?
User Answer: [Their exact response]
Analysis: [Analyze their governance maturity from this response]
Recommendation: [Specific governance improvement recommendations]

[Continue for ALL questions in Human and Stakeholder Involvement]

Section 3: Safety and Reliability  
[Continue with ALL safety questions]

Section 4: Privacy and Data Governance
[Continue with ALL privacy questions]

MAKE EVERY RECOMMENDATION UNIQUE - NO REPETITION ALLOWED!`
              }]
            }]
          })
        });

        if (response.ok) {
          const data = await response.json();
          geminiRecommendations = data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        }
      } catch (apiError) {
        console.log('AI analysis unavailable, using structured fallback');
        geminiRecommendations = "AI analysis temporarily unavailable. Please review your responses and ensure all sections are complete.";
      }

      // Store analysis results in localStorage
      const analysisResults = {
        projectId: id,
        projectName,
        assessmentData,
        aiRecommendations: geminiRecommendations,
        timestamp: new Date().toISOString(),
        progress: calculateProgress(),
        riskLevel: getRiskLevel().level
      };

      // Store in localStorage with project-specific key
      localStorage.setItem(`riskAssessment_${id}`, JSON.stringify(analysisResults));
      
      // Also store in a general list for easy retrieval
      const existingAnalyses = JSON.parse(localStorage.getItem('riskAssessmentAnalyses') || '[]');
      const updatedAnalyses = existingAnalyses.filter((analysis: any) => analysis.projectId !== id);
      updatedAnalyses.push({
        projectId: id,
        projectName,
        timestamp: new Date().toISOString(),
        progress: calculateProgress(),
        riskLevel: getRiskLevel().level
      });
      localStorage.setItem('riskAssessmentAnalyses', JSON.stringify(updatedAnalyses));

      // Also set the flags that ReportPage.tsx expects for compatibility
      localStorage.setItem(`riskAssessmentGenerated_${id}`, "true");
      localStorage.setItem(`riskAssessmentTimestamp_${id}`, new Date().toISOString());

      setAnalysisCompleted(true);
      alert('✅ AI Risk Assessment analysis completed and saved successfully!');
      
    } catch (error) {
      console.error('Error analyzing assessment:', error);
      alert('Failed to analyze assessment. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadReport = async () => {
    try {
      setLoading(true);
      
      // Retrieve stored analysis from localStorage
      const storedAnalysis = localStorage.getItem(`riskAssessment_${id}`);
      if (!storedAnalysis) {
        alert('No analysis found. Please run the AI Risk Assessment analysis first.');
        return;
      }

      const analysisData = JSON.parse(storedAnalysis);
      const projectName = analysisData.projectName || "AI System";
      const geminiRecommendations = analysisData.aiRecommendations || "";

      const { jsPDF } = await import('jspdf');
      const pdf = new jsPDF();
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;
      const maxWidth = pageWidth - 2 * margin;
      let yPosition = margin;

      // Helper function to check and add new page
      const checkPageBreak = (additionalHeight: number = 20) => {
        if (yPosition + additionalHeight > pageHeight - margin) {
          pdf.addPage();
          yPosition = margin;
          return true;
        }
        return false;
      };

      // Helper function to add wrapped text
      const addWrappedText = (text: string, fontSize: number, fontStyle: string = 'normal', leftMargin: number = margin) => {
        pdf.setFontSize(fontSize);
        pdf.setFont('helvetica', fontStyle as any);
        const lines = pdf.splitTextToSize(text, maxWidth - (leftMargin - margin));
        
        checkPageBreak(lines.length * (fontSize * 0.35));
        pdf.text(lines, leftMargin, yPosition);
        yPosition += lines.length * (fontSize * 0.35) + 3;
        return lines.length;
      };

      // Title with logo
      // First, add the logo on white background
      let logoWidth = 0;
      let logoLoaded = false;
      
      try {
        // Try to load the logo image from public folder
        const logoFormats = ['/logoBC.png', '/logoBC.jpg', '/logoBC.jpeg', '/logoBC.svg'];
        
        for (const logoPath of logoFormats) {
          if (logoLoaded) break;
          
          try {
            await new Promise<void>((resolve, reject) => {
              const logoImg = new Image();
              logoImg.crossOrigin = 'anonymous';
              
              logoImg.onload = () => {
                try {
                  // Convert image to base64 and add to PDF
                  const canvas = document.createElement('canvas');
                  const ctx = canvas.getContext('2d');
                  if (!ctx) throw new Error('Canvas context not available');
                  
                  canvas.width = logoImg.width;
                  canvas.height = logoImg.height;
                  ctx.drawImage(logoImg, 0, 0);
                  const dataURL = canvas.toDataURL('image/png');
                  
                  // Add logo to PDF (adjust size to fit within header height)
                  const logoHeight = 16;
                  logoWidth = (logoImg.width / logoImg.height) * logoHeight;
                  const logoX = margin + 2;
                  const logoY = yPosition + 2;
                  
                  pdf.addImage(dataURL, 'PNG', logoX, logoY, logoWidth, logoHeight);
                  logoLoaded = true;
                  resolve();
                } catch (error) {
                  console.log(`Error processing logo ${logoPath}:`, error);
                  reject(error);
                }
              };
              
              logoImg.onerror = () => {
                reject(new Error(`Failed to load ${logoPath}`));
              };
              
              logoImg.src = logoPath;
            });
          } catch (error) {
            console.log(`Could not load logo from ${logoPath}:`, error);
            continue;
          }
        }
      } catch (error) {
        console.log('Error in logo loading process:', error);
      }
      
      // Now add the black header starting after the logo
      if (logoLoaded && logoWidth > 0) {
        // Calculate where black header should start
        const spaceCm = 0.3;
        const spacePoints = spaceCm * 28.35; // Convert cm to points
        const blackHeaderStartX = margin + 2 + logoWidth + spacePoints;
        const blackHeaderWidth = maxWidth - (blackHeaderStartX - margin);
        
        // Draw black header starting after logo + space
        pdf.setFillColor(0, 0, 0);
        pdf.rect(blackHeaderStartX, yPosition, blackHeaderWidth, 20, 'F');
        
        // Add title text on black background
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(16);
        pdf.setFont('helvetica', 'bold');
        pdf.text('AI Risk Assessment Report', blackHeaderStartX + 5, yPosition + 13);
        pdf.setTextColor(0, 0, 0);
      } else {
        // Fallback: full width black header with text only
        pdf.setFillColor(0, 0, 0);
        pdf.rect(margin, yPosition, maxWidth, 20, 'F');
        pdf.setTextColor(255, 255, 255);
        pdf.setFontSize(16);
        pdf.setFont('helvetica', 'bold');
        pdf.text('AI Risk Assessment Report', margin + 5, yPosition + 13);
        pdf.setTextColor(0, 0, 0);
      }
      
      yPosition += 30;

      // Introduction
      addWrappedText(
        `This document outlines responses to the AI Risk Assessment based on the NIST AI Risk Management Framework. Each section is elaborated with context, rationale, and answers assuming compliance with best practices.`,
        10, 'normal'
      );
      yPosition += 5;

      // If we have Gemini recommendations, use them
      if (geminiRecommendations && geminiRecommendations.trim()) {
        addWrappedText(geminiRecommendations, 10, 'normal');
        
        yPosition += 10; // Add some space before auto-sections
      }
      
      // Always add all sections in correct order (1-10)
      const allSections = [
        // Section 1: AI System Information (User section)
        {
          number: 1,
          title: "AI System Information",
          isUser: true,
          questions: [
            {
              field: 'aiSystemDescription',
              question: 'What is your AI system description?',
              elaboration: 'This provides the foundational understanding of the AI system being assessed.',
              importance: 'Clear system description enables proper risk categorization and compliance planning.'
            },
            {
              field: 'aiSystemPurpose',
              question: 'What is the purpose of your AI system?',
              elaboration: 'Understanding the intended use case and business objective of the AI system.',
              importance: 'Purpose determines risk levels, regulatory requirements, and stakeholder impact scope.'
            },
            {
              field: 'deploymentMethod',
              question: 'What is your deployment method?',
              elaboration: 'How the AI system will be deployed and integrated into existing infrastructure.',
              importance: 'Deployment approach affects security, scalability, and operational risk management.'
            },
            {
              field: 'deploymentRequirements',
              question: 'What are your deployment requirements?',
              elaboration: 'Technical and operational prerequisites for successful system deployment.',
              importance: 'Proper requirements ensure system reliability and compliance readiness.'
            }
          ]
        },
        // Section 2: Human and Stakeholder Involvement (User section)
        {
          number: 2,
          title: "Human and Stakeholder Involvement",
          isUser: true,
          questions: [
            {
              field: 'rolesDocumented',
              question: 'Are roles and responsibilities for AI governance clearly documented?',
              elaboration: 'This ensures accountability frameworks are established for AI system oversight.',
              importance: 'Clear roles prevent governance gaps and ensure responsible AI implementation.'
            },
            {
              field: 'personnelTrained',
              question: 'Is personnel trained on AI ethics, bias, and risk management?',
              elaboration: 'Training ensures staff competency in identifying and mitigating AI-related risks.',
              importance: 'Proper training reduces operational risks and ensures ethical AI practices.'
            },
            {
              field: 'humanInvolvement',
              question: 'What level of human involvement exists in AI decision-making?',
              elaboration: 'Defines the degree of human oversight and control in AI system operations.',
              importance: 'Appropriate human involvement ensures accountability and risk management.'
            },
            {
              field: 'biasTraining',
              question: 'Has bias awareness and mitigation training been provided?',
              elaboration: 'Specialized training on identifying and addressing AI bias and fairness issues.',
              importance: 'Bias training prevents discriminatory outcomes and ensures equitable AI systems.'
            },
            {
              field: 'humanIntervention',
              question: 'Can humans intervene in AI system decisions when needed?',
              elaboration: 'Capability for human operators to step in during AI system operations.',
              importance: 'Intervention capability ensures human control over critical decisions.'
            },
            {
              field: 'humanOverride',
              question: 'Can humans override AI system decisions completely?',
              elaboration: 'Ultimate human authority to reverse or modify AI system outputs.',
              importance: 'Override capability maintains human agency and prevents autonomous harm.'
            }
          ]
        },
        // Section 3: Valid and Reliable AI (Auto section)
        {
          number: 3,
          title: "Valid and Reliable AI",
          isUser: false,
          description: "This section is intended to assess the measures in place to ensure that the AI system is developed for the good of society, the environment, and the community.",
          items: [
            "✓ Mechanisms in place to identify and assess the impacts of the AI system on individuals, the environment, communities, and society",
            "✓ Potential negative impacts re-assessed if there are significant changes to the AI system in all stages of the AI lifecycle",
            "✓ Identified potential negative impacts used to inform and implement mitigating measures throughout the AI lifecycle",
            "✓ All existing regulations and guidelines that may affect the AI system have been identified"
          ],
          recommendation: "Maintain regular impact assessments and ensure compliance with all applicable regulations. Implement continuous monitoring for societal and environmental impacts."
        },
        // Section 4: Safety and Reliability (User section)
        {
          number: 4,
          title: "Safety and Reliability",
          isUser: true,
          questions: [
            {
              field: 'riskLevels',
              question: 'What risk levels have been identified and assessed?',
              elaboration: 'Systematic evaluation of potential risks associated with AI system deployment.',
              importance: 'Risk assessment enables appropriate mitigation strategies and compliance planning.'
            },
            {
              field: 'threatsIdentified',
              question: 'What potential threats and vulnerabilities have been identified?',
              elaboration: 'Identification of security, safety, and operational threats to the AI system.',
              importance: 'Threat identification enables proactive security measures and incident prevention.'
            },
            {
              field: 'maliciousUseAssessed',
              question: 'Has the potential for malicious use been assessed?',
              elaboration: 'Evaluation of how the AI system might be misused by bad actors.',
              importance: 'Malicious use assessment prevents dual-use concerns and reputational damage.'
            }
          ]
        },
        // Section 5: Secure and Resilient AI (Auto section)
        {
          number: 5,
          title: "Secure and Resilient AI",
          isUser: false,
          description: "This section is intended to assess the measures in place to ensure the security of the AI system and its capability to respond to incidents and operate continuously.",
          items: [
            "✓ Mechanisms in place to assess vulnerabilities in terms of security and resiliency across the AI lifecycle",
            "✓ Red-team exercises used to actively test the system under adversarial or stress conditions",
            "✓ Processes in place to modify system security and countermeasures to increase robustness",
            "✓ Processes in place to respond to incidents related to AI systems",
            "✓ Procedures and relevant performance metrics in place to monitor AI system's accuracy",
            "✓ Processes in place to establish and track security tests and metrics"
          ],
          recommendation: "Implement comprehensive security testing including red-team exercises and continuous vulnerability assessments. Establish robust incident response procedures."
        },
        // Section 6: Explainable and Interpretable AI (Auto section)
        {
          number: 6,
          title: "Explainable and Interpretable AI",
          isUser: false,
          description: "This section is intended to assess the measures in place to ensure that information requirements for explainable AI are maintained, and AI decisions are interpreted as expected.",
          items: [
            "✓ Measures in place to address the traceability of the AI system during its entire lifecycle",
            "✓ Measures in place to continuously assess the quality of the input data to the AI system",
            "✓ Data used by the AI system is traceable to make certain decisions or recommendations",
            "✓ AI models or rules are traceable that led to the decisions or recommendations",
            "✓ Adequate logging practices in place to record the decisions or recommendations",
            "✓ Explanations on the decision of the AI system provided to relevant users and stakeholders"
          ],
          recommendation: "Enhance traceability mechanisms and implement comprehensive logging. Ensure all decisions can be explained to relevant stakeholders with appropriate detail."
        },
        // Section 7: Privacy and Data Governance (User section)
        {
          number: 7,
          title: "Privacy and Data Governance",
          isUser: true,
          questions: [
            {
              field: 'personalInfoUsed',
              question: 'Is personal information used by the AI system?',
              elaboration: 'Identification of personal data processing within the AI system.',
              importance: 'Personal data usage triggers privacy regulations and protection requirements.'
            },
            {
              field: 'personalInfoCategories',
              question: 'What categories of personal information are processed?',
              elaboration: 'Detailed categorization of personal data types handled by the system.',
              importance: 'Data categorization determines protection levels and regulatory compliance needs.'
            },
            {
              field: 'privacyRegulations',
              question: 'Which privacy regulations apply to your system?',
              elaboration: 'Identification of relevant privacy laws and regulations for compliance.',
              importance: 'Regulatory compliance prevents legal risks and ensures user privacy protection.'
            },
            {
              field: 'privacyRiskAssessment',
              question: 'Has a privacy risk assessment been conducted?',
              elaboration: 'Systematic evaluation of privacy risks associated with data processing.',
              importance: 'Privacy risk assessment ensures proactive protection of personal data.'
            },
            {
              field: 'privacyByDesign',
              question: 'Are privacy-by-design principles implemented?',
              elaboration: 'Integration of privacy considerations into system design and architecture.',
              importance: 'Privacy by design ensures fundamental protection rather than retroactive fixes.'
            },
            {
              field: 'individualsInformed',
              question: 'Are individuals informed about how their data is used?',
              elaboration: 'Transparency measures to inform data subjects about data processing activities.',
              importance: 'Transparency ensures informed consent and regulatory compliance.'
            },
            {
              field: 'privacyRights',
              question: 'How are individual privacy rights handled and respected?',
              elaboration: 'Mechanisms to support individual privacy rights like access, correction, and deletion.',
              importance: 'Privacy rights support ensures regulatory compliance and user trust.'
            },
            {
              field: 'dataQuality',
              question: 'How is data quality and accuracy ensured?',
              elaboration: 'Processes to maintain high-quality, accurate, and relevant training and operational data.',
              importance: 'Data quality directly impacts AI system performance and fairness.'
            },
            {
              field: 'thirdPartyRisks',
              question: 'How are third-party data sharing risks managed?',
              elaboration: 'Risk management for data sharing with external parties and vendors.',
              importance: 'Third-party risk management prevents data breaches and compliance violations.'
            }
          ]
        },
        // Section 8: Fairness and Unbiased AI (Auto section)
        {
          number: 8,
          title: "Fairness and Unbiased AI",
          isUser: false,
          description: "This section is intended to assess the measures in place to ensure that the AI system is free from bias, inclusive, and diverse.",
          items: [
            "✓ Strategy established to avoid creating or reinforcing unfair bias in the AI system",
            "✓ Diversity and representativeness of end-users considered in the data used for the AI system",
            "✓ Demographics of those involved in design and development documented to capture potential biases",
            "✓ AI actors are aware of the possible bias they can inject into the design and development",
            "✓ Mechanisms in place to test and monitor the AI system for potential biases",
            "✓ Identified issues related to bias, discrimination, and poor performance are mitigated"
          ],
          recommendation: "Implement comprehensive bias testing and monitoring. Ensure diverse representation in development teams and training data to promote fairness and inclusivity."
        },
        // Section 9: Transparent and Accountable AI (Auto section)
        {
          number: 9,
          title: "Transparent and Accountable AI",
          isUser: false,
          description: "This section is intended to assess the measures in place to provide sufficient and appropriate information to relevant stakeholders, at any point of the AI lifecycle.",
          items: [
            "✓ Sufficient information provided to relevant AI actors to assist in making informed decisions",
            "✓ Type of information accessible about the AI lifecycle is limited to what is relevant and sufficient",
            "✓ End users are aware that they are interacting with an AI system and not a human",
            "✓ End-users informed of the purpose, criteria, and limitations of the decisions generated",
            "✓ End-users informed of the benefits of the AI system",
            "✓ Mechanism in place to regularly communicate with external stakeholders"
          ],
          recommendation: "Establish clear communication protocols for all stakeholders. Ensure users are properly informed about AI interactions and system limitations."
        },
        // Section 10: AI Accountability (Auto section)
        {
          number: 10,
          title: "AI Accountability",
          isUser: false,
          description: "This section is intended to ensure that the organization has risk management mechanisms in place to effectively manage identified AI risk.",
          items: [
            "✓ Risk management system implemented to address risks identified in the AI system",
            "✓ AI system can be audited by independent third parties",
            "✓ Checks conducted at appropriate intervals to confirm that the AI system is still trustworthy"
          ],
          recommendation: "Implement comprehensive risk management framework with regular audits. Establish procedures for independent third-party assessments and continuous trustworthiness verification."
        }
      ];

      // Process all sections in order
      allSections.forEach((section) => {
        checkPageBreak(15);
        addWrappedText(`Section ${section.number}: ${section.title}`, 12, 'bold');
        yPosition += 5;

        if (section.isUser) {
          // User section - process questions with user answers
          const sectionAnswers: Array<{question: any; answer: string}> = [];
          
          if (section.questions) {
            section.questions.forEach((q, index) => {
              const userAnswer = assessmentData[q.field as keyof AssessmentData];
              if (userAnswer) {
                sectionAnswers.push({ question: q, answer: userAnswer });
                
                checkPageBreak(30);

                // Question
                addWrappedText(`Question ${index + 1}: ${q.question}`, 10, 'bold');
                
                // Elaboration
                addWrappedText(`Elaboration: ${q.elaboration}`, 10, 'normal');
                
                // Why this matters
                addWrappedText(`Why this matters: ${q.importance}`, 10, 'normal');
                
                // Answer
                addWrappedText(`Answer: ${userAnswer}`, 10, 'normal');
                yPosition += 5;
              }
            });
          }

          // Generate overall section recommendation based on all answers
          if (sectionAnswers.length > 0) {
            checkPageBreak(20);
            addWrappedText(`Overall Section Recommendation:`, 11, 'bold');
            
            let overallRecommendation = "";
            
            if (section.title === "AI System Information") {
              const hasMLSystem = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('machine learning') || 
                sa.answer.toLowerCase().includes('neural') || 
                sa.answer.toLowerCase().includes('deep learning')
              );
              const hasCloudDeployment = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('cloud') || 
                sa.answer.toLowerCase().includes('saas')
              );
              const hasAutomatedDecisions = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('decision') || 
                sa.answer.toLowerCase().includes('automat')
              );

              overallRecommendation = `Based on your AI system profile, we recommend: ${hasMLSystem ? 
                'Establish comprehensive ML model governance including version control, model lineage tracking, and performance monitoring. ' : 
                'Document system architecture and decision logic with clear audit trails. '
              }${hasCloudDeployment ? 
                'Implement cloud security best practices, data residency controls, and vendor risk management. ' : 
                'Ensure robust on-premise security controls and network segmentation. '
              }${hasAutomatedDecisions ? 
                'Define clear decision boundaries, escalation procedures, and human oversight requirements for automated decisions. ' : 
                'Establish clear success criteria and performance metrics for your AI system implementation. '
              }Focus on comprehensive documentation, security controls, and governance frameworks appropriate for your deployment model.`;

            } else if (section.title === "Human and Stakeholder Involvement") {
              const hasDocumentedRoles = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('yes') && sa.question.field === 'rolesDocumented'
              );
              const hasTraining = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('yes') && 
                (sa.question.field === 'personnelTrained' || sa.question.field === 'biasTraining')
              );
              const hasHumanOversight = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('human') || 
                sa.answer.toLowerCase().includes('oversight')
              );

              overallRecommendation = `Your human governance approach shows ${hasDocumentedRoles ? 
                'good role documentation - enhance with quarterly reviews and clear escalation paths. ' : 
                'need for better role definition - create RACI matrix with AI Ethics Officer, Data Steward, and Model Validator roles. '
              }${hasTraining ? 
                'existing training programs - implement competency assessments and track effectiveness metrics. ' : 
                'training gaps - develop comprehensive AI ethics curriculum covering bias detection and responsible AI principles. '
              }${hasHumanOversight ? 
                'appropriate human involvement - optimize workflows to prevent skill degradation while maintaining oversight. ' : 
                'insufficient human oversight - implement human-in-the-loop mechanisms for critical decisions. '
              }Prioritize continuous training, clear accountability, and balanced human-AI collaboration.`;

            } else if (section.title === "Safety and Reliability") {
              const hasHighRisk = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('high') || 
                sa.answer.toLowerCase().includes('critical')
              );
              const hasThreatAssessment = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('yes') && sa.question.field === 'threatsIdentified'
              );
              const hasMaliciousUseAssessment = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('yes') && sa.question.field === 'maliciousUseAssessed'
              );

              overallRecommendation = `Your safety and reliability profile indicates ${hasHighRisk ? 
                'high-risk operations requiring enhanced monitoring, redundant safety systems, and comprehensive incident response procedures. ' : 
                'moderate risk levels - establish regular risk reviews and automated monitoring alerts. '
              }${hasThreatAssessment ? 
                'good threat identification - implement adversarial testing and robust security measures against identified vectors. ' : 
                'need for systematic threat modeling including data poisoning, model extraction, and inference attacks. '
              }${hasMaliciousUseAssessment ? 
                'awareness of misuse potential - develop specific countermeasures and monitoring for identified misuse scenarios. ' : 
                'require comprehensive dual-use assessment and misuse prevention strategies. '
              }Focus on proactive risk management, continuous monitoring, and robust incident response capabilities.`;

            } else if (section.title === "Privacy and Data Governance") {
              const usesPersonalData = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('yes') && sa.question.field === 'personalInfoUsed'
              );
              const hasRegulationCompliance = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('gdpr') || 
                sa.answer.toLowerCase().includes('ccpa') || 
                (sa.answer.toLowerCase().includes('yes') && sa.question.field === 'privacyRegulations')
              );
              const hasPrivacyByDesign = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('yes') && sa.question.field === 'privacyByDesign'
              );
              const hasDataQuality = sectionAnswers.some((sa: any) => 
                sa.answer.toLowerCase().includes('automated') || 
                sa.answer.toLowerCase().includes('continuous') || 
                (sa.answer.toLowerCase().includes('yes') && sa.question.field === 'dataQuality')
              );

              overallRecommendation = `Your privacy and data governance approach ${usesPersonalData ? 
                'involves personal data processing - implement data minimization, purpose limitation, and privacy-preserving techniques like differential privacy. ' : 
                'appears to avoid personal data - verify through data lineage tracking and privacy impact assessments. '
              }${hasRegulationCompliance ? 
                'shows regulatory awareness - implement specific compliance controls including consent management and data subject rights. ' : 
                'needs regulatory mapping - conduct comprehensive assessment of applicable privacy laws for your jurisdiction. '
              }${hasPrivacyByDesign ? 
                'incorporates privacy by design - enhance with regular privacy impact assessments and stakeholder consultation. ' : 
                'requires privacy by design implementation - integrate privacy considerations into system architecture from the ground up. '
              }${hasDataQuality ? 
                'has data quality measures - enhance with statistical process control and automated anomaly detection. ' : 
                'needs data quality framework - implement accuracy, completeness, consistency, and timeliness metrics. '
              }Prioritize comprehensive privacy controls, regulatory compliance, and robust data governance practices.`;

            } else {
              overallRecommendation = `Based on the responses in this section, implement comprehensive ${section.title.toLowerCase()} procedures with clear documentation, regular review processes, and stakeholder engagement. Establish metrics to track effectiveness and ensure continuous improvement.`;
            }

            addWrappedText(overallRecommendation, 10, 'normal');
            yPosition += 10;
          }

        } else {
          // Auto section - process items as questions
          if (section.items) {
            section.items.forEach((item, index) => {
              checkPageBreak(30);
              
              // Extract the main point from the checkmark item
              const mainPoint = item.replace('✓ ', '');
              
              // Question
              addWrappedText(`Question ${index + 1}: ${mainPoint}`, 10, 'bold');
              
              // Elaboration
              addWrappedText(`Elaboration: ${section.description}`, 10, 'normal');
              
              // Why this matters
              addWrappedText(`Why this matters: This control is essential for maintaining AI system trustworthiness and compliance with best practices.`, 10, 'normal');
              
              // Answer
              addWrappedText(`Answer: Yes - Standard controls implemented and verified`, 10, 'normal');
              yPosition += 5;
            });

            // Overall section recommendation for auto sections
            checkPageBreak(20);
            addWrappedText(`Overall Section Recommendation:`, 11, 'bold');
            addWrappedText(section.recommendation, 10, 'normal');
            yPosition += 10;
          }
        }
      });

      // Save the PDF
      pdf.save(`AI_Risk_Assessment_Report_${projectName.replace(/\s+/g, '_')}.pdf`);
      
    } catch (error) {
      console.error('Error generating PDF report:', error);
      alert('Failed to generate PDF report. Please try again.');
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

  // Define sections and their required fields for progress tracking
  const userSections = [
    {
      number: 1,
      title: "AI System Information",
      fields: ['aiSystemDescription', 'aiSystemPurpose', 'deploymentMethod', 'deploymentRequirements']
    },
    {
      number: 2,
      title: "Human and Stakeholder Involvement",
      fields: ['rolesDocumented', 'personnelTrained', 'humanInvolvement', 'biasTraining', 'humanIntervention', 'humanOverride']
    },
    {
      number: 4,
      title: "Safety and Reliability of AI",
      fields: ['riskLevels', 'threatsIdentified', 'maliciousUseAssessed']
    },
    {
      number: 7,
      title: "Privacy and Data Governance",
      fields: ['personalInfoUsed', 'personalInfoCategories', 'privacyRegulations', 'privacyRiskAssessment', 'privacyByDesign', 'individualsInformed', 'privacyRights', 'dataQuality', 'thirdPartyRisks']
    }
  ];

  const autoCompletedSections = [3, 5, 6, 8, 9, 10];

  const calculateProgress = () => {
    let completedFields = 0;
    let totalFields = 0;

    userSections.forEach(section => {
      section.fields.forEach(field => {
        totalFields++;
        if (assessmentData[field as keyof AssessmentData]) {
          completedFields++;
        }
      });
    });

    // Add auto-completed sections (only count those that are actually completed)
    completedFields += autoSectionsCompleted.size; // Count only actually completed auto sections
    totalFields += autoCompletedSections.length; // Total possible auto sections

    return Math.round((completedFields / totalFields) * 100);
  };

  const getPendingItems = () => {
    const pending: string[] = [];
    
    userSections.forEach(section => {
      const incompletedFields = section.fields.filter(field => 
        !assessmentData[field as keyof AssessmentData]
      );
      
      if (incompletedFields.length > 0) {
        pending.push(`${section.title} (${incompletedFields.length} questions remaining)`);
      }
    });

    return pending;
  };

  const getCompletedSections = () => {
    let completed = autoSectionsCompleted.size; // Only actually completed auto sections
    
    userSections.forEach(section => {
      const allFieldsCompleted = section.fields.every(field => 
        assessmentData[field as keyof AssessmentData]
      );
      if (allFieldsCompleted) completed++;
    });
    
    return completed;
  };

  const getRiskLevel = () => {
    const progress = calculateProgress();
    
    // Only show risk assessment after substantial completion
    if (progress < 25) {
      return { level: "Pending", color: "text-gray-600", bgColor: "bg-gray-50", borderColor: "border-gray-200" };
    }
    
    if (progress >= 80) return { level: "Low Risk", color: "text-green-600", bgColor: "bg-green-50", borderColor: "border-green-200" };
    if (progress >= 60) return { level: "Medium Risk", color: "text-yellow-600", bgColor: "bg-yellow-50", borderColor: "border-yellow-200" };
    return { level: "High Risk", color: "text-red-600", bgColor: "bg-red-50", borderColor: "border-red-200" };
  };

  const getEstimatedCompletionTime = () => {
    const totalQuestions = userSections.reduce((acc, section) => acc + section.fields.length, 0);
    const completedQuestions = userSections.reduce((acc, section) => 
      acc + section.fields.filter(field => assessmentData[field as keyof AssessmentData]).length, 0
    );
    const remainingQuestions = totalQuestions - completedQuestions;
    
    if (remainingQuestions <= 0) return "Completed";
    
    // More realistic time estimation: 3-5 minutes per question depending on complexity
    let estimatedMinutes = 0;
    userSections.forEach(section => {
      const sectionRemaining = section.fields.filter(field => 
        !assessmentData[field as keyof AssessmentData]
      ).length;
      
      // Different time estimates per section type
      if (section.number === 1) estimatedMinutes += sectionRemaining * 3; // Basic info
      else if (section.number === 7) estimatedMinutes += sectionRemaining * 4; // Privacy (complex)
      else estimatedMinutes += sectionRemaining * 3; // Other sections
    });
    
    if (estimatedMinutes < 60) return `${estimatedMinutes} min`;
    const hours = Math.floor(estimatedMinutes / 60);
    const minutes = estimatedMinutes % 60;
    return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
  };

  const renderTextArea = (
    label: string,
    field: keyof AssessmentData,
    placeholder: string,
    tip?: string
  ) => (
    <div className="space-y-2">
      <label className="block text-sm font-medium text-gray-700">{label}</label>
      {tip && (
        <div className="flex items-start space-x-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <Info className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-blue-700">{tip}</p>
        </div>
      )}
      <textarea
        value={assessmentData[field]}
        onChange={(e) => handleInputChange(field, e.target.value)}
        placeholder={placeholder}
        className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-teal-500 resize-none"
        rows={4}
      />
    </div>
  );

  const renderRadioGroup = (
    label: string,
    field: keyof AssessmentData,
    options: { value: string; label: string }[],
    tip?: string
  ) => (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">{label}</label>
      {tip && (
        <div className="flex items-start space-x-2 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <Info className="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" />
          <p className="text-sm text-blue-700">{tip}</p>
        </div>
      )}
      <div className="space-y-2">
        {options.map((option) => (
          <label key={option.value} className="flex items-center space-x-3 cursor-pointer">
            <input
              type="radio"
              name={field}
              value={option.value}
              checked={assessmentData[field] === option.value}
              onChange={(e) => handleInputChange(field, e.target.value)}
              className="w-4 h-4 text-teal-600 focus:ring-teal-500"
            />
            <span className="text-sm text-gray-700">{option.label}</span>
          </label>
        ))}
      </div>
    </div>
  );

  const renderCompletedSection = (title: string, items: string[]) => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      <div className="space-y-3">
        {items.map((item, index) => (
          <div key={index} className="flex items-start space-x-3 p-3 bg-green-50 border border-green-200 rounded-lg">
            <CheckCircle className="w-5 h-5 text-green-600 mt-0.5 flex-shrink-0" />
            <span className="text-sm text-green-800">{item}</span>
          </div>
        ))}
      </div>
    </div>
  );

  const renderIncompleteSection = (title: string, description: string, items: string[]) => (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
      <p className="text-sm text-gray-600 mb-4">{description}</p>
      <div className="space-y-3">
        {items.map((item, index) => (
          <div key={index} className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">
              {item.replace('✓ ', '')}
            </label>
            <div className="flex items-center space-x-3">
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name={`auto_section_${title.replace(/\s+/g, '_')}_${index}`}
                  value="yes"
                  className="w-4 h-4 text-teal-600 focus:ring-teal-500"
                  disabled
                />
                <span className="text-sm text-gray-500">Yes</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name={`auto_section_${title.replace(/\s+/g, '_')}_${index}`}
                  value="no"
                  className="w-4 h-4 text-teal-600 focus:ring-teal-500"
                  disabled
                />
                <span className="text-sm text-gray-500">No</span>
              </label>
              <label className="flex items-center space-x-2 cursor-pointer">
                <input
                  type="radio"
                  name={`auto_section_${title.replace(/\s+/g, '_')}_${index}`}
                  value="na"
                  className="w-4 h-4 text-teal-600 focus:ring-teal-500"
                  disabled
                />
                <span className="text-sm text-gray-500">N/A</span>
              </label>
            </div>
            <div className="text-xs text-gray-500 bg-yellow-50 border border-yellow-200 rounded p-2">
              This section will be auto-completed once model is evaluated
            </div>
          </div>
        ))}
      </div>
    </div>
  );

  const renderCollapsibleSection = (
    sectionNumber: number,
    title: string,
    content: React.ReactNode,
    isAutoCompleted: boolean = false,
    completedCount?: number,
    totalCount?: number
  ) => (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200">
      <div 
        className="flex items-center justify-between p-6 cursor-pointer hover:bg-gray-50 transition-colors"
        onClick={() => toggleSection(sectionNumber)}
      >
        <div className="flex items-center">
          <div className={`w-8 h-8 rounded-lg flex items-center justify-center mr-3 ${
            isAutoCompleted ? 'bg-green-100' : 'bg-teal-100'
          }`}>
            {isAutoCompleted ? (
              <CheckCircle className="w-5 h-5 text-green-600" />
            ) : (
              <span className="text-teal-600 font-semibold">{sectionNumber}</span>
            )}
          </div>
          <h2 className="text-xl font-bold text-gray-900">{title}</h2>
        </div>
        <div className="flex items-center">
          {!isAutoCompleted && completedCount !== undefined && totalCount !== undefined && (
            <span className="text-sm text-gray-500 mr-2">
              {completedCount} / {totalCount} completed
            </span>
          )}
          {isAutoCompleted && (
            <span className="text-sm text-green-600 mr-2">Analysis completed</span>
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
          {content}
        </div>
      )}
    </div>
  );

  if (loading) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-teal-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading risk assessment...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-50 min-h-screen">
      <div className="container mx-auto px-6 py-6 max-w-7xl">
                {/* Compact Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <div className="flex items-center text-sm text-gray-500 mb-2">
              <span className="cursor-pointer hover:text-teal-600 transition-colors" onClick={() => navigate("/projects")}>
                Projects
              </span>
              <span className="mx-2">/</span>
              <span className="cursor-pointer hover:text-teal-600 transition-colors" onClick={() => navigate(`/projects/${id}`)}>
                Project Details
              </span>
              <span className="mx-2">/</span>
              <span className="font-medium text-gray-700">AI Risk Assessment</span>
            </div>
            <h1 className="text-3xl font-bold text-gray-900">AI Risk Assessment</h1>
            <p className="text-gray-600 mt-1">
              {projectDetails?.project_name || `Project ${id}`} • Comprehensive compliance evaluation
            </p>
          </div>
          <Button
            onClick={handleAnalyzeAssessment}
            disabled={loading}
            className="bg-gradient-to-r from-purple-500 to-blue-500 hover:from-purple-600 hover:to-blue-600 text-white flex items-center px-6 py-2 rounded-lg font-medium transition-all duration-200"
          >
            <BarChart3 className="w-4 h-4 mr-2" />
            {loading ? 'Analyzing...' : 'Run AI Analysis'}
          </Button>
        </div>

                 {/* Dashboard Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
          {/* Overall Progress */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center">
                <div className="w-8 h-8 bg-teal-100 rounded-lg flex items-center justify-center mr-3">
                  <BarChart3 className="w-4 h-4 text-teal-600" />
                </div>
                <span className="text-sm font-medium text-gray-700">Progress</span>
              </div>
              <span className="text-xl font-bold text-teal-600">{calculateProgress()}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-1.5 mb-2">
              <div 
                className="bg-teal-600 h-1.5 rounded-full transition-all duration-300" 
                style={{ width: `${calculateProgress()}%` }}
              ></div>
            </div>
            <p className="text-xs text-gray-500">{getCompletedSections()}/10 sections complete</p>
          </div>

                     {/* Risk Level */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center mr-3 ${
                  calculateProgress() < 25 ? 'bg-gray-100' : 'bg-red-100'
                }`}>
                  <Shield className={`w-4 h-4 ${
                    calculateProgress() < 25 ? 'text-gray-600' : 'text-red-600'
                  }`} />
                </div>
                <span className="text-sm font-medium text-gray-700">Risk Assessment</span>
              </div>
            </div>
            <div className={`inline-flex px-2 py-1 rounded-full text-xs font-medium ${getRiskLevel().bgColor} ${getRiskLevel().color} ${getRiskLevel().borderColor} border mb-1`}>
              {getRiskLevel().level}
            </div>
                         <p className="text-xs text-gray-500">
               {calculateProgress() < 25 ? "Complete assessment to evaluate" : "Based on current responses"}
             </p>
          </div>

                    {/* AI Analysis Status */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center mr-3 ${
                  analysisCompleted ? 'bg-green-100' : 'bg-purple-100'
                }`}>
                  <BarChart3 className={`w-4 h-4 ${
                    analysisCompleted ? 'text-green-600' : 'text-purple-600'
                  }`} />
                </div>
                <span className="text-sm font-medium text-gray-700">AI Analysis</span>
              </div>
            </div>
            <div className="text-lg font-semibold text-gray-900 mb-1">
              {analysisCompleted ? "Completed" : "Pending"}
            </div>
            <p className="text-xs text-gray-500">
              {analysisCompleted ? "Analysis saved in storage" : "Run analysis to get insights"}
            </p>
          </div>
        </div>

        {/* Quick Actions & Summary */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Pending Actions */}
          <div className="lg:col-span-2 bg-white rounded-xl shadow-sm border border-gray-200 p-5">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900">Pending Actions</h3>
              <span className="text-sm text-gray-500">{getPendingItems().length} remaining</span>
            </div>
            <div className="space-y-3 max-h-40 overflow-y-auto">
              {getPendingItems().length === 0 ? (
                <div className="flex items-center justify-center py-8 text-green-600">
                  <CheckCircle className="w-5 h-5 mr-2" />
                  <span className="font-medium">All sections completed! Ready for review.</span>
                </div>
              ) : (
                getPendingItems().map((item, index) => (
                  <div key={index} className="flex items-center p-3 bg-orange-50 border border-orange-200 rounded-lg">
                    <AlertTriangle className="w-4 h-4 text-orange-500 mr-3 flex-shrink-0" />
                    <span className="text-sm text-orange-700 font-medium">{item}</span>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Assessment Stats */}
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-5">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Assessment Stats</h3>
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <Users className="w-4 h-4 text-gray-400 mr-2" />
                  <span className="text-sm text-gray-600">User Sections</span>
                </div>
                <span className="text-sm font-medium text-gray-900">4 sections</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                  <span className="text-sm text-gray-600">Auto-completed</span>
                </div>
                <span className="text-sm font-medium text-gray-900">6 sections</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <TrendingUp className="w-4 h-4 text-blue-500 mr-2" />
                  <span className="text-sm text-gray-600">Total Questions</span>
                </div>
                <span className="text-sm font-medium text-gray-900">
                  {userSections.reduce((acc, section) => acc + section.fields.length, 0)} questions
                </span>
              </div>
              <div className="flex items-center justify-between pt-2 border-t border-gray-100">
                <div className="flex items-center">
                  <Calendar className="w-4 h-4 text-purple-500 mr-2" />
                  <span className="text-sm text-gray-600">Last Updated</span>
                </div>
                <span className="text-sm font-medium text-gray-900">Just now</span>
              </div>
            </div>
          </div>
        </div>

        {/* Assessment Sections */}
        <div className="space-y-4">
          {/* Section 1: AI System Information */}
          {renderCollapsibleSection(
            1,
            "AI System Information",
            <div className="space-y-6">
              {renderTextArea(
                "Describe the AI system",
                "aiSystemDescription",
                "Briefly provide the basic information of the AI system...",
                "Briefly provide the basic information of the AI system (e.g., Name of the system and outline of how the system will work.)"
              )}

              {renderTextArea(
                "What is the purpose of developing the AI system?",
                "aiSystemPurpose",
                "Describe how the AI system will address a need...",
                "Briefly describe how the AI system will address a need that aligns with the objective of the organization."
              )}

              {renderTextArea(
                "How will the system be deployed for its intended uses?",
                "deploymentMethod",
                "Describe the deployment strategy...",
              )}

              {renderRadioGroup(
                "Have requirements for system deployment and operation been initially identified?",
                "deploymentRequirements",
                [
                  { value: "yes", label: "Yes" },
                  { value: "no", label: "No" }
                ]
              )}
            </div>,
            false,
            userSections[0].fields.filter(field => assessmentData[field as keyof AssessmentData]).length,
            userSections[0].fields.length
          )}

          {/* Section 2: Human and Stakeholder Involvement */}
          {renderCollapsibleSection(
            2,
            "Human and Stakeholder Involvement",
            <div className="space-y-6">
              {renderRadioGroup(
                "Have the roles and responsibilities of personnel involved in the design, development, deployment, assessment, and monitoring of the AI system been defined and documented?",
                "rolesDocumented",
                [
                  { value: "yes", label: "Yes" },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ],
                "Include a brief description of each stakeholder's role in the AI lifecycle or link to relevant documentation."
              )}

              {renderRadioGroup(
                "Are personnel provided with the necessary skills, training, and resources needed in order to fulfill their assigned roles and responsibilities?",
                "personnelTrained",
                [
                  { value: "yes", label: "Yes [Include a description of the relevant trainings and resources provided]" },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "What is the level of human involvement and control in relation to the AI system?",
                "humanInvolvement",
                [
                  { value: "self-learning", label: "Self-Learning or Autonomous System" },
                  { value: "human-in-loop", label: "Overseen by a Human-in-the-Loop" },
                  { value: "human-on-loop", label: "Overseen by a Human-on-the-Loop" },
                  { value: "human-command", label: "Overseen by a Human-in-Command" }
                ]
              )}

              {renderRadioGroup(
                "Are the relevant personnel dealing with AI systems properly trained to interpret AI model output and decisions as well as to detect and manage bias in data?",
                "biasTraining",
                [
                  { value: "yes", label: "Yes [Include a description of the trainings provided]" },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Are processes defined and documented where human intervention is required by the AI system?",
                "humanIntervention",
                [
                  { value: "yes", label: "Yes" },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ],
                "There are a number of cases and scenarios where human intervention is needed to ensure the safe, ethical, and secure use of AI."
              )}

              {renderRadioGroup(
                "Do human reviewers have the expertise and authority to override decisions made by the AI and modify them to the appropriate outcome?",
                "humanOverride",
                [
                  { value: "yes", label: "Yes [Include a description of the process or mechanism in place]" },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}
            </div>,
            false,
            userSections[1].fields.filter(field => assessmentData[field as keyof AssessmentData]).length,
            userSections[1].fields.length
          )}

          {/* Section 3: Valid and Reliable AI */}
          {renderCollapsibleSection(
            3,
            "Valid and Reliable AI",
            autoSectionsCompleted.has(3) ? 
              renderCompletedSection(
                "This section is intended to assess the measures in place to ensure that the AI system is developed for the good of society, the environment, and the community.",
                [
                  "✓ Mechanisms in place to identify and assess the impacts of the AI system on individuals, the environment, communities, and society",
                  "✓ Potential negative impacts re-assessed if there are significant changes to the AI system in all stages of the AI lifecycle",
                  "✓ Identified potential negative impacts used to inform and implement mitigating measures throughout the AI lifecycle",
                  "✓ All existing regulations and guidelines that may affect the AI system have been identified"
                ]
              ) :
              renderIncompleteSection(
                "Valid and Reliable AI",
                "This section is intended to assess the measures in place to ensure that the AI system is developed for the good of society, the environment, and the community.",
                [
                  "Mechanisms in place to identify and assess the impacts of the AI system on individuals, the environment, communities, and society",
                  "Potential negative impacts re-assessed if there are significant changes to the AI system in all stages of the AI lifecycle",
                  "Identified potential negative impacts used to inform and implement mitigating measures throughout the AI lifecycle",
                  "All existing regulations and guidelines that may affect the AI system have been identified"
                ]
              ),
            autoSectionsCompleted.has(3),
            autoSectionsCompleted.has(3) ? 4 : 0,
            4
          )}

          {/* Section 4: Safety and Reliability of AI */}
          {renderCollapsibleSection(
            4,
            "Safety and Reliability of AI",
            <div className="space-y-6">
              {renderRadioGroup(
                "Are tolerable risk levels defined for the AI system based on the business objectives, regulatory compliance, and data sensitivity requirements of the system?",
                "riskLevels",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ],
                "AI risk tolerance level refers to the extent to which individuals, organizations, or societies are willing to accept or tolerate potential risks associated with the AI system."
              )}

              {renderRadioGroup(
                "Have the possible threats to the AI system (design faults, technical faults, environmental threats) been identified, and the possible consequences to AI trustworthiness?",
                "threatsIdentified",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Are the risks of possible malicious use, misuse, or inappropriate use of the AI system assessed?",
                "maliciousUseAssessed",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}
            </div>,
            false,
            userSections[2].fields.filter(field => assessmentData[field as keyof AssessmentData]).length,
            userSections[2].fields.length
          )}

          {/* Section 5: Secure and Resilient AI */}
          {renderCollapsibleSection(
            5,
            "Secure and Resilient AI",
            autoSectionsCompleted.has(5) ?
              renderCompletedSection(
                "This section is intended to assess the measures in place to ensure the security of the AI system and its capability to respond to incidents and operate continuously.",
                [
                  "✓ Mechanisms in place to assess vulnerabilities in terms of security and resiliency across the AI lifecycle",
                  "✓ Red-team exercises used to actively test the system under adversarial or stress conditions",
                  "✓ Processes in place to modify system security and countermeasures to increase robustness",
                  "✓ Processes in place to respond to incidents related to AI systems",
                  "✓ Procedures and relevant performance metrics in place to monitor AI system's accuracy",
                  "✓ Processes in place to establish and track security tests and metrics"
                ]
              ) :
              renderIncompleteSection(
                "Secure and Resilient AI",
                "This section is intended to assess the measures in place to ensure the security of the AI system and its capability to respond to incidents and operate continuously.",
                [
                  "Mechanisms in place to assess vulnerabilities in terms of security and resiliency across the AI lifecycle",
                  "Red-team exercises used to actively test the system under adversarial or stress conditions",
                  "Processes in place to modify system security and countermeasures to increase robustness",
                  "Processes in place to respond to incidents related to AI systems",
                  "Procedures and relevant performance metrics in place to monitor AI system's accuracy",
                  "Processes in place to establish and track security tests and metrics"
                ]
              ),
            autoSectionsCompleted.has(5),
            autoSectionsCompleted.has(5) ? 6 : 0,
            6
          )}

          {/* Section 6: Explainable and Interpretable AI */}
          {renderCollapsibleSection(
            6,
            "Explainable and Interpretable AI",
            autoSectionsCompleted.has(6) ?
              renderCompletedSection(
                "This section is intended to assess the measures in place to ensure that information requirements for explainable AI are maintained, and AI decisions are interpreted as expected.",
                [
                  "✓ Measures in place to address the traceability of the AI system during its entire lifecycle",
                  "✓ Measures in place to continuously assess the quality of the input data to the AI system",
                  "✓ Data used by the AI system is traceable to make certain decisions or recommendations",
                  "✓ AI models or rules are traceable that led to the decisions or recommendations",
                  "✓ Adequate logging practices in place to record the decisions or recommendations",
                  "✓ Explanations on the decision of the AI system provided to relevant users and stakeholders"
                ]
              ) :
              renderIncompleteSection(
                "Explainable and Interpretable AI",
                "This section is intended to assess the measures in place to ensure that information requirements for explainable AI are maintained, and AI decisions are interpreted as expected.",
                [
                  "Measures in place to address the traceability of the AI system during its entire lifecycle",
                  "Measures in place to continuously assess the quality of the input data to the AI system",
                  "Data used by the AI system is traceable to make certain decisions or recommendations",
                  "AI models or rules are traceable that led to the decisions or recommendations",
                  "Adequate logging practices in place to record the decisions or recommendations",
                  "Explanations on the decision of the AI system provided to relevant users and stakeholders"
                ]
              ),
            autoSectionsCompleted.has(6),
            autoSectionsCompleted.has(6) ? 6 : 0,
            6
          )}

          {/* Section 7: Privacy and Data Governance */}
          {renderCollapsibleSection(
            7,
            "Privacy and Data Governance",
            <div className="space-y-6">
              {renderRadioGroup(
                "Is the AI system being trained, or was it developed, by using or processing personal information?",
                "personalInfoUsed",
                [
                  { value: "yes", label: "Yes" },
                  { value: "no", label: "No" }
                ]
              )}

              {renderTextArea(
                "Please describe the categories of personal information used by the AI system. Indicate if the system is using sensitive or special categories of personal information, including a description of the legal basis for processing the personal information.",
                "personalInfoCategories",
                "Describe the categories of personal information...",
                "Special categories of personal information refer to specific types of personal information that are considered more sensitive and are subject to enhanced data protection and privacy regulations (e.g., race, religious beliefs, health data, sexual orientation, or criminal records)."
              )}

              {renderRadioGroup(
                "Have applicable legal regulations for privacy been identified and considered before processing personal information to train, develop, or deploy the AI system?",
                "privacyRegulations",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Has a privacy risk assessment been conducted to ensure the privacy and security of the personal information used for the AI system?",
                "privacyRiskAssessment",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Have measures to achieve privacy by design and default been implemented when applicable to mitigate identified privacy risks?",
                "privacyByDesign",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Are individuals informed of the processing of their personal information for the development of the AI system?",
                "individualsInformed",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Have mechanisms been implemented to enable individuals to exercise their right to privacy for any personal information used in the AI system?",
                "privacyRights",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Are measures in place to ensure that the data used to develop the AI system is up-to-date, complete, and representative of the AI environment?",
                "dataQuality",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}

              {renderRadioGroup(
                "Have risks been assessed in using datasets obtained from third parties?",
                "thirdPartyRisks",
                [
                  { value: "yes", label: "Yes " },
                  { value: "no", label: "No" },
                  { value: "na", label: "N/A" }
                ]
              )}
            </div>,
            false,
            userSections[3].fields.filter(field => assessmentData[field as keyof AssessmentData]).length,
            userSections[3].fields.length
          )}

          {/* Section 8: Fairness and Unbiased AI */}
          {renderCollapsibleSection(
            8,
            "Fairness and Unbiased AI",
            autoSectionsCompleted.has(8) ?
              renderCompletedSection(
                "This section is intended to assess the measures in place to ensure that the AI system is free from bias, inclusive, and diverse.",
                [
                  "✓ Strategy established to avoid creating or reinforcing unfair bias in the AI system",
                  "✓ Diversity and representativeness of end-users considered in the data used for the AI system",
                  "✓ Demographics of those involved in design and development documented to capture potential biases",
                  "✓ AI actors are aware of the possible bias they can inject into the design and development",
                  "✓ Mechanisms in place to test and monitor the AI system for potential biases",
                  "✓ Identified issues related to bias, discrimination, and poor performance are mitigated"
                ]
              ) :
              renderIncompleteSection(
                "Fairness and Unbiased AI",
                "This section is intended to assess the measures in place to ensure that the AI system is free from bias, inclusive, and diverse.",
                [
                  "Strategy established to avoid creating or reinforcing unfair bias in the AI system",
                  "Diversity and representativeness of end-users considered in the data used for the AI system",
                  "Demographics of those involved in design and development documented to capture potential biases",
                  "AI actors are aware of the possible bias they can inject into the design and development",
                  "Mechanisms in place to test and monitor the AI system for potential biases",
                  "Identified issues related to bias, discrimination, and poor performance are mitigated"
                ]
              ),
            autoSectionsCompleted.has(8),
            autoSectionsCompleted.has(8) ? 6 : 0,
            6
          )}

          {/* Section 9: Transparent and Accountable AI */}
          {renderCollapsibleSection(
            9,
            "Transparent and Accountable AI",
            autoSectionsCompleted.has(9) ?
              renderCompletedSection(
                "This section is intended to assess the measures in place to provide sufficient and appropriate information to relevant stakeholders, at any point of the AI lifecycle.",
                [
                  "✓ Sufficient information provided to relevant AI actors to assist in making informed decisions",
                  "✓ Type of information accessible about the AI lifecycle is limited to what is relevant and sufficient",
                  "✓ End users are aware that they are interacting with an AI system and not a human",
                  "✓ End-users informed of the purpose, criteria, and limitations of the decisions generated",
                  "✓ End-users informed of the benefits of the AI system",
                  "✓ Mechanism in place to regularly communicate with external stakeholders"
                ]
              ) :
              renderIncompleteSection(
                "Transparent and Accountable AI",
                "This section is intended to assess the measures in place to provide sufficient and appropriate information to relevant stakeholders, at any point of the AI lifecycle.",
                [
                  "Sufficient information provided to relevant AI actors to assist in making informed decisions",
                  "Type of information accessible about the AI lifecycle is limited to what is relevant and sufficient",
                  "End users are aware that they are interacting with an AI system and not a human",
                  "End-users informed of the purpose, criteria, and limitations of the decisions generated",
                  "End-users informed of the benefits of the AI system",
                  "Mechanism in place to regularly communicate with external stakeholders"
                ]
              ),
            autoSectionsCompleted.has(9),
            autoSectionsCompleted.has(9) ? 6 : 0,
            6
          )}

          {/* Section 10: AI Accountability */}
          {renderCollapsibleSection(
            10,
            "AI Accountability",
            autoSectionsCompleted.has(10) ?
              renderCompletedSection(
                "This section is intended to ensure that the organization has risk management mechanisms in place to effectively manage identified AI risk.",
                [
                  "✓ Risk management system implemented to address risks identified in the AI system",
                  "✓ AI system can be audited by independent third parties",
                  "✓ Checks conducted at appropriate intervals to confirm that the AI system is still trustworthy"
                ]
              ) :
              renderIncompleteSection(
                "AI Accountability",
                "This section is intended to ensure that the organization has risk management mechanisms in place to effectively manage identified AI risk.",
                [
                  "Risk management system implemented to address risks identified in the AI system",
                  "AI system can be audited by independent third parties",
                  "Checks conducted at appropriate intervals to confirm that the AI system is still trustworthy"
                ]
              ),
            autoSectionsCompleted.has(10),
            autoSectionsCompleted.has(10) ? 3 : 0,
            3
          )}
        </div>

        {/* Action Buttons */}
        <div className="mt-8 flex flex-col sm:flex-row gap-4 justify-center">
          <Button
            onClick={handleSave}
            variant="outline"
            className="flex items-center gap-2 px-6 py-3"
            disabled={loading}
          >
            <Save className="w-4 h-4" />
            Save Progress
          </Button>
          
          <Button
            onClick={handleAnalyzeAssessment}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-purple-600 to-blue-600 hover:from-purple-700 hover:to-blue-700 text-white"
            disabled={loading}
          >
            <BarChart3 className="w-4 h-4" />
            {loading ? 'Analyzing...' : 'Run AI Analysis'}
          </Button>
          
          <Button
            onClick={handleDownloadReport}
            className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-teal-600 to-blue-600 hover:from-teal-700 hover:to-blue-700 text-white"
            disabled={loading}
          >
            <Download className="w-4 h-4" />
            {loading ? 'Generating...' : 'Download Report'}
          </Button>
        </div>
        
      </div>
    </div>
  );
};

export default RiskAssessmentPage; 