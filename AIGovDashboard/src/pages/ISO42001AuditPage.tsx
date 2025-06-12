import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  CheckCircle,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Award,
  Home,
  Upload,
  X,
  Settings,
  FileText,
  Shield,
  Zap,
  Target,
} from "lucide-react";
import { Link } from "react-router-dom";
import axios from "axios";

const ISO42001AuditPage: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [expandedSections, setExpandedSections] = useState<Set<number>>(
    new Set()
  );
  const [completedSections, setCompletedSections] = useState<Set<number>>(
    new Set()
  );
  const [autoSectionsCompleted, setAutoSectionsCompleted] = useState<
    Set<string>
  >(new Set());
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [currentUploadSection, setCurrentUploadSection] = useState<string>("");
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResults, setAnalysisResults] = useState<{
    score: number;
    explanation: string;
    recommendations: string[];
  } | null>(null);

  useEffect(() => {
    checkAutoSectionsCompletion();
  }, []);

  const checkAutoSectionsCompletion = async () => {
    try {
      setLoading(true);

      // Check if risk assessment was completed (from ReportPage)
      const riskAssessmentCompleted = localStorage.getItem("riskassessment") === "true";
      
      if (riskAssessmentCompleted) {
        console.log("Risk assessment completed, auto-completing related subsections");
        // Auto-complete risk assessment related subsections
        setAutoSectionsCompleted(
          new Set([
            "impact-assessment",
            "testing-framework",
            "monitoring-systems",
            "reporting-mechanisms",
          ])
        );
        // Update completed sections to reflect which main sections have any completed subsections
        setCompletedSections(new Set([2, 3, 4])); // Risk Assessment, AI System Lifecycle, Performance Monitoring
        setLoading(false);
        return;
      }

      // Get token from localStorage
      const token = localStorage.getItem("access_token");
      if (!token) {
        console.log("No access token found");
        setLoading(false);
        return;
      }

      const config = {
        headers: {
          Authorization: `Bearer ${token}`,
          "Content-Type": "application/json",
        },
      };

      // Check if models/data exist for any project (since this is a general audit page)
      // We'll check for a dummy project or use a default approach
      try {
        // Try to check for models - if successful, auto-complete certain subsections
        const modelsResponse = await axios.get(
          `https://prism-backend-dtcc-dot-block-convey-p1.uc.r.appspot.com/ml/dummy-1/models/list`,
          config
        );

        // Only mark subsections as auto-completed if we get a successful response with actual data
        if (
          modelsResponse.data &&
          Array.isArray(modelsResponse.data) &&
          modelsResponse.data.length > 0
        ) {
          console.log(
            "Models found, auto-completing subsections:",
            modelsResponse.data
          );
          // Auto-complete specific subsections when models exist
          setAutoSectionsCompleted(
            new Set([
              "impact-assessment",
              "risk-mitigation-strategies",
              "testing-framework",
              "kpi-definition",
              "monitoring-systems",
              "reporting-mechanisms",
            ])
          );
          // Update completed sections to reflect which main sections have any completed subsections
          setCompletedSections(new Set([1, 2, 3, 4]));
        } else {
          console.log("No model data found, all sections remain manual");
          // No model data, all sections remain manual (clickable for upload)
          setAutoSectionsCompleted(new Set());
          setCompletedSections(new Set());
        }
      } catch (apiError) {
        console.log(
          "Models API call failed, all sections remain manual:",
          apiError
        );
        // API call failed, all sections remain manual (clickable for upload)
        setAutoSectionsCompleted(new Set());
        setCompletedSections(new Set());
      }
    } catch (error) {
      console.error("Error checking auto sections completion:", error);
      setAutoSectionsCompleted(new Set());
      setCompletedSections(new Set());
    } finally {
      setLoading(false);
    }
  };

  const toggleSection = (sectionNumber: number) => {
    setExpandedSections((prev) => {
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
    // Calculate based on subsections completed out of total 12 subsections
    const totalSubsections = 12; // 3 subsections per section × 4 sections
    return Math.round((autoSectionsCompleted.size / totalSubsections) * 100);
  };

  const getProgressColor = (progress: number) => {
    if (progress >= 80) return "bg-green-500";
    if (progress >= 60) return "bg-yellow-500";
    if (progress >= 40) return "bg-orange-500";
    return "bg-red-500";
  };

  const handleSubsectionClick = (
    subsectionId: string,
    subsectionTitle: string
  ) => {
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

  const handleSubmitDocument = async () => {
    if (uploadedFile) {
      console.log(
        `Uploading document for ${currentUploadSection}:`,
        uploadedFile
      );

      // ✅ NEW - Perform AI validation for specific sections
      if (currentUploadSection === "AI Policy Documentation") {
        await analyzeAIPolicyDocument();
      } else if (currentUploadSection === "Procedures and Guidelines") {
        await analyzeProceduresAndGuidelines();
      } else if (currentUploadSection === "Record Keeping System") {
        await analyzeRecordKeepingSystem();
      } else if (currentUploadSection === "Risk Mitigation Strategies") {
        await analyzeRiskMitigationStrategies();
      } else if (currentUploadSection === "Development Controls") {
        await analyzeDevelopmentControls();
      } else if (currentUploadSection === "Deployment Procedures") {
        await analyzeDeploymentProcedures();
      } else {
        // For other documents, mark as completed immediately
        const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
        if (subsectionId) {
          setAutoSectionsCompleted((prev) => new Set([...prev, subsectionId]));
        }

        alert(`Document uploaded successfully for ${currentUploadSection}!`);
        setShowUploadModal(false);
        setUploadedFile(null);
        setCurrentUploadSection("");
        setAnalysisResults(null);
      }
    }
  };

  // ✅ NEW - Document Relevance Validation Function
  const checkDocumentRelevance = async (
    documentText: string,
    sectionType: string
  ): Promise<{ isRelevant: boolean; reason: string }> => {
    try {
      const relevancePrompt = `You are a document classification expert. Analyze the following document and determine if it is relevant for the specified section type.

SECTION TYPE: ${sectionType}

EXPECTED CONTENT FOR EACH SECTION TYPE:

AI Policy Documentation:
- AI governance frameworks and policies
- Organizational AI principles and ethics
- AI management roles and responsibilities
- AI risk management policies
- Data governance policies for AI
- AI compliance and regulatory frameworks

Procedures and Guidelines:
- Operational procedures and workflows
- Step-by-step process documentation
- Implementation guidelines
- Standard operating procedures
- Process flows and methodologies
- Detailed procedural instructions

Record Keeping System:
- Documentation management systems
- Record retention policies
- Audit trail procedures
- Version control systems
- Documentation standards and templates
- Information governance frameworks

Risk Mitigation Strategies:
- Risk assessment methodologies
- Risk mitigation plans and strategies
- Control frameworks and measures
- Risk monitoring procedures
- Incident response strategies
- Risk treatment approaches

Development Controls:
- Software development standards
- Code review processes
- Development lifecycle controls
- Quality assurance procedures
- Testing and validation frameworks
- Development security controls

Deployment Procedures:
- Deployment processes and procedures
- Production deployment guidelines
- Environment management procedures
- Release management processes
- Deployment automation and controls
- Post-deployment validation procedures

DOCUMENT CONTENT TO ANALYZE:
${documentText.substring(0, 2000)}

ANALYSIS REQUIREMENTS:
1. Determine if the document content has ANY reasonable connection to the expected section type
2. Look for keywords, topics, and content structure that might align with the section
3. Be LENIENT - if document is even 30-40% relevant, consider it acceptable
4. Only reject if document is completely unrelated (e.g., recipe for cooking when expecting AI policy)

RESPONSE FORMAT:
RELEVANT: [YES/NO]
REASON: [Brief explanation of relevance level]

Be LENIENT in your assessment. Accept documents that have reasonable relevance to the section type. Only reject completely unrelated documents.`;

      const apiConfigs = [
        {
          name: "gemini-1.5-flash-latest",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-pro",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ];

      let response: Response | null = null;

      for (const config of apiConfigs) {
        try {
          response = await fetch(config.url, {
            method: "POST",
            headers: config.headers,
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: relevancePrompt,
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.1,
                topK: 32,
                topP: 1,
                maxOutputTokens: 500,
              },
            }),
          });

          if (response.ok) {
            break;
          } else {
            response = null;
          }
        } catch (error) {
          response = null;
        }
      }

      if (response && response.ok) {
        const data = await response.json();
        const analysisText =
          data.candidates?.[0]?.content?.parts?.[0]?.text || "";

        const relevantMatch = analysisText.match(/RELEVANT:\s*(YES|NO)/i);
        const reasonMatch = analysisText.match(/REASON:\s*(.*)/is);

        const isRelevant = relevantMatch
          ? relevantMatch[1].toUpperCase() === "YES"
          : false;
        const reason = reasonMatch
          ? reasonMatch[1].trim()
          : "Could not determine document relevance.";

        return { isRelevant, reason };
      } else {
        // Fallback: basic keyword check
        return await performBasicRelevanceCheck(documentText, sectionType);
      }
    } catch (error) {
      console.error("Error checking document relevance:", error);
      // Fallback: basic keyword check
      return await performBasicRelevanceCheck(documentText, sectionType);
    }
  };

  // ✅ NEW - Fallback Basic Relevance Check
  const performBasicRelevanceCheck = async (
    documentText: string,
    sectionType: string
  ): Promise<{ isRelevant: boolean; reason: string }> => {
    const text = documentText.toLowerCase();

    const keywords = {
      "AI Policy Documentation": [
        "policy",
        "governance",
        "framework",
        "ethics",
        "principles",
        "management",
        "artificial intelligence",
        "ai governance",
      ],
      "Procedures and Guidelines": [
        "procedure",
        "process",
        "guideline",
        "workflow",
        "step",
        "instruction",
        "method",
        "implementation",
      ],
      "Record Keeping System": [
        "record",
        "documentation",
        "archive",
        "retention",
        "audit trail",
        "version control",
        "document management",
      ],
      "Risk Mitigation Strategies": [
        "risk",
        "mitigation",
        "strategy",
        "assessment",
        "control",
        "threat",
        "vulnerability",
        "incident",
      ],
      "Development Controls": [
        "development",
        "code",
        "software",
        "programming",
        "testing",
        "quality",
        "standards",
        "review",
      ],
      "Deployment Procedures": [
        "deployment",
        "production",
        "release",
        "environment",
        "installation",
        "configuration",
        "launch",
      ],
    };

    const sectionKeywords =
      keywords[sectionType as keyof typeof keywords] || [];
    const foundKeywords = sectionKeywords.filter((keyword) =>
      text.includes(keyword)
    );

    if (foundKeywords.length >= 1) {
      return {
        isRelevant: true,
        reason: `Document appears relevant - contains keywords: ${foundKeywords
          .slice(0, 3)
          .join(", ")}`,
      };
    } else {
      return {
        isRelevant: false,
        reason: `Document appears completely unrelated to ${sectionType}. Expected to find at least one of: ${sectionKeywords
          .slice(0, 4)
          .join(", ")}`,
      };
    }
  };

  // ✅ NEW - AI Policy Document Analysis Function
  const analyzeAIPolicyDocument = async () => {
    try {
      setIsAnalyzing(true);

      // Extract text content from the uploaded file
      const documentText = await extractTextFromFile(uploadedFile!);

      // First, check document relevance
      const relevanceCheck = await checkDocumentRelevance(
        documentText,
        "AI Policy Documentation"
      );
      if (!relevanceCheck.isRelevant) {
        setAnalysisResults({
          score: 0,
          explanation: `Document Relevance Issue: ${relevanceCheck.reason}`,
          recommendations: [
            "Please upload a document that contains AI policy and governance content",
            "Ensure the document covers AI management framework, roles, responsibilities, and ethical guidelines",
            "The document should focus on organizational AI policies rather than technical procedures",
            "Consider uploading documents with titles containing: AI Policy, AI Governance, AI Ethics Policy, or AI Management Framework",
          ],
        });
        return;
      }

      // Proceed with detailed analysis if document is relevant
      const analysisPrompt = `You are an ISO 42001 AI Management System compliance expert conducting a professional audit. Analyze the following AI Policy Documentation and evaluate its compliance with ISO 42001 requirements.

ISO 42001 AI POLICY DOCUMENTATION REQUIREMENTS:
1. AI governance framework and objectives
2. Roles and responsibilities for AI management
3. AI risk management approach and procedures
4. AI system development and deployment policies
5. Data governance and privacy protection measures
6. AI ethics principles and guidelines
7. Monitoring and performance measurement procedures
8. Incident response and management protocols
9. Training and competency requirements
10. Documentation and record keeping standards
11. Legal and regulatory compliance framework
12. Stakeholder engagement and communication policies

DOCUMENT CONTENT TO ANALYZE:
${documentText}

ANALYSIS REQUIREMENTS:
Provide a professional compliance assessment with:
1. A compliance score from 0-100 based on comprehensive evaluation
2. A structured explanation with clear analysis points
3. Specific actionable recommendations

RESPONSE FORMAT:
SCORE: [0-100]
EXPLANATION:
• [Professional analysis point 1 - evaluate specific ISO 42001 requirement coverage]
• [Professional analysis point 2 - assess implementation depth and quality]
• [Professional analysis point 3 - identify key strengths in the documentation]
• [Professional analysis point 4 - highlight critical gaps requiring attention]
• [Professional analysis point 5 - overall compliance assessment summary]
RECOMMENDATIONS:
• [Specific actionable recommendation 1]
• [Specific actionable recommendation 2]
• [Specific actionable recommendation 3]
• [Continue with all relevant recommendations]

IMPORTANT FORMATTING RULES:
- Do NOT use any markdown formatting symbols like **, *, _, or ##
- Write in plain text only
- Use clear, professional language without any special formatting characters
- Each bullet point should be a complete, well-structured sentence or paragraph

Provide enterprise-grade analysis. Each explanation bullet point should be substantive and professionally written. Focus on specific ISO 42001 compliance elements present or missing in the document.`;

      console.log("Starting Gemini API call for ISO 42001 analysis...");
      console.log("Document text length:", documentText.length);

      // Try multiple API configurations to handle different potential issues
      const apiConfigs = [
        {
          name: "gemini-1.5-flash-latest",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-pro",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-1.5-flash",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ];

      let response: Response | null = null;
      let lastError: string = "";

      for (const config of apiConfigs) {
        try {
          console.log(`Trying ${config.name}...`);

          response = await fetch(config.url, {
            method: "POST",
            headers: config.headers,
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: analysisPrompt,
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.3,
                topK: 32,
                topP: 1,
                maxOutputTokens: 4096,
              },
            }),
          });

          if (response.ok) {
            console.log(`Success with ${config.name}`);
            break;
          } else {
            const errorText = await response.text();
            lastError = `${config.name}: ${response.status} - ${errorText}`;
            console.log(`Failed with ${config.name}:`, lastError);
            response = null;
          }
        } catch (error) {
          lastError = `${config.name}: ${error}`;
          console.log(`Error with ${config.name}:`, error);
          response = null;
        }
      }

      if (!response) {
        throw new Error(`All API attempts failed. Last error: ${lastError}`);
      }

      console.log("API Response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("API Response received successfully");

        const analysisText =
          data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        console.log("Analysis text length:", analysisText.length);

        if (!analysisText) {
          throw new Error("No analysis text received from Gemini API");
        }

        console.log("Raw API response:", analysisText);

        // Parse the response - improved parsing for professional format
        const scoreMatch = analysisText.match(/SCORE:\s*(\d+)/i);
        const explanationMatch = analysisText.match(
          /EXPLANATION:\s*(.*?)(?=RECOMMENDATIONS:|$)/is
        );
        const recommendationsMatch = analysisText.match(
          /RECOMMENDATIONS:\s*(.*)/is
        );

        // Extract score - must be from API analysis
        const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

        // Extract explanation - parse as bullet points
        let explanationPoints: string[] = [];
        if (explanationMatch) {
          const explanationText = explanationMatch[1].trim();
          // Extract bullet points from explanation
          explanationPoints = explanationText
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((point: string) => point.length > 10); // Only meaningful points
        }

        // Fallback if no bullet points found in explanation
        if (explanationPoints.length === 0) {
          const lines = analysisText
            .split("\n")
            .filter((line: string) => line.trim().length > 0);
          const meaningfulLines = lines.filter(
            (line: string) =>
              !line.includes("SCORE:") &&
              !line.includes("EXPLANATION:") &&
              !line.includes("RECOMMENDATIONS:") &&
              line.length > 20
          );
          explanationPoints = meaningfulLines
            .slice(0, 3)
            .map((line: string) => line.trim().replace(/\*\*/g, ""));
        }

        // Extract recommendations
        let recommendations: string[] = [];
        if (recommendationsMatch) {
          recommendations = recommendationsMatch[1]
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 0);
        }

        // If no recommendations found, try alternative parsing
        if (recommendations.length === 0) {
          const allLines = analysisText.split("\n");
          recommendations = allLines
            .filter(
              (line: string) =>
                line.trim().startsWith("•") ||
                line.trim().startsWith("-") ||
                line.trim().match(/^\d+\./)
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/^\d+\.\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 10); // Only meaningful recommendations
        }

        console.log("Parsed score:", score);
        console.log(
          "Parsed explanation points count:",
          explanationPoints.length
        );
        console.log("Parsed recommendations count:", recommendations.length);

        // Set results with actual API analysis
        setAnalysisResults({
          score: Math.max(0, Math.min(100, score)), // Ensure score is 0-100
          explanation:
            explanationPoints.length > 0
              ? explanationPoints.join("|")
              : "ISO 42001 compliance analysis completed. Please review the detailed assessment.",
          recommendations:
            recommendations.length > 0
              ? recommendations
              : [
                  "Document requires comprehensive review against ISO 42001 standards",
                ],
        });

        // Only mark as completed if score is 80% or above for ISO 42001 compliance
        if (score >= 80) {
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted(
              (prev) => new Set([...prev, subsectionId])
            );
          }
        } else {
          // Remove from completed sections if score is below 80%
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted((prev) => {
              const newSet = new Set(prev);
              newSet.delete(subsectionId);
              return newSet;
            });
          }
        }
      } else {
        // Handle API errors - get actual error details
        const errorData = await response.json().catch(() => null);
        const errorText = await response.text().catch(() => "Unknown error");

        console.error("Gemini API Error:", {
          status: response.status,
          statusText: response.statusText,
          errorData,
          errorText,
          lastError,
        });

        throw new Error(
          `Gemini API failed with status ${response.status}: ${errorText}. All attempts: ${lastError}`
        );
      }
    } catch (error) {
      console.error("Error analyzing document:", error);

      // Show actual error to user for debugging
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";

      setAnalysisResults({
        score: 0,
        explanation: `Analysis failed: ${errorMessage}. Please check browser console for details and ensure the Gemini API is accessible.`,
        recommendations: [
          "Check browser console for detailed error information",
          "Verify Gemini API key and endpoint accessibility",
          "Ensure document is in a readable format",
          "Try uploading the document again",
          "Check network connectivity and CORS settings",
        ],
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ✅ NEW - Procedures and Guidelines Analysis Function
  const analyzeProceduresAndGuidelines = async () => {
    try {
      setIsAnalyzing(true);

      // Extract text content from the uploaded file
      const documentText = await extractTextFromFile(uploadedFile!);

      // First, check document relevance
      const relevanceCheck = await checkDocumentRelevance(
        documentText,
        "Procedures and Guidelines"
      );
      if (!relevanceCheck.isRelevant) {
        setAnalysisResults({
          score: 0,
          explanation: `Document Relevance Issue: ${relevanceCheck.reason}`,
          recommendations: [
            "Please upload a document that contains operational procedures and guidelines",
            "Ensure the document includes step-by-step processes, workflows, and implementation guidelines",
            "The document should focus on procedural instructions rather than policies or technical specifications",
            "Consider uploading documents with titles containing: Standard Operating Procedures, Process Guidelines, Implementation Manual, or Workflow Documentation",
          ],
        });
        return;
      }

      // Proceed with detailed analysis if document is relevant
      const analysisPrompt = `You are an ISO 42001 AI Management System compliance expert conducting a comprehensive professional audit. Perform a DEEP ANALYSIS of the entire Procedures and Guidelines documentation provided. Read through the complete document thoroughly and evaluate its compliance with ISO 42001 operational procedure requirements.

COMPREHENSIVE ISO 42001 PROCEDURES AND GUIDELINES ANALYSIS FRAMEWORK:

SECTION 1: AI SYSTEM LIFECYCLE MANAGEMENT PROCEDURES
- Development procedures: Requirements gathering, design documentation, development standards
- Testing procedures: Unit testing, integration testing, validation testing, acceptance criteria
- Deployment procedures: Production deployment, rollback procedures, environment management
- Monitoring procedures: Performance monitoring, anomaly detection, system health checks
- Retirement procedures: Decommissioning, data archival, system sunset processes

SECTION 2: DATA GOVERNANCE AND MANAGEMENT PROCEDURES
- Data collection procedures: Source validation, quality checks, consent management
- Data processing procedures: Transformation workflows, quality assurance, lineage tracking
- Data storage procedures: Security protocols, access controls, retention policies
- Data retention procedures: Lifecycle management, archival processes, deletion protocols
- Data privacy procedures: Anonymization, pseudonymization, consent management

SECTION 3: RISK MANAGEMENT AND COMPLIANCE PROCEDURES
- AI risk assessment procedures: Risk identification, impact analysis, likelihood assessment
- Risk mitigation procedures: Control implementation, monitoring effectiveness, remediation
- Compliance verification procedures: Regular audits, compliance checking, corrective actions
- Regulatory compliance procedures: Legal requirements, industry standards, reporting obligations

SECTION 4: OPERATIONAL AND GOVERNANCE PROCEDURES
- Change management procedures: Change requests, approval workflows, impact assessment
- Incident response procedures: Issue detection, escalation paths, resolution workflows
- Training and competency procedures: Skill assessment, training programs, certification
- Documentation control procedures: Version management, approval processes, distribution
- Stakeholder communication procedures: Consultation processes, feedback mechanisms, reporting
- Third-party management procedures: Vendor assessment, contract management, oversight
- Ethical AI procedures: Ethics review boards, decision frameworks, accountability measures

DOCUMENT CONTENT TO ANALYZE:
${documentText}

DEEP ANALYSIS REQUIREMENTS:
1. Read the ENTIRE document comprehensively
2. Analyze each section against the ISO 42001 framework above
3. Evaluate procedure completeness, clarity, and implementability
4. Assess compliance gaps and strengths
5. Provide specific, actionable recommendations
6. Score based on comprehensive coverage and quality

RESPONSE FORMAT:
SCORE: [0-100]
EXPLANATION:
• [Comprehensive assessment of AI system lifecycle management procedures coverage and quality]
• [Detailed evaluation of data governance and management procedure completeness and compliance]
• [In-depth analysis of risk management and compliance procedure effectiveness and alignment]
• [Thorough review of operational governance procedures and their implementation readiness]
• [Overall procedural framework assessment and ISO 42001 compliance readiness evaluation]
RECOMMENDATIONS:
• [Specific recommendation for enhancing AI system lifecycle management procedures]
• [Specific recommendation for strengthening data governance and management processes]
• [Specific recommendation for improving risk management and compliance procedures]
• [Specific recommendation for enhancing operational governance and stakeholder management]
• [Additional recommendations for achieving full ISO 42001 operational compliance]

IMPORTANT FORMATTING RULES:
- Do NOT use any markdown formatting symbols like **, *, _, or ##
- Write in plain text only
- Use clear, professional language without any special formatting characters
- Each bullet point should be a complete, well-structured sentence or paragraph

Perform COMPREHENSIVE analysis. Read the entire document thoroughly and provide enterprise-grade assessment focusing on operational procedures' depth, clarity, completeness, and full alignment with ISO 42001 requirements for AI management systems.`;

      console.log(
        "Starting Gemini API call for Procedures and Guidelines analysis..."
      );
      console.log("Document text length:", documentText.length);

      // Try multiple API configurations to handle different potential issues
      const apiConfigs = [
        {
          name: "gemini-1.5-flash-latest",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-pro",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-1.5-flash",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ];

      let response: Response | null = null;
      let lastError: string = "";

      for (const config of apiConfigs) {
        try {
          console.log(`Trying ${config.name}...`);

          response = await fetch(config.url, {
            method: "POST",
            headers: config.headers,
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: analysisPrompt,
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.3,
                topK: 32,
                topP: 1,
                maxOutputTokens: 4096,
              },
            }),
          });

          if (response.ok) {
            console.log(`Success with ${config.name}`);
            break;
          } else {
            const errorText = await response.text();
            lastError = `${config.name}: ${response.status} - ${errorText}`;
            console.log(`Failed with ${config.name}:`, lastError);
            response = null;
          }
        } catch (error) {
          lastError = `${config.name}: ${error}`;
          console.log(`Error with ${config.name}:`, error);
          response = null;
        }
      }

      if (!response) {
        throw new Error(`All API attempts failed. Last error: ${lastError}`);
      }

      console.log("API Response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("API Response received successfully");

        const analysisText =
          data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        console.log("Analysis text length:", analysisText.length);

        if (!analysisText) {
          throw new Error("No analysis text received from Gemini API");
        }

        console.log("Raw API response:", analysisText);

        // Parse the response - improved parsing for professional format
        const scoreMatch = analysisText.match(/SCORE:\s*(\d+)/i);
        const explanationMatch = analysisText.match(
          /EXPLANATION:\s*(.*?)(?=RECOMMENDATIONS:|$)/is
        );
        const recommendationsMatch = analysisText.match(
          /RECOMMENDATIONS:\s*(.*)/is
        );

        // Extract score - must be from API analysis
        const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

        // Extract explanation - parse as bullet points
        let explanationPoints: string[] = [];
        if (explanationMatch) {
          const explanationText = explanationMatch[1].trim();
          // Extract bullet points from explanation
          explanationPoints = explanationText
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((point: string) => point.length > 10); // Only meaningful points
        }

        // Fallback if no bullet points found in explanation
        if (explanationPoints.length === 0) {
          const lines = analysisText
            .split("\n")
            .filter((line: string) => line.trim().length > 0);
          const meaningfulLines = lines.filter(
            (line: string) =>
              !line.includes("SCORE:") &&
              !line.includes("EXPLANATION:") &&
              !line.includes("RECOMMENDATIONS:") &&
              line.length > 20
          );
          explanationPoints = meaningfulLines
            .slice(0, 3)
            .map((line: string) => line.trim().replace(/\*\*/g, ""));
        }

        // Extract recommendations
        let recommendations: string[] = [];
        if (recommendationsMatch) {
          recommendations = recommendationsMatch[1]
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 0);
        }

        // If no recommendations found, try alternative parsing
        if (recommendations.length === 0) {
          const allLines = analysisText.split("\n");
          recommendations = allLines
            .filter(
              (line: string) =>
                line.trim().startsWith("•") ||
                line.trim().startsWith("-") ||
                line.trim().match(/^\d+\./)
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/^\d+\.\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 10); // Only meaningful recommendations
        }

        console.log("Parsed score:", score);
        console.log(
          "Parsed explanation points count:",
          explanationPoints.length
        );
        console.log("Parsed recommendations count:", recommendations.length);

        // Set results with actual API analysis
        setAnalysisResults({
          score: Math.max(0, Math.min(100, score)), // Ensure score is 0-100
          explanation:
            explanationPoints.length > 0
              ? explanationPoints.join("|")
              : "ISO 42001 procedures and guidelines analysis completed. Please review the detailed assessment.",
          recommendations:
            recommendations.length > 0
              ? recommendations
              : [
                  "Procedures require comprehensive review against ISO 42001 operational standards",
                ],
        });

        // Only mark as completed if score is 80% or above for ISO 42001 compliance
        if (score >= 80) {
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted(
              (prev) => new Set([...prev, subsectionId])
            );
          }
        } else {
          // Remove from completed sections if score is below 80%
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted((prev) => {
              const newSet = new Set(prev);
              newSet.delete(subsectionId);
              return newSet;
            });
          }
        }
      } else {
        // Handle API errors - get actual error details
        const errorData = await response.json().catch(() => null);
        const errorText = await response.text().catch(() => "Unknown error");

        console.error("Gemini API Error:", {
          status: response.status,
          statusText: response.statusText,
          errorData,
          errorText,
          lastError,
        });

        throw new Error(
          `Gemini API failed with status ${response.status}: ${errorText}. All attempts: ${lastError}`
        );
      }
    } catch (error) {
      console.error("Error analyzing Procedures and Guidelines:", error);

      // Show actual error to user for debugging
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";

      setAnalysisResults({
        score: 0,
        explanation: `Analysis failed: ${errorMessage}. Please check browser console for details and ensure the Gemini API is accessible.`,
        recommendations: [
          "Check browser console for detailed error information",
          "Verify Gemini API key and endpoint accessibility",
          "Ensure document is in a readable format",
          "Try uploading the document again",
          "Check network connectivity and CORS settings",
        ],
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ✅ NEW - Record Keeping System Analysis Function
  const analyzeRecordKeepingSystem = async () => {
    try {
      setIsAnalyzing(true);

      // Extract text content from the uploaded file
      const documentText = await extractTextFromFile(uploadedFile!);

      // First, check document relevance
      const relevanceCheck = await checkDocumentRelevance(
        documentText,
        "Record Keeping System"
      );
      if (!relevanceCheck.isRelevant) {
        setAnalysisResults({
          score: 0,
          explanation: `Document Relevance Issue: ${relevanceCheck.reason}`,
          recommendations: [
            "Please upload a document that contains record keeping and documentation management systems",
            "Ensure the document covers document lifecycle, retention policies, and audit trail procedures",
            "The document should focus on information governance rather than operational procedures",
            "Consider uploading documents with titles containing: Document Management Policy, Record Retention Schedule, Information Governance Framework, or Audit Trail Procedures",
          ],
        });
        return;
      }

      // Proceed with detailed analysis if document is relevant
      const analysisPrompt = `You are an ISO 42001 AI Management System compliance expert conducting a comprehensive professional audit. Perform a DEEP ANALYSIS of the entire Record Keeping System documentation provided. Read through the complete document thoroughly and evaluate its compliance with ISO 42001 documentation and record management requirements.

COMPREHENSIVE ISO 42001 RECORD KEEPING SYSTEM ANALYSIS FRAMEWORK:

SECTION 1: AI SYSTEM DOCUMENTATION LIFECYCLE MANAGEMENT
- Document creation standards: Templates, authoring guidelines, review criteria
- Document review processes: Peer review, technical review, compliance validation
- Document approval workflows: Authorization levels, approval criteria, sign-off procedures
- Document update procedures: Version control, change tracking, revision management
- Document archival processes: Retention policies, storage systems, retrieval procedures

SECTION 2: DATA LINEAGE AND PROVENANCE TRACKING
- Data source documentation: Origin tracking, source validation, quality metrics
- Data transformation records: Processing logs, transformation rules, quality checks
- Data lineage mapping: End-to-end traceability, dependency tracking, impact analysis
- Data provenance records: Audit trails, access logs, modification histories
- Data quality documentation: Validation results, cleansing procedures, quality metrics

SECTION 3: AI MODEL AND SYSTEM RECORDS
- Model versioning systems: Version control, branching strategies, release management
- Configuration management: Parameter tracking, environment documentation, deployment records
- Performance monitoring records: Metrics collection, trend analysis, benchmark comparisons
- Evaluation documentation: Testing results, validation reports, performance assessments
- System architecture records: Design documentation, infrastructure specifications, integration maps

SECTION 4: GOVERNANCE AND COMPLIANCE RECORDS
- Risk assessment documentation: Risk registers, assessment methodologies, mitigation plans
- Incident response records: Issue logs, investigation reports, resolution documentation
- Training and competency records: Skill assessments, training completion, certification tracking
- Audit trail systems: Access logs, activity tracking, compliance verification
- Stakeholder communication records: Meeting minutes, decision logs, consultation records
- Change management documentation: Change requests, impact assessments, approval records
- Third-party management records: Vendor assessments, contract documentation, oversight reports
- Privacy and governance records: Consent management, policy compliance, regulatory reporting

DOCUMENT CONTENT TO ANALYZE:
${documentText}

DEEP ANALYSIS REQUIREMENTS:
1. Read the ENTIRE document comprehensively
2. Analyze each section against the ISO 42001 record keeping framework above
3. Evaluate documentation completeness, accessibility, and auditability
4. Assess record management system effectiveness and compliance
5. Provide specific, actionable recommendations for improvement
6. Score based on comprehensive coverage, quality, and audit trail capabilities

RESPONSE FORMAT:
SCORE: [0-100]
EXPLANATION:
• [Comprehensive assessment of AI system documentation lifecycle management completeness and effectiveness]
• [Detailed evaluation of data lineage and provenance tracking capabilities and audit trail quality]
• [In-depth analysis of AI model and system records management including versioning and performance tracking]
• [Thorough review of governance and compliance records including risk management and stakeholder documentation]
• [Overall record keeping system assessment and ISO 42001 documentation management compliance evaluation]
RECOMMENDATIONS:
• [Specific recommendation for enhancing AI system documentation lifecycle management and standards]
• [Specific recommendation for strengthening data lineage and provenance tracking capabilities]
• [Specific recommendation for improving AI model and system records management processes]
• [Specific recommendation for enhancing governance and compliance documentation systems]
• [Additional recommendations for achieving comprehensive ISO 42001 record keeping compliance]

IMPORTANT FORMATTING RULES:
- Do NOT use any markdown formatting symbols like **, *, _, or ##
- Write in plain text only
- Use clear, professional language without any special formatting characters
- Each bullet point should be a complete, well-structured sentence or paragraph

Perform COMPREHENSIVE analysis. Read the entire document thoroughly and provide enterprise-grade assessment focusing on record keeping system depth, completeness, auditability, and full alignment with ISO 42001 documentation management requirements.`;

      console.log(
        "Starting Gemini API call for Record Keeping System analysis..."
      );
      console.log("Document text length:", documentText.length);

      // Try multiple API configurations to handle different potential issues
      const apiConfigs = [
        {
          name: "gemini-1.5-flash-latest",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-pro",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-1.5-flash",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ];

      let response: Response | null = null;
      let lastError: string = "";

      for (const config of apiConfigs) {
        try {
          console.log(`Trying ${config.name}...`);

          response = await fetch(config.url, {
            method: "POST",
            headers: config.headers,
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: analysisPrompt,
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.3,
                topK: 32,
                topP: 1,
                maxOutputTokens: 4096,
              },
            }),
          });

          if (response.ok) {
            console.log(`Success with ${config.name}`);
            break;
          } else {
            const errorText = await response.text();
            lastError = `${config.name}: ${response.status} - ${errorText}`;
            console.log(`Failed with ${config.name}:`, lastError);
            response = null;
          }
        } catch (error) {
          lastError = `${config.name}: ${error}`;
          console.log(`Error with ${config.name}:`, error);
          response = null;
        }
      }

      if (!response) {
        throw new Error(`All API attempts failed. Last error: ${lastError}`);
      }

      console.log("API Response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("API Response received successfully");

        const analysisText =
          data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        console.log("Analysis text length:", analysisText.length);

        if (!analysisText) {
          throw new Error("No analysis text received from Gemini API");
        }

        console.log("Raw API response:", analysisText);

        // Parse the response - improved parsing for professional format
        const scoreMatch = analysisText.match(/SCORE:\s*(\d+)/i);
        const explanationMatch = analysisText.match(
          /EXPLANATION:\s*(.*?)(?=RECOMMENDATIONS:|$)/is
        );
        const recommendationsMatch = analysisText.match(
          /RECOMMENDATIONS:\s*(.*)/is
        );

        // Extract score - must be from API analysis
        const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

        // Extract explanation - parse as bullet points
        let explanationPoints: string[] = [];
        if (explanationMatch) {
          const explanationText = explanationMatch[1].trim();
          // Extract bullet points from explanation
          explanationPoints = explanationText
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((point: string) => point.length > 10); // Only meaningful points
        }

        // Fallback if no bullet points found in explanation
        if (explanationPoints.length === 0) {
          const lines = analysisText
            .split("\n")
            .filter((line: string) => line.trim().length > 0);
          const meaningfulLines = lines.filter(
            (line: string) =>
              !line.includes("SCORE:") &&
              !line.includes("EXPLANATION:") &&
              !line.includes("RECOMMENDATIONS:") &&
              line.length > 20
          );
          explanationPoints = meaningfulLines
            .slice(0, 3)
            .map((line: string) => line.trim().replace(/\*\*/g, ""));
        }

        // Extract recommendations
        let recommendations: string[] = [];
        if (recommendationsMatch) {
          recommendations = recommendationsMatch[1]
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 0);
        }

        // If no recommendations found, try alternative parsing
        if (recommendations.length === 0) {
          const allLines = analysisText.split("\n");
          recommendations = allLines
            .filter(
              (line: string) =>
                line.trim().startsWith("•") ||
                line.trim().startsWith("-") ||
                line.trim().match(/^\d+\./)
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/^\d+\.\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 10); // Only meaningful recommendations
        }

        console.log("Parsed score:", score);
        console.log(
          "Parsed explanation points count:",
          explanationPoints.length
        );
        console.log("Parsed recommendations count:", recommendations.length);

        // Set results with actual API analysis
        setAnalysisResults({
          score: Math.max(0, Math.min(100, score)), // Ensure score is 0-100
          explanation:
            explanationPoints.length > 0
              ? explanationPoints.join("|")
              : "ISO 42001 record keeping system analysis completed. Please review the detailed assessment.",
          recommendations:
            recommendations.length > 0
              ? recommendations
              : [
                  "Record keeping system requires comprehensive review against ISO 42001 documentation standards",
                ],
        });

        // Only mark as completed if score is 80% or above for ISO 42001 compliance
        if (score >= 80) {
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted(
              (prev) => new Set([...prev, subsectionId])
            );
          }
        } else {
          // Remove from completed sections if score is below 80%
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted((prev) => {
              const newSet = new Set(prev);
              newSet.delete(subsectionId);
              return newSet;
            });
          }
        }
      } else {
        // Handle API errors - get actual error details
        const errorData = await response.json().catch(() => null);
        const errorText = await response.text().catch(() => "Unknown error");

        console.error("Gemini API Error:", {
          status: response.status,
          statusText: response.statusText,
          errorData,
          errorText,
          lastError,
        });

        throw new Error(
          `Gemini API failed with status ${response.status}: ${errorText}. All attempts: ${lastError}`
        );
      }
    } catch (error) {
      console.error("Error analyzing Record Keeping System:", error);

      // Show actual error to user for debugging
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";

      setAnalysisResults({
        score: 0,
        explanation: `Analysis failed: ${errorMessage}. Please check browser console for details and ensure the Gemini API is accessible.`,
        recommendations: [
          "Check browser console for detailed error information",
          "Verify Gemini API key and endpoint accessibility",
          "Ensure document is in a readable format",
          "Try uploading the document again",
          "Check network connectivity and CORS settings",
        ],
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ✅ NEW - Risk Mitigation Strategies Analysis Function
  const analyzeRiskMitigationStrategies = async () => {
    try {
      setIsAnalyzing(true);

      // Extract text content from the uploaded file
      const documentText = await extractTextFromFile(uploadedFile!);

      // First, check document relevance
      const relevanceCheck = await checkDocumentRelevance(
        documentText,
        "Risk Mitigation Strategies"
      );
      if (!relevanceCheck.isRelevant) {
        setAnalysisResults({
          score: 0,
          explanation: `Document Relevance Issue: ${relevanceCheck.reason}`,
          recommendations: [
            "Please upload a document that contains risk assessment and mitigation strategies",
            "Ensure the document covers risk identification, assessment methodologies, and control measures",
            "The document should focus on risk management rather than operational procedures or policies",
            "Consider uploading documents with titles containing: Risk Management Plan, Risk Assessment Framework, Risk Mitigation Strategy, or Risk Control Matrix",
          ],
        });
        return;
      }

      // Proceed with detailed analysis if document is relevant
      const analysisPrompt = `You are an ISO 42001 AI Management System compliance expert conducting a comprehensive professional audit. Perform a DEEP ANALYSIS of the entire Risk Mitigation Strategies documentation provided. Read through the complete document thoroughly and evaluate its compliance with ISO 42001 risk management and mitigation requirements.

COMPREHENSIVE ISO 42001 RISK MITIGATION STRATEGIES ANALYSIS FRAMEWORK:

SECTION 1: AI RISK IDENTIFICATION AND ASSESSMENT
- AI bias and fairness risks: Algorithmic bias identification, fairness metrics, bias detection methods
- Data quality and integrity risks: Data corruption, incomplete datasets, data drift assessment
- Model performance risks: Accuracy degradation, overfitting, underfitting, generalization failures
- Security and privacy risks: Data breaches, adversarial attacks, privacy violations, unauthorized access
- Operational risks: System failures, integration issues, scalability problems, performance bottlenecks
- Regulatory and compliance risks: Legal non-compliance, regulatory changes, audit failures
- Ethical and societal risks: Unintended consequences, societal impact, stakeholder concerns

SECTION 2: RISK ASSESSMENT METHODOLOGIES
- Risk identification frameworks: Systematic identification processes, risk categorization, impact analysis
- Risk probability assessment: Likelihood evaluation, frequency analysis, statistical modeling
- Risk impact evaluation: Business impact assessment, operational consequences, financial implications
- Risk prioritization methods: Risk matrices, scoring systems, criticality assessment
- Risk monitoring systems: Continuous monitoring, early warning indicators, trend analysis

SECTION 3: RISK MITIGATION STRATEGIES AND CONTROLS
- Preventive controls: Design safeguards, input validation, access controls, encryption protocols
- Detective controls: Monitoring systems, anomaly detection, audit trails, compliance checking
- Corrective controls: Incident response, remediation procedures, rollback mechanisms, recovery plans
- Compensating controls: Alternative measures, backup systems, manual overrides, contingency plans
- Risk transfer mechanisms: Insurance, outsourcing, contractual agreements, liability sharing

SECTION 4: IMPLEMENTATION AND GOVERNANCE
- Risk mitigation planning: Strategy development, implementation roadmaps, resource allocation
- Control implementation: Technical controls, procedural controls, organizational controls
- Risk treatment decisions: Accept, avoid, mitigate, transfer strategies and decision criteria
- Monitoring and review: Effectiveness assessment, control testing, performance metrics
- Continuous improvement: Lessons learned, strategy updates, control enhancements
- Stakeholder communication: Risk reporting, transparency measures, communication protocols
- Integration with business processes: Risk-aware decision making, operational integration

SECTION 5: REGULATORY AND COMPLIANCE ALIGNMENT
- ISO 42001 compliance mapping: Standard requirements alignment, gap analysis, compliance verification
- Industry-specific requirements: Sector regulations, best practices, compliance frameworks
- International standards alignment: Other relevant standards, cross-standard compliance
- Audit and assurance: Independent verification, audit trails, compliance reporting

DOCUMENT CONTENT TO ANALYZE:
${documentText}

DEEP ANALYSIS REQUIREMENTS:
1. Read the ENTIRE document comprehensively
2. Analyze each section against the ISO 42001 risk mitigation framework above
3. Evaluate strategy completeness, effectiveness, and implementability
4. Assess risk coverage comprehensiveness and mitigation adequacy
5. Provide specific, actionable recommendations for improvement
6. Score based on comprehensive coverage, quality, and effectiveness of risk mitigation strategies

RESPONSE FORMAT:
SCORE: [0-100]
EXPLANATION:
• [Comprehensive assessment of AI risk identification and assessment coverage including bias, security, and operational risks]
• [Detailed evaluation of risk assessment methodologies and their alignment with ISO 42001 requirements]
• [In-depth analysis of risk mitigation strategies and controls including preventive, detective, and corrective measures]
• [Thorough review of implementation and governance frameworks for risk management integration]
• [Overall risk mitigation strategy assessment and ISO 42001 compliance readiness evaluation]
RECOMMENDATIONS:
• [Specific recommendation for enhancing AI risk identification and assessment capabilities]
• [Specific recommendation for strengthening risk assessment methodologies and frameworks]
• [Specific recommendation for improving risk mitigation strategies and control effectiveness]
• [Specific recommendation for enhancing implementation governance and monitoring systems]
• [Additional recommendations for achieving comprehensive ISO 42001 risk management compliance]

IMPORTANT FORMATTING RULES:
- Do NOT use any markdown formatting symbols like **, *, _, or ##
- Write in plain text only
- Use clear, professional language without any special formatting characters
- Each bullet point should be a complete, well-structured sentence or paragraph

Perform COMPREHENSIVE analysis. Read the entire document thoroughly and provide enterprise-grade assessment focusing on risk mitigation strategy depth, completeness, effectiveness, and full alignment with ISO 42001 risk management requirements.`;

      console.log(
        "Starting Gemini API call for Risk Mitigation Strategies analysis..."
      );
      console.log("Document text length:", documentText.length);

      // Try multiple API configurations to handle different potential issues
      const apiConfigs = [
        {
          name: "gemini-1.5-flash-latest",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-pro",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-1.5-flash",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ];

      let response: Response | null = null;
      let lastError: string = "";

      for (const config of apiConfigs) {
        try {
          console.log(`Trying ${config.name}...`);

          response = await fetch(config.url, {
            method: "POST",
            headers: config.headers,
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: analysisPrompt,
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.3,
                topK: 32,
                topP: 1,
                maxOutputTokens: 4096,
              },
            }),
          });

          if (response.ok) {
            console.log(`Success with ${config.name}`);
            break;
          } else {
            const errorText = await response.text();
            lastError = `${config.name}: ${response.status} - ${errorText}`;
            console.log(`Failed with ${config.name}:`, lastError);
            response = null;
          }
        } catch (error) {
          lastError = `${config.name}: ${error}`;
          console.log(`Error with ${config.name}:`, error);
          response = null;
        }
      }

      if (!response) {
        throw new Error(`All API attempts failed. Last error: ${lastError}`);
      }

      console.log("API Response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("API Response received successfully");

        const analysisText =
          data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        console.log("Analysis text length:", analysisText.length);

        if (!analysisText) {
          throw new Error("No analysis text received from Gemini API");
        }

        console.log("Raw API response:", analysisText);

        // Parse the response - improved parsing for professional format
        const scoreMatch = analysisText.match(/SCORE:\s*(\d+)/i);
        const explanationMatch = analysisText.match(
          /EXPLANATION:\s*(.*?)(?=RECOMMENDATIONS:|$)/is
        );
        const recommendationsMatch = analysisText.match(
          /RECOMMENDATIONS:\s*(.*)/is
        );

        // Extract score - must be from API analysis
        const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

        // Extract explanation - parse as bullet points
        let explanationPoints: string[] = [];
        if (explanationMatch) {
          const explanationText = explanationMatch[1].trim();
          // Extract bullet points from explanation
          explanationPoints = explanationText
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((point: string) => point.length > 10); // Only meaningful points
        }

        // Fallback if no bullet points found in explanation
        if (explanationPoints.length === 0) {
          const lines = analysisText
            .split("\n")
            .filter((line: string) => line.trim().length > 0);
          const meaningfulLines = lines.filter(
            (line: string) =>
              !line.includes("SCORE:") &&
              !line.includes("EXPLANATION:") &&
              !line.includes("RECOMMENDATIONS:") &&
              line.length > 20
          );
          explanationPoints = meaningfulLines
            .slice(0, 3)
            .map((line: string) => line.trim().replace(/\*\*/g, ""));
        }

        // Extract recommendations
        let recommendations: string[] = [];
        if (recommendationsMatch) {
          recommendations = recommendationsMatch[1]
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 0);
        }

        // If no recommendations found, try alternative parsing
        if (recommendations.length === 0) {
          const allLines = analysisText.split("\n");
          recommendations = allLines
            .filter(
              (line: string) =>
                line.trim().startsWith("•") ||
                line.trim().startsWith("-") ||
                line.trim().match(/^\d+\./)
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/^\d+\.\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 10); // Only meaningful recommendations
        }

        console.log("Parsed score:", score);
        console.log(
          "Parsed explanation points count:",
          explanationPoints.length
        );
        console.log("Parsed recommendations count:", recommendations.length);

        // Set results with actual API analysis
        setAnalysisResults({
          score: Math.max(0, Math.min(100, score)), // Ensure score is 0-100
          explanation:
            explanationPoints.length > 0
              ? explanationPoints.join("|")
              : "ISO 42001 risk mitigation strategies analysis completed. Please review the detailed assessment.",
          recommendations:
            recommendations.length > 0
              ? recommendations
              : [
                  "Risk mitigation strategies require comprehensive review against ISO 42001 risk management standards",
                ],
        });

        // Only mark as completed if score is 80% or above for ISO 42001 compliance
        if (score >= 80) {
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted(
              (prev) => new Set([...prev, subsectionId])
            );
          }
        } else {
          // Remove from completed sections if score is below 80%
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted((prev) => {
              const newSet = new Set(prev);
              newSet.delete(subsectionId);
              return newSet;
            });
          }
        }
      } else {
        // Handle API errors - get actual error details
        const errorData = await response.json().catch(() => null);
        const errorText = await response.text().catch(() => "Unknown error");

        console.error("Gemini API Error:", {
          status: response.status,
          statusText: response.statusText,
          errorData,
          errorText,
          lastError,
        });

        throw new Error(
          `Gemini API failed with status ${response.status}: ${errorText}. All attempts: ${lastError}`
        );
      }
    } catch (error) {
      console.error("Error analyzing Risk Mitigation Strategies:", error);

      // Show actual error to user for debugging
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";

      setAnalysisResults({
        score: 0,
        explanation: `Analysis failed: ${errorMessage}. Please check browser console for details and ensure the Gemini API is accessible.`,
        recommendations: [
          "Check browser console for detailed error information",
          "Verify Gemini API key and endpoint accessibility",
          "Ensure document is in a readable format",
          "Try uploading the document again",
          "Check network connectivity and CORS settings",
        ],
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ✅ NEW - Development Controls Analysis Function
  const analyzeDevelopmentControls = async () => {
    try {
      setIsAnalyzing(true);

      // Extract text content from the uploaded file
      const documentText = await extractTextFromFile(uploadedFile!);

      // First, check document relevance
      const relevanceCheck = await checkDocumentRelevance(
        documentText,
        "Development Controls"
      );
      if (!relevanceCheck.isRelevant) {
        setAnalysisResults({
          score: 0,
          explanation: `Document Relevance Issue: ${relevanceCheck.reason}`,
          recommendations: [
            "Please upload a document that contains software development controls and standards",
            "Ensure the document covers coding standards, review processes, and quality assurance procedures",
            "The document should focus on development lifecycle controls rather than deployment or operational procedures",
            "Consider uploading documents with titles containing: Development Standards, Code Review Guidelines, Software Quality Manual, or Development Lifecycle Controls",
          ],
        });
        return;
      }

      // Proceed with detailed analysis if document is relevant
      const analysisPrompt = `You are an ISO 42001 AI Management System compliance expert conducting a comprehensive professional audit. Perform a DEEP ANALYSIS of the entire Development Controls documentation provided. Read through the complete document thoroughly and evaluate its compliance with ISO 42001 AI system development control requirements.

COMPREHENSIVE ISO 42001 DEVELOPMENT CONTROLS ANALYSIS FRAMEWORK:

SECTION 1: AI SYSTEM DEVELOPMENT LIFECYCLE CONTROLS
- Requirements management: Functional requirements, non-functional requirements, acceptance criteria definition
- Design controls: Architecture review, design patterns, scalability considerations, security by design
- Code development standards: Coding guidelines, code review processes, version control, documentation standards
- Quality assurance controls: Code quality metrics, static analysis, peer review processes, compliance checking
- Configuration management: Environment controls, dependency management, build processes, artifact management

SECTION 2: DATA AND MODEL DEVELOPMENT CONTROLS
- Data governance controls: Data quality validation, data lineage tracking, data access controls, privacy protection
- Data preparation controls: Data cleaning, transformation validation, feature engineering oversight, bias detection
- Model development controls: Algorithm selection criteria, hyperparameter tuning governance, model validation frameworks
- Training controls: Training data validation, model performance monitoring, overfitting prevention, reproducibility requirements
- Model evaluation controls: Performance metrics, fairness assessment, robustness testing, validation datasets

SECTION 3: SECURITY AND PRIVACY CONTROLS
- Secure development practices: Security coding standards, vulnerability scanning, penetration testing, threat modeling
- Access control systems: Role-based access, authentication mechanisms, authorization frameworks, audit logging
- Data protection controls: Encryption standards, data anonymization, privacy impact assessments, consent management
- Infrastructure security: Secure development environments, network security, container security, cloud security controls
- Supply chain security: Third-party component validation, dependency scanning, license compliance, vulnerability management

SECTION 4: TESTING AND VALIDATION CONTROLS
- Unit testing frameworks: Test coverage requirements, automated testing, test-driven development, regression testing
- Integration testing controls: API testing, system integration, end-to-end testing, performance testing
- AI-specific testing: Model validation, bias testing, fairness evaluation, adversarial testing, robustness assessment
- User acceptance testing: Stakeholder validation, usability testing, business requirement verification, acceptance criteria
- Compliance testing: Regulatory compliance validation, standard adherence testing, audit trail verification

SECTION 5: DOCUMENTATION AND TRACEABILITY CONTROLS
- Development documentation: Technical specifications, API documentation, deployment guides, user manuals
- Process documentation: Development workflows, approval processes, change management, incident response
- Traceability controls: Requirements traceability, code-to-requirement mapping, issue tracking, decision logging
- Version control: Code versioning, documentation versioning, release management, branch management
- Audit trail maintenance: Development activity logging, change tracking, approval records, compliance evidence

SECTION 6: GOVERNANCE AND COMPLIANCE CONTROLS
- Development governance: Oversight committees, approval workflows, compliance checkpoints, quality gates
- Regulatory compliance: Industry standards adherence, legal requirement compliance, certification processes
- Risk management integration: Risk assessment during development, mitigation controls, continuous monitoring
- Stakeholder management: Communication protocols, feedback incorporation, change approval processes
- Continuous improvement: Lessons learned, process optimization, control effectiveness assessment, metrics tracking

DOCUMENT CONTENT TO ANALYZE:
${documentText}

DEEP ANALYSIS REQUIREMENTS:
1. Read the ENTIRE document comprehensively
2. Analyze each section against the ISO 42001 development controls framework above
3. Evaluate control completeness, effectiveness, and implementation adequacy
4. Assess development process maturity and compliance readiness
5. Provide specific, actionable recommendations for improvement
6. Score based on comprehensive coverage, quality, and effectiveness of development controls

RESPONSE FORMAT:
SCORE: [0-100]
EXPLANATION:
• [Comprehensive assessment of AI system development lifecycle controls including requirements, design, and quality assurance]
• [Detailed evaluation of data and model development controls including governance and validation frameworks]
• [In-depth analysis of security and privacy controls throughout the development process]
• [Thorough review of testing and validation controls including AI-specific testing methodologies]
• [Overall development controls assessment and ISO 42001 compliance readiness evaluation]
RECOMMENDATIONS:
• [Specific recommendation for enhancing AI system development lifecycle controls and processes]
• [Specific recommendation for strengthening data and model development governance and validation]
• [Specific recommendation for improving security and privacy controls in development processes]
• [Specific recommendation for enhancing testing and validation frameworks and methodologies]
• [Additional recommendations for achieving comprehensive ISO 42001 development controls compliance]

IMPORTANT FORMATTING RULES:
- Do NOT use any markdown formatting symbols like **, *, _, or ##
- Write in plain text only
- Use clear, professional language without any special formatting characters
- Each bullet point should be a complete, well-structured sentence or paragraph

Perform COMPREHENSIVE analysis. Read the entire document thoroughly and provide enterprise-grade assessment focusing on development controls depth, completeness, effectiveness, and full alignment with ISO 42001 AI system development requirements.`;
      console.log(
        "Starting Gemini API call for Development Controls analysis..."
      );
      console.log("Document text length:", documentText.length);

      // Try multiple API configurations to handle different potential issues
      const apiConfigs = [
        {
          name: "gemini-1.5-flash-latest",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-pro",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-1.5-flash",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ];

      let response: Response | null = null;
      let lastError: string = "";

      for (const config of apiConfigs) {
        try {
          console.log(`Trying ${config.name}...`);

          response = await fetch(config.url, {
            method: "POST",
            headers: config.headers,
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: analysisPrompt,
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.3,
                topK: 32,
                topP: 1,
                maxOutputTokens: 4096,
              },
            }),
          });

          if (response.ok) {
            console.log(`Success with ${config.name}`);
            break;
          } else {
            const errorText = await response.text();
            lastError = `${config.name}: ${response.status} - ${errorText}`;
            console.log(`Failed with ${config.name}:`, lastError);
            response = null;
          }
        } catch (error) {
          lastError = `${config.name}: ${error}`;
          console.log(`Error with ${config.name}:`, error);
          response = null;
        }
      }

      if (!response) {
        throw new Error(`All API attempts failed. Last error: ${lastError}`);
      }

      console.log("API Response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("API Response received successfully");

        const analysisText =
          data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        console.log("Analysis text length:", analysisText.length);

        if (!analysisText) {
          throw new Error("No analysis text received from Gemini API");
        }

        console.log("Raw API response:", analysisText);

        // Parse the response - improved parsing for professional format
        const scoreMatch = analysisText.match(/SCORE:\s*(\d+)/i);
        const explanationMatch = analysisText.match(
          /EXPLANATION:\s*(.*?)(?=RECOMMENDATIONS:|$)/is
        );
        const recommendationsMatch = analysisText.match(
          /RECOMMENDATIONS:\s*(.*)/is
        );

        // Extract score - must be from API analysis
        const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

        // Extract explanation - parse as bullet points
        let explanationPoints: string[] = [];
        if (explanationMatch) {
          const explanationText = explanationMatch[1].trim();
          // Extract bullet points from explanation
          explanationPoints = explanationText
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((point: string) => point.length > 10); // Only meaningful points
        }

        // Fallback if no bullet points found in explanation
        if (explanationPoints.length === 0) {
          const lines = analysisText
            .split("\n")
            .filter((line: string) => line.trim().length > 0);
          const meaningfulLines = lines.filter(
            (line: string) =>
              !line.includes("SCORE:") &&
              !line.includes("EXPLANATION:") &&
              !line.includes("RECOMMENDATIONS:") &&
              line.length > 20
          );
          explanationPoints = meaningfulLines
            .slice(0, 3)
            .map((line: string) => line.trim().replace(/\*\*/g, ""));
        }

        // Extract recommendations
        let recommendations: string[] = [];
        if (recommendationsMatch) {
          recommendations = recommendationsMatch[1]
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 0);
        }

        // If no recommendations found, try alternative parsing
        if (recommendations.length === 0) {
          const allLines = analysisText.split("\n");
          recommendations = allLines
            .filter(
              (line: string) =>
                line.trim().startsWith("•") ||
                line.trim().startsWith("-") ||
                line.trim().match(/^\d+\./)
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/^\d+\.\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 10); // Only meaningful recommendations
        }

        console.log("Parsed score:", score);
        console.log(
          "Parsed explanation points count:",
          explanationPoints.length
        );
        console.log("Parsed recommendations count:", recommendations.length);

        // Set results with actual API analysis
        setAnalysisResults({
          score: Math.max(0, Math.min(100, score)), // Ensure score is 0-100
          explanation:
            explanationPoints.length > 0
              ? explanationPoints.join("|")
              : "ISO 42001 development controls analysis completed. Please review the detailed assessment.",
          recommendations:
            recommendations.length > 0
              ? recommendations
              : [
                  "Development controls require comprehensive review against ISO 42001 development standards",
                ],
        });

        // Only mark as completed if score is 80% or above for ISO 42001 compliance
        if (score >= 80) {
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted(
              (prev) => new Set([...prev, subsectionId])
            );
          }
        } else {
          // Remove from completed sections if score is below 80%
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted((prev) => {
              const newSet = new Set(prev);
              newSet.delete(subsectionId);
              return newSet;
            });
          }
        }
      } else {
        // Handle API errors - get actual error details
        const errorData = await response.json().catch(() => null);
        const errorText = await response.text().catch(() => "Unknown error");

        console.error("Gemini API Error:", {
          status: response.status,
          statusText: response.statusText,
          errorData,
          errorText,
          lastError,
        });

        throw new Error(
          `Gemini API failed with status ${response.status}: ${errorText}. All attempts: ${lastError}`
        );
      }
    } catch (error) {
      console.error("Error analyzing Development Controls:", error);

      // Show actual error to user for debugging
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";

      setAnalysisResults({
        score: 0,
        explanation: `Analysis failed: ${errorMessage}. Please check browser console for details and ensure the Gemini API is accessible.`,
        recommendations: [
          "Check browser console for detailed error information",
          "Verify Gemini API key and endpoint accessibility",
          "Ensure document is in a readable format",
          "Try uploading the document again",
          "Check network connectivity and CORS settings",
        ],
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ✅ NEW - Deployment Procedures Analysis Function
  const analyzeDeploymentProcedures = async () => {
    try {
      setIsAnalyzing(true);

      // Extract text content from the uploaded file
      const documentText = await extractTextFromFile(uploadedFile!);

      // First, check document relevance
      const relevanceCheck = await checkDocumentRelevance(
        documentText,
        "Deployment Procedures"
      );
      if (!relevanceCheck.isRelevant) {
        setAnalysisResults({
          score: 0,
          explanation: `Document Relevance Issue: ${relevanceCheck.reason}`,
          recommendations: [
            "Please upload a document that contains deployment and production procedures",
            "Ensure the document covers deployment processes, environment management, and release procedures",
            "The document should focus on deployment and operational procedures rather than development or policy documents",
            "Consider uploading documents with titles containing: Deployment Guide, Production Procedures, Release Management, or Environment Setup Guide",
          ],
        });
        return;
      }

      // Proceed with detailed analysis if document is relevant
      const analysisPrompt = `You are an ISO 42001 AI Management System compliance expert conducting a comprehensive professional audit. Perform a DEEP ANALYSIS of the entire Deployment Procedures documentation provided. Read through the complete document thoroughly and evaluate its compliance with ISO 42001 AI system deployment and operational control requirements.

COMPREHENSIVE ISO 42001 DEPLOYMENT PROCEDURES ANALYSIS FRAMEWORK:

SECTION 1: PRE-DEPLOYMENT PREPARATION AND VALIDATION
- Deployment readiness assessment: System validation, performance benchmarking, scalability testing, integration verification
- Environment preparation: Infrastructure provisioning, security configuration, monitoring setup, backup systems
- Change management controls: Deployment approval workflows, impact assessment, rollback planning, stakeholder notification
- Documentation preparation: Deployment guides, operational procedures, troubleshooting documentation, user manuals
- Compliance verification: Regulatory compliance check, security validation, privacy assessment, audit trail preparation

SECTION 2: DEPLOYMENT EXECUTION CONTROLS
- Deployment orchestration: Automated deployment pipelines, configuration management, version control, artifact management
- Progressive deployment strategies: Blue-green deployment, canary releases, rolling updates, feature flags, gradual rollout
- Quality gates and checkpoints: Automated testing, smoke tests, health checks, performance validation, acceptance criteria
- Security controls during deployment: Access controls, encryption in transit, secure configuration, vulnerability scanning
- Monitoring and alerting: Real-time monitoring setup, alert configuration, logging systems, performance metrics collection

SECTION 3: PRODUCTION ENVIRONMENT CONTROLS
- Production infrastructure management: Scalability controls, load balancing, high availability, disaster recovery
- Operational security: Runtime security monitoring, threat detection, incident response, access audit logging
- Performance monitoring: System performance metrics, resource utilization, response time monitoring, capacity planning
- Data management in production: Data backup procedures, retention policies, archival processes, data recovery plans
- Configuration management: Production configuration control, change tracking, environment consistency, drift detection

SECTION 4: POST-DEPLOYMENT VALIDATION AND MONITORING
- System validation procedures: Functional testing, performance validation, integration verification, user acceptance testing
- AI model performance monitoring: Accuracy tracking, bias detection, data drift monitoring, model degradation assessment
- Business impact assessment: Key performance indicators, business metrics, user feedback collection, success criteria evaluation
- Compliance monitoring: Regulatory compliance verification, audit trail maintenance, policy adherence tracking
- Issue identification and resolution: Problem detection, root cause analysis, remediation procedures, lessons learned

SECTION 5: OPERATIONAL PROCEDURES AND MAINTENANCE
- Ongoing operational procedures: System maintenance, updates and patches, configuration changes, capacity management
- Incident response procedures: Issue escalation, emergency procedures, communication protocols, recovery processes
- Change management in production: Change approval processes, impact assessment, testing procedures, rollback capabilities
- User support and training: User onboarding, training materials, support procedures, feedback mechanisms
- Continuous monitoring and improvement: Performance optimization, process refinement, technology updates, best practice adoption

SECTION 6: GOVERNANCE AND COMPLIANCE INTEGRATION
- Deployment governance: Oversight committees, approval workflows, compliance checkpoints, quality assurance
- Risk management integration: Deployment risk assessment, mitigation strategies, continuous risk monitoring
- Audit and compliance: Deployment audit trails, compliance reporting, regulatory adherence, certification maintenance
- Stakeholder communication: Deployment communication, status reporting, feedback collection, change notification
- Documentation and knowledge management: Procedure documentation, knowledge transfer, training materials, version control

DOCUMENT CONTENT TO ANALYZE:
${documentText}

DEEP ANALYSIS REQUIREMENTS:
1. Read the ENTIRE document comprehensively
2. Analyze each section against the ISO 42001 deployment procedures framework above
3. Evaluate procedure completeness, effectiveness, and operational readiness
4. Assess deployment process maturity and compliance adequacy
5. Provide specific, actionable recommendations for improvement
6. Score based on comprehensive coverage, quality, and effectiveness of deployment procedures

RESPONSE FORMAT:
SCORE: [0-100]
EXPLANATION:
• [Comprehensive assessment of pre-deployment preparation and validation procedures including readiness and compliance verification]
• [Detailed evaluation of deployment execution controls including orchestration, security, and quality gates]
• [In-depth analysis of production environment controls including infrastructure management and operational security]
• [Thorough review of post-deployment validation and monitoring including AI model performance and business impact assessment]
• [Overall deployment procedures assessment and ISO 42001 operational compliance readiness evaluation]
RECOMMENDATIONS:
• [Specific recommendation for enhancing pre-deployment preparation and validation procedures]
• [Specific recommendation for strengthening deployment execution controls and automation]
• [Specific recommendation for improving production environment controls and operational security]
• [Specific recommendation for enhancing post-deployment monitoring and validation procedures]
• [Additional recommendations for achieving comprehensive ISO 42001 deployment procedures compliance]

IMPORTANT FORMATTING RULES:
- Do NOT use any markdown formatting symbols like **, *, _, or ##
- Write in plain text only
- Use clear, professional language without any special formatting characters
- Each bullet point should be a complete, well-structured sentence or paragraph

Perform COMPREHENSIVE analysis. Read the entire document thoroughly and provide enterprise-grade assessment focusing on deployment procedures depth, completeness, operational readiness, and full alignment with ISO 42001 AI system deployment requirements.`;

      console.log(
        "Starting Gemini API call for Deployment Procedures analysis..."
      );
      console.log("Document text length:", documentText.length);

      // Try multiple API configurations to handle different potential issues
      const apiConfigs = [
        {
          name: "gemini-1.5-flash-latest",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-pro",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
        {
          name: "gemini-1.5-flash",
          url: `https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=AIzaSyCb8vE2NtQApNeMNsZ6ZfaG0Wtxyzl3pGE`,
          headers: {
            "Content-Type": "application/json",
          },
        },
      ];

      let response: Response | null = null;
      let lastError: string = "";

      for (const config of apiConfigs) {
        try {
          console.log(`Trying ${config.name}...`);

          response = await fetch(config.url, {
            method: "POST",
            headers: config.headers,
            body: JSON.stringify({
              contents: [
                {
                  parts: [
                    {
                      text: analysisPrompt,
                    },
                  ],
                },
              ],
              generationConfig: {
                temperature: 0.3,
                topK: 32,
                topP: 1,
                maxOutputTokens: 4096,
              },
            }),
          });

          if (response.ok) {
            console.log(`Success with ${config.name}`);
            break;
          } else {
            const errorText = await response.text();
            lastError = `${config.name}: ${response.status} - ${errorText}`;
            console.log(`Failed with ${config.name}:`, lastError);
            response = null;
          }
        } catch (error) {
          lastError = `${config.name}: ${error}`;
          console.log(`Error with ${config.name}:`, error);
          response = null;
        }
      }

      if (!response) {
        throw new Error(`All API attempts failed. Last error: ${lastError}`);
      }

      console.log("API Response status:", response.status);

      if (response.ok) {
        const data = await response.json();
        console.log("API Response received successfully");

        const analysisText =
          data.candidates?.[0]?.content?.parts?.[0]?.text || "";
        console.log("Analysis text length:", analysisText.length);

        if (!analysisText) {
          throw new Error("No analysis text received from Gemini API");
        }

        console.log("Raw API response:", analysisText);

        // Parse the response - improved parsing for professional format
        const scoreMatch = analysisText.match(/SCORE:\s*(\d+)/i);
        const explanationMatch = analysisText.match(
          /EXPLANATION:\s*(.*?)(?=RECOMMENDATIONS:|$)/is
        );
        const recommendationsMatch = analysisText.match(
          /RECOMMENDATIONS:\s*(.*)/is
        );

        // Extract score - must be from API analysis
        const score = scoreMatch ? parseInt(scoreMatch[1]) : 0;

        // Extract explanation - parse as bullet points
        let explanationPoints: string[] = [];
        if (explanationMatch) {
          const explanationText = explanationMatch[1].trim();
          // Extract bullet points from explanation
          explanationPoints = explanationText
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((point: string) => point.length > 10); // Only meaningful points
        }

        // Fallback if no bullet points found in explanation
        if (explanationPoints.length === 0) {
          const lines = analysisText
            .split("\n")
            .filter((line: string) => line.trim().length > 0);
          const meaningfulLines = lines.filter(
            (line: string) =>
              !line.includes("SCORE:") &&
              !line.includes("EXPLANATION:") &&
              !line.includes("RECOMMENDATIONS:") &&
              line.length > 20
          );
          explanationPoints = meaningfulLines
            .slice(0, 3)
            .map((line: string) => line.trim().replace(/\*\*/g, ""));
        }

        // Extract recommendations
        let recommendations: string[] = [];
        if (recommendationsMatch) {
          recommendations = recommendationsMatch[1]
            .split("\n")
            .filter(
              (line: string) =>
                line.trim().startsWith("•") || line.trim().startsWith("-")
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 0);
        }

        // If no recommendations found, try alternative parsing
        if (recommendations.length === 0) {
          const allLines = analysisText.split("\n");
          recommendations = allLines
            .filter(
              (line: string) =>
                line.trim().startsWith("•") ||
                line.trim().startsWith("-") ||
                line.trim().match(/^\d+\./)
            )
            .map((line: string) =>
              line
                .trim()
                .replace(/^[•-]\s*/, "")
                .replace(/^\d+\.\s*/, "")
                .replace(/\*\*/g, "")
                .trim()
            )
            .filter((rec: string) => rec.length > 10); // Only meaningful recommendations
        }

        console.log("Parsed score:", score);
        console.log(
          "Parsed explanation points count:",
          explanationPoints.length
        );
        console.log("Parsed recommendations count:", recommendations.length);

        // Set results with actual API analysis
        setAnalysisResults({
          score: Math.max(0, Math.min(100, score)), // Ensure score is 0-100
          explanation:
            explanationPoints.length > 0
              ? explanationPoints.join("|")
              : "ISO 42001 deployment procedures analysis completed. Please review the detailed assessment.",
          recommendations:
            recommendations.length > 0
              ? recommendations
              : [
                  "Deployment procedures require comprehensive review against ISO 42001 deployment standards",
                ],
        });

        // Only mark as completed if score is 80% or above for ISO 42001 compliance
        if (score >= 80) {
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted(
              (prev) => new Set([...prev, subsectionId])
            );
          }
        } else {
          // Remove from completed sections if score is below 80%
          const subsectionId = getSubsectionIdFromTitle(currentUploadSection);
          if (subsectionId) {
            setAutoSectionsCompleted((prev) => {
              const newSet = new Set(prev);
              newSet.delete(subsectionId);
              return newSet;
            });
          }
        }
      } else {
        // Handle API errors - get actual error details
        const errorData = await response.json().catch(() => null);
        const errorText = await response.text().catch(() => "Unknown error");

        console.error("Gemini API Error:", {
          status: response.status,
          statusText: response.statusText,
          errorData,
          errorText,
          lastError,
        });

        throw new Error(
          `Gemini API failed with status ${response.status}: ${errorText}. All attempts: ${lastError}`
        );
      }
    } catch (error) {
      console.error("Error analyzing Deployment Procedures:", error);

      // Show actual error to user for debugging
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error occurred";

      setAnalysisResults({
        score: 0,
        explanation: `Analysis failed: ${errorMessage}. Please check browser console for details and ensure the Gemini API is accessible.`,
        recommendations: [
          "Check browser console for detailed error information",
          "Verify Gemini API key and endpoint accessibility",
          "Ensure document is in a readable format",
          "Try uploading the document again",
          "Check network connectivity and CORS settings",
        ],
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ✅ NEW - Extract text from uploaded file
  const extractTextFromFile = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (e) => {
        const text = e.target?.result as string;
        // For demo purposes, return the text content
        // In a real implementation, you'd handle different file types (PDF, DOC, etc.)
        resolve(text || "Document content could not be extracted.");
      };
      reader.onerror = () => reject(new Error("Failed to read file"));

      // For text files, read as text. For others, you'd need specialized parsers
      if (file.type.includes("text")) {
        reader.readAsText(file);
      } else {
        // For demo, return placeholder text for non-text files
        resolve(
          `AI Policy Document uploaded: ${file.name}. This is a ${file.type} file that would be processed by the document parser. For demo purposes, analyzing based on typical AI policy content structure.`
        );
      }
    });
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
            ? "bg-green-50 border-green-200"
            : "bg-gray-50 border-gray-200 hover:bg-gray-100 cursor-pointer"
        }`}
        onClick={() =>
          !isAutoCompleted && handleSubsectionClick(subsectionId, title)
        }
      >
        {isAutoCompleted ? (
          <CheckCircle className="w-5 h-5 text-green-500 mt-0.5 flex-shrink-0" />
        ) : (
          <div className="w-5 h-5 border-2 border-gray-300 rounded mt-0.5 flex-shrink-0"></div>
        )}
        <div className="flex-1">
          <h4
            className={`font-medium ${
              isAutoCompleted ? "text-green-900" : "text-gray-900"
            }`}
          >
            {title}
          </h4>
          <p
            className={`text-sm ${
              isAutoCompleted ? "text-green-700" : "text-gray-600"
            }`}
          >
            {isAutoCompleted
              ? "Meets ISO 42001 compliance requirements"
              : description}
          </p>
          {isAutoCompleted && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-700 mt-1">
              ISO Compliant
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
    subsections: Array<{ id: string; title: string; description: string }>,
    isCompleted: boolean = false
  ) => {
    // Check if any subsections are completed to determine section status
    const hasCompletedSubsections = subsections.some((sub) =>
      autoSectionsCompleted.has(sub.id)
    );
    const allSubsectionsCompleted = subsections.every((sub) =>
      autoSectionsCompleted.has(sub.id)
    );

    return (
      <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
        <div
          className="flex items-center justify-between p-6 cursor-pointer hover:bg-gray-50 transition-colors"
          onClick={() => toggleSection(sectionNumber)}
        >
          <div className="flex items-center">
            <div
              className={`w-10 h-10 rounded-xl flex items-center justify-center mr-4 ${
                allSubsectionsCompleted
                  ? "bg-green-100"
                  : hasCompletedSubsections
                  ? "bg-yellow-100"
                  : "bg-blue-100"
              }`}
            >
              {allSubsectionsCompleted ? (
                <CheckCircle className="w-6 h-6 text-green-600" />
              ) : (
                icon
              )}
            </div>
            <div>
              <h3 className="text-lg font-semibold text-gray-900">{title}</h3>
              <p className="text-sm text-gray-600">
                {allSubsectionsCompleted
                  ? "Completed"
                  : hasCompletedSubsections
                  ? "Partially Complete"
                  : "In Progress"}
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
                  ? "Some requirements are ISO compliant, others require documentation"
                  : "Click on incomplete items to upload documentation"}
              </p>
              <div className="space-y-3">
                {subsections.map((subsection) =>
                  renderSubsectionItem(
                    subsection.id,
                    subsection.title,
                    subsection.description
                  )
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // ✅ ENHANCED - Upload Modal Component with AI Analysis Results
  const UploadModal = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div
        className={`bg-white rounded-xl p-6 w-full mx-4 ${
          analysisResults ? "max-w-4xl" : "max-w-md"
        }`}
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">
            Upload Documentation
          </h3>
          <button
            onClick={() => {
              setShowUploadModal(false);
              setUploadedFile(null);
              setCurrentUploadSection("");
              setAnalysisResults(null);
            }}
            className="text-gray-400 hover:text-gray-600"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div
          className={`${
            analysisResults ? "grid grid-cols-1 lg:grid-cols-2 gap-6" : ""
          }`}
        >
          {/* Upload Section */}
          <div>
            <p className="text-sm text-gray-600 mb-4">
              Upload documentation for: <strong>{currentUploadSection}</strong>
            </p>

            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center mb-4">
              <Upload className="w-8 h-8 text-gray-400 mx-auto mb-2" />
              <p className="text-sm text-gray-600 mb-2">
                {uploadedFile
                  ? uploadedFile.name
                  : "Click to upload or drag and drop"}
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
              disabled={!uploadedFile || isAnalyzing}
              className={`w-full py-2 px-4 rounded-lg font-medium ${
                uploadedFile && !isAnalyzing
                  ? "bg-blue-600 text-white hover:bg-blue-700"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`}
            >
              {isAnalyzing ? "Analyzing..." : "Submit"}
            </button>

            {/* Close button for analyzed documents */}
            {analysisResults && (
              <button
                onClick={() => {
                  setShowUploadModal(false);
                  setUploadedFile(null);
                  setCurrentUploadSection("");
                  setAnalysisResults(null);
                }}
                className="w-full mt-3 py-2 px-4 rounded-lg font-medium bg-gray-600 text-white hover:bg-gray-700"
              >
                Close
              </button>
            )}
          </div>

          {/* ✅ NEW - Analysis Results Section */}
          {analysisResults && (
            <div className="border-l border-gray-200 pl-6">
              <h4 className="text-lg font-semibold text-gray-900 mb-4">
                ISO 42001 Compliance Analysis
              </h4>

              {/* Compliance Score */}
              <div className="bg-gray-50 rounded-lg p-4 mb-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm font-medium text-gray-700">
                    Compliance Score
                  </span>
                  <span
                    className={`text-2xl font-bold ${
                      analysisResults.score >= 80
                        ? "text-green-600"
                        : analysisResults.score >= 60
                        ? "text-yellow-600"
                        : "text-red-600"
                    }`}
                  >
                    {analysisResults.score}%
                  </span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`h-2 rounded-full ${
                      analysisResults.score >= 80
                        ? "bg-green-500"
                        : analysisResults.score >= 60
                        ? "bg-yellow-500"
                        : "bg-red-500"
                    }`}
                    style={{ width: `${analysisResults.score}%` }}
                  ></div>
                </div>
                {analysisResults.score >= 80 ? (
                  <div className="mt-2 flex items-center text-sm text-green-600">
                    <CheckCircle className="w-4 h-4 mr-1" />
                    <span>Compliant with ISO 42001 requirements (≥80%)</span>
                  </div>
                ) : (
                  <div className="mt-2 flex items-center text-sm text-red-600">
                    <AlertCircle className="w-4 h-4 mr-1" />
                    <span>Not compliant - Document &lt;80% threshold</span>
                  </div>
                )}
              </div>

              {/* Explanation */}
              <div className="mb-4">
                <h5 className="text-sm font-semibold text-gray-900 mb-2">
                  Analysis Explanation
                </h5>
                <div
                  className={`${
                    analysisResults.explanation.includes(
                      "Document Relevance Issue"
                    )
                      ? "bg-red-50"
                      : "bg-blue-50"
                  } rounded-lg p-3 max-h-32 overflow-y-auto`}
                >
                  {analysisResults.explanation.includes(
                    "Document Relevance Issue"
                  ) && (
                    <div className="flex items-center mb-2">
                      <div className="w-4 h-4 bg-red-500 rounded-full mr-2 flex items-center justify-center">
                        <span className="text-white text-xs font-bold">!</span>
                      </div>
                      <span className="text-red-700 font-medium text-xs">
                        DOCUMENT RELEVANCE WARNING
                      </span>
                    </div>
                  )}
                  <ul className="space-y-2">
                    {analysisResults.explanation
                      .split("|")
                      .filter((point: string) => point.trim().length > 0)
                      .map((point: string, index: number) => (
                        <li
                          key={index}
                          className={`text-sm flex items-start ${
                            analysisResults.explanation.includes(
                              "Document Relevance Issue"
                            )
                              ? "text-red-700"
                              : "text-gray-700"
                          }`}
                        >
                          <span
                            className={`mr-2 ${
                              analysisResults.explanation.includes(
                                "Document Relevance Issue"
                              )
                                ? "text-red-600"
                                : "text-blue-600"
                            }`}
                          >
                            •
                          </span>
                          <span>{point.trim()}</span>
                        </li>
                      ))}
                  </ul>
                </div>
              </div>

              {/* Recommendations */}
              <div>
                <h5 className="text-sm font-semibold text-gray-900 mb-2">
                  Recommendations
                </h5>
                <div className="bg-yellow-50 rounded-lg p-3 max-h-40 overflow-y-auto">
                  <ul className="space-y-1">
                    {analysisResults.recommendations.map(
                      (recommendation, index) => (
                        <li
                          key={index}
                          className="text-sm text-gray-700 flex items-start"
                        >
                          <span className="text-yellow-600 mr-2">•</span>
                          <span>{recommendation}</span>
                        </li>
                      )
                    )}
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Loading state during analysis */}
        {isAnalyzing && (
          <div className="mt-4 flex items-center justify-center p-4 bg-blue-50 rounded-lg">
            <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mr-3"></div>
            <span className="text-blue-700 font-medium">
              Analyzing document for ISO 42001 compliance...
            </span>
          </div>
        )}
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
                <span className="font-medium text-gray-700">
                  ISO 42001 Audit
                </span>
              </div>
              <h1 className="text-3xl font-bold text-gray-900">
                ISO 42001 Compliance Audit
              </h1>
              <p className="text-gray-600 mt-1">
                AI Management System Standard
              </p>
            </div>
          </div>
        </header>

        <div className="p-8 space-y-8">
          {/* Dashboard Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Framework Adaptation */}
            <div className="bg-white rounded-xl border border-gray-200 shadow-sm p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <div className="w-10 h-10 bg-blue-100 rounded-xl flex items-center justify-center">
                    <Settings className="w-5 h-5 text-blue-600" />
                  </div>
                  <div className="ml-3">
                    <h3 className="font-semibold text-gray-900">
                      Framework Adaptation
                    </h3>
                    <p className="text-sm text-gray-600">Overall Progress</p>
                  </div>
                </div>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-2xl font-bold text-gray-900">
                  {calculateComplianceScore()}%
                </span>
                <div className="w-20 h-2 bg-gray-200 rounded-full">
                  <div
                    className={`h-2 rounded-full ${getProgressColor(
                      calculateComplianceScore()
                    )}`}
                    style={{ width: `${calculateComplianceScore()}%` }}
                  ></div>
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
                    <h3 className="font-semibold text-gray-900">
                      Required Actions
                    </h3>
                    <p className="text-sm text-gray-600">Pending Items</p>
                  </div>
                </div>
              </div>
              <div className="space-y-2">
                {(() => {
                  const docSubsections = ["ai-policy-documentation", "procedures-guidelines", "record-keeping-system"];
                  const completedDocSubsections = docSubsections.filter(sub => autoSectionsCompleted.has(sub));
                  const allDocCompleted = completedDocSubsections.length === docSubsections.length;
                  const anyDocCompleted = completedDocSubsections.length > 0;
                  
                  return (
                    <div className="flex items-center text-sm">
                      {allDocCompleted ? (
                        <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                      ) : anyDocCompleted ? (
                        <div className="w-4 h-4 mr-2 relative">
                          <div className="w-4 h-4 bg-yellow-100 border-2 border-yellow-500 rounded-full"></div>
                          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-yellow-500 rounded-full"></div>
                        </div>
                      ) : (
                        <AlertCircle className="w-4 h-4 text-yellow-500 mr-2" />
                      )}
                      <span className="text-gray-700">
                        Documentation and Policy Development {anyDocCompleted && `(${completedDocSubsections.length}/${docSubsections.length})`}
                      </span>
                    </div>
                  );
                })()}
                {(() => {
                  const riskSubsections = ["impact-assessment", "risk-mitigation-strategies", "incident-response"];
                  const completedRiskSubsections = riskSubsections.filter(sub => autoSectionsCompleted.has(sub));
                  const allRiskCompleted = completedRiskSubsections.length === riskSubsections.length;
                  const anyRiskCompleted = completedRiskSubsections.length > 0;
                  
                  return (
                    <div className="flex items-center text-sm">
                      {allRiskCompleted ? (
                        <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                      ) : anyRiskCompleted ? (
                        <div className="w-4 h-4 mr-2 relative">
                          <div className="w-4 h-4 bg-yellow-100 border-2 border-yellow-500 rounded-full"></div>
                          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-yellow-500 rounded-full"></div>
                        </div>
                      ) : (
                        <AlertCircle className="w-4 h-4 text-yellow-500 mr-2" />
                      )}
                      <span className="text-gray-700">
                        Risk Assessment and Management {anyRiskCompleted && `(${completedRiskSubsections.length}/${riskSubsections.length})`}
                      </span>
                    </div>
                  );
                })()}
                {(() => {
                  const lifecycleSubsections = ["development-controls", "testing-framework", "deployment-procedures"];
                  const completedLifecycleSubsections = lifecycleSubsections.filter(sub => autoSectionsCompleted.has(sub));
                  const allLifecycleCompleted = completedLifecycleSubsections.length === lifecycleSubsections.length;
                  const anyLifecycleCompleted = completedLifecycleSubsections.length > 0;
                  
                  return (
                    <div className="flex items-center text-sm">
                      {allLifecycleCompleted ? (
                        <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                      ) : anyLifecycleCompleted ? (
                        <div className="w-4 h-4 mr-2 relative">
                          <div className="w-4 h-4 bg-yellow-100 border-2 border-yellow-500 rounded-full"></div>
                          <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-2 h-2 bg-yellow-500 rounded-full"></div>
                        </div>
                      ) : (
                        <AlertCircle className="w-4 h-4 text-yellow-500 mr-2" />
                      )}
                      <span className="text-gray-700">
                        AI System Lifecycle Management {anyLifecycleCompleted && `(${completedLifecycleSubsections.length}/${lifecycleSubsections.length})`}
                      </span>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>

          {/* ISO 42001 Compliance Checklist */}
          <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
            <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
              <h2 className="text-xl font-bold text-gray-900">
                ISO 42001 Compliance Checklist
              </h2>
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
                    description: "Create and maintain AI policy documents",
                  },
                  {
                    id: "procedures-guidelines",
                    title: "Procedures and Guidelines",
                    description: "Develop operational procedures",
                  },
                  {
                    id: "record-keeping-system",
                    title: "Record Keeping System",
                    description: "Implement documentation management",
                  },
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
                    description: "Conduct AI impact analysis",
                  },
                  {
                    id: "risk-mitigation-strategies",
                    title: "Risk Mitigation Strategies",
                    description: "Develop risk management plans",
                  },
                  {
                    id: "incident-response",
                    title: "Incident Response",
                    description: "Create incident handling procedures",
                  },
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
                    description: "Implement development standards",
                  },
                  {
                    id: "testing-framework",
                    title: "Testing Framework",
                    description: "Establish testing protocols",
                  },
                  {
                    id: "deployment-procedures",
                    title: "Deployment Procedures",
                    description: "Define deployment guidelines",
                  },
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
                    description: "Define performance indicators",
                  },
                  {
                    id: "monitoring-systems",
                    title: "Monitoring Systems",
                    description: "Implement monitoring tools",
                  },
                  {
                    id: "reporting-mechanisms",
                    title: "Reporting Mechanisms",
                    description: "Establish reporting procedures",
                  },
                ]
              )}
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

