import { useState, useRef, useEffect } from 'react';
import { Sparkles, Send } from 'lucide-react';
import { getExplanation } from '../services/AIExplanationService';
import { 
  Dialog, 
  DialogContent, 
  DialogHeader, 
  DialogTitle,
  DialogDescription
} from './ui/dialog';
import DOMPurify from 'isomorphic-dompurify';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface InfoTooltipProps {
  title: string;
  entityType: 'chart' | 'metric';
  entityName: string;
  data?: {
    value?: number;
    status?: string;
    description?: string;
    isPercentage?: boolean;
    chartData?: any[];
  };
}

interface Message {
  type: 'assistant' | 'user';
  content: string;
  id: string;
}

export const InfoTooltip: React.FC<InfoTooltipProps> = ({ 
  title, 
  entityType, 
  entityName,
  data
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [userInput, setUserInput] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Generate a unique ID for messages
  const generateId = () => `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

  useEffect(() => {
    // Auto-focus input when dialog opens
    if (isOpen && inputRef.current && messages.length > 0) {
      inputRef.current.focus();
    }
  }, [isOpen, messages.length]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const processResponse = (response: any): string => {
    // Handle case where response might be JSON string or already parsed
    try {
      const parsedResponse = typeof response === 'string' ? JSON.parse(response) : response;
      let content = parsedResponse.answer || parsedResponse.response || '';
      
      // Clean up the content
      if (typeof content === 'string') {
        if (content.includes('{"response":')) {
          content = content.replace(/^\s*\{\"response\":\s*|\}\s*$/g, '');
          content = content.replace(/^\"|\"\s*$/g, '');
        }
        
        // Replace escaped Unicode characters with their actual characters
        content = content
          .replace(/\\n/g, '\n')
          .replace(/\\"/g, '"')
          // Replace Unicode escape sequences with actual characters
          .replace(/\\u([0-9a-fA-F]{4})/g, (match: string, code: string) => 
            String.fromCharCode(parseInt(code, 16))
          );
          
        // Decode HTML entities
        const decodeEntities = (text: string): string => {
          const textarea = document.createElement('textarea');
          textarea.innerHTML = text;
          return textarea.value;
        };
        
        content = decodeEntities(content);
      }
      
      return content;
    } catch (e) {
      // Fallback to using the raw response if parsing fails
      return typeof response === 'object' ? 
        (response.answer || '') : 
        response;
    }
  };

  const handleInfoClick = async () => {
    setIsOpen(true);
    setIsLoading(true);
    setMessages([]);
    
    // Extract data values for context
    const metricValue = data?.value;
    const metricStatus = data?.status;
    const metricDescription = data?.description;
    const isMetricPercentage = data?.isPercentage;
    const chartData = data?.chartData;
    
    // Formulate context-aware question
    let question = '';
    if (entityType === 'chart') {
      // Enhanced chart data context
      const dataContext = chartData 
        ? `\n\nHere is the detailed data for this chart: ${JSON.stringify(chartData, null, 2)}`
        : (data ? `\n\nHere is the data for this chart: ${JSON.stringify(data, null, 2)}` : '');
      
       
          question = `You are a senior AI evaluation engineer preparing an investor-facing and internal audit report on the '${entityName}' chart produced by our model evaluation system.
        
        This analysis must demonstrate the robustness, explainability, fairness monitoring, and operational stability of our AI models in compliance with leading regulatory frameworks (e.g., EU AI Act, GDPR, PCI-DSS AI requirements).
        
        Write a highly structured, expert-level report following the format below:
        
        ---
        
        # 1. Definition 
        Clearly define what the '${entityName}' chart measures and why it is a critical component in evaluating AI model performance, fairness, or risk exposure.
        
        # 2. Interpretation of Observed Values 
        Summarize what the observed chart data ${dataContext} indicates about model behavior, trustworthiness, performance stability, or compliance risk.
        
        # 3. Technical Methodology Overview
        Provide a formal, structured explanation of how the '${entityName}' was computed according to our Model Evaluation Framework (MLService backend):
        
        - For ROC-AUC:
          - Probability scores generated using \`predict_proba\` if available; fallback to min-max normalization of raw outputs.
          - ROC curve calculated based on true positive rate (TPR) and false positive rate (FPR) across thresholds.
          - Area Under the Curve (AUC) computed to quantify overall classification separability.
          
        - For Drift Detection:
          - Feature drift for categorical data assessed via Chi-Square test between train/test distributions.
          - Feature drift for numerical data assessed via Kolmogorov-Smirnov (KS) test comparing continuous distributions.
        
        - For Fairness Metrics:
          - Sensitive attributes with low cardinality (≤10 unique values) segmented.
          - Per-group confusion matrix breakdown calculated.
          - Demographic Parity, Equal Opportunity, Disparate Impact, Equalized Odds, and Statistical Parity metrics computed.
        
        - For Explainability Metrics (SHAP):
          - TreeExplainer applied to sampled test data (≤100 rows).
          - SHAP values computed and global feature importances aggregated.
        
        - For Cross-Validation:
          - Stratified 5-fold KFold used for robust estimation.
          - Accuracy (classification) or R² (regression) computed and summarized.
        
        Reference all computation techniques precisely. Ensure strict traceability to regulatory audit standards.
        
        # 4. Risk Identification and Strategic Action Plan
        Systematically identify risks and recommend proactive mitigation strategies:
        
        - **Detected Risk:** Clearly state any risk detected (e.g., drift, bias, model decay, fairness gap, operational instability).
        - **Severity Level:** Classify the risk based on impact likelihood and potential model harm:
          - Critical: Immediate risk to reliability, fairness, or compliance.
          - High: High likelihood of operational/model degradation if left unaddressed.
          - Medium: Moderate concern; should be monitored.
          - Low: Minor risk but worth observing over time.
        
        - **Recommended Strategic Action:**
          - For Critical risks: Immediate retraining, model deprecation, or feature audits.
          - For High risks: Deploy additional monitoring, initiate retraining pipelines, or apply fairness corrections.
          - For Medium risks: Enhance monitoring granularity and schedule model refresh.
          - For Low risks: Continue passive monitoring with periodic reviews.
        
        Risk/Action recommendations must be measurable, actionable, and framed in operational governance language.
        
        # 5. Commitment Statement (Final 1-2 sentences)
        Conclude by emphasizing our organization's commitment to proactive AI governance, operational robustness, fairness, explainability, and continuous model risk management.
        
        ---
        ---

# Scope Restriction
- Only describe the specific methods used to compute the '${entityName}'.
- Do not reference fairness metrics, explainability methods (SHAP), drift detection, or cross-validation unless they are intrinsically part of the '${entityName}' computation.
- Keep the report strictly focused and technically scoped.

---

        
        # Formatting Guidelines
        - Write the full response in Markdown format.
        - Use proper header levels (#, ##, ###) as per sections.
        - Bullet points are allowed **only** under "Technical Methodology Overview" and "Risk Identification and Strategic Action Plan" for better clarity.
        - Maintain a formal, regulatory-aligned, investor-grade tone throughout.
        - Focus on measurable insights, traceability, and audit-readiness.
        
        This report will be used in both investor briefings and internal/external AI audits. Precision, structure, and depth are mandatory.`;
        } else {
          const valueDisplay = metricValue !== undefined && metricValue !== null ?
            `${metricValue}${isMetricPercentage ? '%' : ''}` : 'N/A';
          const statusDisplay = metricStatus || 'N/A';
          const descriptionDisplay = metricDescription || 'N/A';
        
          const metricContext = `\n\nHere is the current metric data:
        - Value: ${valueDisplay}
        - Status: ${statusDisplay}
        - Description: ${descriptionDisplay}`;
        
          const analysisRequest = metricValue !== undefined && metricValue !== null && metricStatus ?
            `Additionally, interpret what the specific metric value of ${valueDisplay} with status "${statusDisplay}" indicates for model operational health and compliance.` : '';
        
          question = `You are a senior AI evaluation engineer preparing an investor-facing and internal audit report on the '${entityName}' metric produced by our model evaluation system.
        
        This analysis must demonstrate the robustness, explainability, fairness monitoring, and operational stability of our AI models in compliance with leading regulatory frameworks (e.g., EU AI Act, GDPR, PCI-DSS AI requirements).
        
        Write a highly structured, expert-level report following the format below:
        
        ---
        
        # 1. Definition 
        Clearly define what the '${entityName}' metric measures and why it is a critical component in evaluating AI model performance, fairness, or risk exposure.
        
        # 2. Interpretation of Metric Value 
        Summarize what the current metric data ${valueDisplay} indicates about model behavior, operational performance, or risk signals. ${analysisRequest}
        
        # 3. Technical Methodology Overview
        Provide a formal, structured explanation of how the '${entityName}' metric was computed based on our MLService evaluation framework:
        
        - For Classification Metrics:
          - Predicted probabilities thresholded at 0.5 to generate binary predictions.
          - Accuracy, Precision, Recall, and F1 scores computed using scikit-learn with weighted averaging.
        
        - For Fairness Metrics:
          - Sensitive feature segmentation.
          - Group-specific confusion matrices calculated.
          - Fairness disparity ratios and statistical parity differences derived.
        
        - For Drift Metrics:
          - Feature drift assessed via Chi-Square (categorical) and KS tests (numerical).
          - Label distribution drift evaluated via Chi-Square on label frequencies.
        
        - For Regression Metrics:
          - Residuals computed as actual vs. predicted values.
          - MSE, RMSE, MAE, and R² calculated.
        
        - For Explainability (SHAP):
          - TreeExplainer applied to a subset of test samples.
          - SHAP values generated to infer feature attribution.
        
        - For Cross-Validation:
          - 5-fold KFold used with consistent scoring metrics based on problem type.
        
        Be precise, ensuring every computation is auditable.
        
        # 4. Risk Identification and Strategic Action Plan
        Systematically identify risks and propose mitigation strategies:
        
        - **Detected Risk:** Identify drift, fairness gaps, instability, underperformance, or any compliance-relevant issues.
        - **Severity Level:**
          - Critical, High, Medium, Low (based on operational and compliance impact).
        
        - **Recommended Strategic Action:**
          - Tailored based on severity, focusing on retraining, monitoring, remediation, or further evaluation.
        
        Recommendations must be clear, actionable, and governance-ready.
        
        # 5. Commitment Statement (Final 1-2 sentences)
        Conclude by reaffirming our dedication to proactive model governance, transparency, fairness, and continuous improvement of AI system resilience.
        
        ---
        ---

# Scope Restriction
- Only describe the specific methods used to compute the '${entityName}'.
- Do not reference fairness metrics, explainability methods (SHAP), drift detection, or cross-validation unless they are intrinsically part of the '${entityName}' computation.
- Keep the report strictly focused and technically scoped.

---

        
        # Formatting Guidelines
        - Markdown format required.
        - Structured headers and bullet points (only inside methodology and risk/action sections).
        - Maintain audit-validated and investor-credible language.
        - Focus on traceability, operational risk readiness, and governance leadership.
        
        This report will be presented to investors and used in official AI audit documentation. Precision and audit-traceable structure are mandatory.${metricContext}`;
        }
        
    
    try {
      const response = await getExplanation(question);
      const content = processResponse(response);
      
      // Create a new message with unique ID
      const messageId = generateId();
      setMessages([{ type: 'assistant', content, id: messageId }]);
      
      // Log the content to check for special characters
      console.log('Processed content:', content);
    } catch (error) {
      console.error('Error getting explanation:', error);
      const errorMessage = 'Sorry, I couldn\'t retrieve an explanation at this time. Please try again later.';
      const messageId = generateId();
      setMessages([{ type: 'assistant', content: errorMessage, id: messageId }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!userInput.trim()) return;
    
    // Add user message with unique ID
    const userMessage = userInput.trim();
    const userMessageId = generateId();
    setMessages(prevMessages => [...prevMessages, { type: 'user', content: userMessage, id: userMessageId }]);
    setUserInput('');
    setIsLoading(true);
    
    // Get chart data if available
    const chartData = data?.chartData;
    const chartDataContext = chartData 
      ? `\n\nChart Data: ${JSON.stringify(chartData, null, 2)}`
      : '';
    
    // Create contextual prompt
    const contextualizedQuestion = `The following is a conversation between a user and an AI assistant about ${entityName} in AI model evaluation.
    
Previous conversation:
${messages.map(msg => `${msg.type === 'user' ? 'User' : 'Assistant'}: ${msg.content}`).join('\n')}

User: ${userMessage}

Please provide a helpful, concise response about ${entityName} focusing on the user's specific question. 
Use the chart data provided to give specific insights when appropriate.
Use markdown formatting for clear structure.${chartDataContext}`;
    
    try {
      const response = await getExplanation(contextualizedQuestion);
      const content = processResponse(response);
      
      // Log the content to check for special characters
      console.log('Follow-up response content:', content);
      
      // Add assistant message with unique ID
      const assistantMessageId = generateId();
      setMessages(prevMessages => [
        ...prevMessages, 
        { type: 'assistant', content, id: assistantMessageId }
      ]);
    } catch (error) {
      console.error('Error getting response:', error);
      const errorMessage = 'Sorry, I couldn\'t generate a response at this time. Please try again later.';
      const assistantMessageId = generateId();
      setMessages(prevMessages => [
        ...prevMessages, 
        { type: 'assistant', content: errorMessage, id: assistantMessageId }
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  // Enhanced markdown renderer
  const renderMarkdown = (content: string) => {
    // Configure DOMPurify to keep UTF-8 and special characters
    const sanitizedContent = DOMPurify.sanitize(content, {
      ADD_ATTR: ['target'],
      USE_PROFILES: { html: true }
    });
    
    return (
      <ReactMarkdown 
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({children}) => <p className="mb-3 text-gray-700">{children}</p>,
          h1: ({children}) => <h1 className="text-xl text-indigo-700 font-bold mt-4 mb-2">{children}</h1>,
          h2: ({children}) => <h2 className="text-lg text-indigo-700 font-bold mt-4 mb-2">{children}</h2>,
          h3: ({children}) => <h3 className="text-base text-indigo-700 font-bold mt-4 mb-2">{children}</h3>,
          strong: ({children}) => <strong className="text-indigo-700 font-semibold">{children}</strong>,
          em: ({children}) => <em className="text-gray-600 italic">{children}</em>,
          ul: ({children}) => <ul className="my-2 list-disc pl-5">{children}</ul>,
          ol: ({children}) => <ol className="my-2 list-decimal pl-5">{children}</ol>,
          li: ({children}) => <li className="my-1 ml-2">{children}</li>,
          a: ({href, children}) => <a href={href} className="text-blue-600 hover:underline" target="_blank" rel="noopener noreferrer">{children}</a>,
          code: ({children, className}) => <code className={`bg-gray-100 text-indigo-600 px-1 py-0.5 rounded ${className || ''}`}>{children}</code>,
          pre: ({children}) => <pre className="bg-gray-100 p-3 rounded-md my-3 overflow-auto">{children}</pre>,
          table: ({children}) => <table className="border-collapse border border-gray-300 my-4 w-full">{children}</table>,
          thead: ({children}) => <thead className="bg-gray-100">{children}</thead>,
          tbody: ({children}) => <tbody>{children}</tbody>,
          tr: ({children}) => <tr className="border-b border-gray-300">{children}</tr>,
          th: ({children}) => <th className="border border-gray-300 px-4 py-2 text-left font-semibold">{children}</th>,
          td: ({children}) => <td className="border border-gray-300 px-4 py-2">{children}</td>,
        }}
      >
        {sanitizedContent}
      </ReactMarkdown>
    );
  };

  return (
    <>
      <button
        onClick={handleInfoClick}
        className="text-indigo-600 hover:text-indigo-800 transition-colors inline-flex items-center ml-2 p-1.5 rounded-full bg-indigo-50 hover:bg-indigo-100 shadow-sm hover:shadow-md transition-all duration-200"
        aria-label={`Get AI insights about ${title}`}
      >
        <Sparkles className="h-4 w-4" />
      </button>

      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent className="sm:max-w-3xl bg-gradient-to-br from-white via-slate-50 to-indigo-50 border border-indigo-100 shadow-[0_20px_60px_rgba(79,70,229,0.15),0_0_30px_rgba(147,197,253,0.3)] rounded-2xl max-h-[80vh] overflow-hidden p-0 transition-all duration-300 ease-in-out">
          <div className="bg-gradient-to-r from-indigo-600 to-violet-600 rounded-t-2xl p-5 text-white shadow-md">
            <DialogHeader>
              <DialogTitle className="text-2xl font-bold flex items-center">
                <span className="bg-white/15 p-2 rounded-full mr-3 backdrop-blur-md shadow-inner border border-white/30 flex items-center justify-center">
                  <Sparkles className="h-6 w-6" />
                </span>
                <span className="bg-clip-text text-transparent bg-gradient-to-r from-white to-indigo-100">
                  {title}
                </span>
              </DialogTitle>
              {isLoading && !messages.length && (
                <DialogDescription className="text-indigo-100 mt-2 ml-12 animate-pulse">
                  Generating insights...
                </DialogDescription>
              )}
            </DialogHeader>
          </div>
          
          <div className="p-6 overflow-y-auto max-h-[50vh] scrollbar-thin scrollbar-thumb-indigo-400 scrollbar-track-slate-100 bg-white/50 backdrop-blur-sm">
            {isLoading && !messages.length ? (
              <div className="flex flex-col items-center justify-center py-12">
                <div className="relative w-16 h-16">
                  <div className="absolute top-0 left-0 w-16 h-16 border-4 border-indigo-200 rounded-full animate-ping opacity-75"></div>
                  <div className="absolute top-0 left-0 w-16 h-16 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin"></div>
                </div>
                <p className="mt-6 text-indigo-600 font-medium animate-pulse">Analyzing information...</p>
              </div>
            ) : (
              <div className="relative chat-container space-y-6">
                <div className="absolute -top-10 -left-6 w-24 h-24 rounded-full bg-indigo-300/20 opacity-50 blur-2xl"></div>
                <div className="absolute -bottom-8 -right-6 w-32 h-32 rounded-full bg-violet-300/20 opacity-50 blur-2xl"></div>
                
                {messages.map((message) => (
                  <div 
                    key={message.id} 
                    className={`chat-message ${
                      message.type === 'assistant' 
                        ? 'assistant-message bg-gradient-to-br from-white to-indigo-50/70 border border-indigo-200/70 shadow-sm transform transition-all duration-500 ease-out animate-fadeInUp' 
                        : 'user-message bg-gradient-to-br from-indigo-500/90 to-violet-500/90 border border-indigo-300/30 ml-auto max-w-[80%] transform transition-all duration-500 ease-out animate-fadeInLeft'
                    } p-4 rounded-2xl shadow-sm backdrop-blur-sm mb-4`}
                  >
                    <div className="flex items-start">
                      {message.type === 'assistant' && (
                        <div className="flex-shrink-0 h-9 w-9 rounded-full bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center mr-3 shadow-md">
                          <span className="text-white text-xs font-bold">AI</span>
                        </div>
                      )}
                      <div className={`${message.type === 'assistant' ? 'text-gray-700' : 'text-white'} relative z-10 w-full`}>
                        {message.type === 'assistant' ? (
                          <div className="max-w-none">
                            {renderMarkdown(message.content)}
                          </div>
                        ) : (
                          <p>{message.content}</p>
                        )}
                      </div>
                      {message.type === 'user' && (
                        <div className="flex-shrink-0 h-9 w-9 rounded-full bg-gradient-to-tr from-slate-600 to-slate-700 flex items-center justify-center ml-3 shadow-md">
                          <span className="text-white text-xs font-bold">You</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {isLoading && messages.length > 0 && (
                  <div className="chat-message assistant-message bg-gradient-to-br from-white to-indigo-50/70 border border-indigo-200/70 p-4 rounded-2xl shadow-sm backdrop-blur-sm mb-4 animate-fadeIn">
                    <div className="flex items-start">
                      <div className="flex-shrink-0 h-9 w-9 rounded-full bg-gradient-to-tr from-indigo-500 to-violet-500 flex items-center justify-center mr-3 shadow-md">
                        <span className="text-white text-xs font-bold">AI</span>
                      </div>
                      <div className="flex space-x-2 items-center h-8">
                        <div className="w-2.5 h-2.5 rounded-full bg-indigo-400 animate-bounce" style={{ animationDelay: '0ms' }}></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-indigo-500 animate-bounce" style={{ animationDelay: '200ms' }}></div>
                        <div className="w-2.5 h-2.5 rounded-full bg-indigo-600 animate-bounce" style={{ animationDelay: '400ms' }}></div>
                      </div>
                    </div>
                  </div>
                )}
                
                <div ref={messagesEndRef} />
              </div>
            )}
          </div>
          
          <div className="p-4 bg-gradient-to-r from-slate-50 to-indigo-50 border-t border-indigo-100/50 flex flex-col shadow-inner">
            {messages.length > 0 && (
              <div className="flex items-center gap-2 mb-4">
                <input
                  ref={inputRef}
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask a follow-up question..."
                  className="flex-1 bg-white text-gray-700 rounded-lg px-4 py-2.5 border border-indigo-200/50 focus:outline-none focus:ring-2 focus:ring-indigo-400/30 focus:border-indigo-400 placeholder-gray-400 shadow-sm transition-all duration-200"
                  disabled={isLoading}
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !userInput.trim()}
                  className={`p-2.5 rounded-lg flex items-center justify-center ${
                    isLoading || !userInput.trim()
                      ? 'bg-slate-200 text-slate-400 cursor-not-allowed'
                      : 'bg-gradient-to-r from-indigo-500 to-violet-500 text-white hover:shadow-md hover:from-indigo-600 hover:to-violet-600 transition-all duration-300'
                  }`}
                >
                  <Send className="h-5 w-5" />
                </button>
              </div>
            )}
            
            <div className="flex justify-end">
              <button 
                onClick={() => setIsOpen(false)}
                className="px-5 py-2.5 bg-gradient-to-r from-indigo-500 to-violet-500 text-white rounded-lg font-medium text-sm hover:shadow-md hover:from-indigo-600 hover:to-violet-600 transition-all duration-300 ease-in-out transform hover:scale-105 focus:ring-2 focus:ring-indigo-400 focus:ring-opacity-50 focus:outline-none"
              >
                {messages.length > 0 ? 'Close' : 'Got it'}
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </>
  );
}; 