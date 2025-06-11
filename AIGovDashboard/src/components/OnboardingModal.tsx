import React, { useState } from "react";
import { supabase } from "../lib/supabase";

interface OnboardingModalProps {
  isOpen: boolean;
  onClose: () => void;
  userId: string;
}

const OnboardingModal: React.FC<OnboardingModalProps> = ({
  isOpen,
  onClose,
  userId,
}) => {
  const [onboardingStep, setOnboardingStep] = useState(1);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [onboardingAnswers, setOnboardingAnswers] = useState({
    userType: "",
    aiProficiency: "",
    usageFrequency: "",
    usageContext: "",
    primaryGoal: "",
    industryFocus: "",
  });

  const handleOnboardingChange = (question: string, answer: string) => {
    setOnboardingAnswers({
      ...onboardingAnswers,
      [question]: answer,
    });
  };

  const goToNextStep = () => {
    if (onboardingStep < 6) {
      setOnboardingStep(onboardingStep + 1);
    } else {
      submitOnboardingData();
    }
  };

  const goToPreviousStep = () => {
    if (onboardingStep > 1) {
      setOnboardingStep(onboardingStep - 1);
    }
  };

  const skipOnboarding = async () => {
    setIsSubmitting(true);
    try {
      // Create default profile in Supabase
      console.log("Creating default profile for userId:", userId);

      const defaultProfile = {
        user_id: userId,
        firsttimeuserstatus: false,
        question_1: "Not Specified",
        question_2: "Not Specified",
        question_3: "Not Specified",
        question_4: "Not Specified",
        question_5: "Not Specified",
        question_6: "Not Specified",
      };

      // Use upsert to handle both create and update cases
      const { error } = await supabase.from("userData").upsert(defaultProfile, {
        onConflict: "user_id",
      });

      if (error) {
        console.error("Error creating/updating profile:", error);
        throw error;
      }

      // Remove the flag from localStorage
      localStorage.removeItem("showOnboarding");

      // Close the modal
      onClose();
    } catch (error) {
      console.error("Error skipping onboarding:", error);
      // Still close the modal to prevent users getting stuck
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  const submitOnboardingData = async () => {
    setIsSubmitting(true);
    try {
      console.log("Updating user data for userId:", userId);
      console.log("Onboarding answers:", onboardingAnswers);

      // Update directly to the question fields
      const profileData = {
        user_id: userId,
        firsttimeuserstatus: false,
        question_1: onboardingAnswers.userType,
        question_2: onboardingAnswers.aiProficiency,
        question_3: onboardingAnswers.usageFrequency,
        question_4: onboardingAnswers.usageContext,
        question_5: onboardingAnswers.primaryGoal,
        question_6: onboardingAnswers.industryFocus,
      };

      // Use upsert to handle both create and update cases
      const { error } = await supabase.from("userData").upsert(profileData, {
        onConflict: "user_id",
      });

      if (error) {
        console.error("Error updating profile:", error);
        throw error;
      }

      console.log("Data updated successfully");

      // Remove the flag from localStorage
      localStorage.removeItem("showOnboarding");

      // Close the modal
      onClose();
    } catch (error) {
      console.error("Onboarding error:", error);
      // Still close the modal to prevent users getting stuck
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  // Render onboarding questions based on current step
  const renderOnboardingStep = () => {
    switch (onboardingStep) {
      case 1:
        return (
          <div className="space-y-5">
            <div className="flex justify-center mb-4">
              <div className="w-14 h-14 rounded-full bg-teal-50 flex items-center justify-center">
                <svg
                  className="w-7 h-7 text-teal-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
              </div>
            </div>

            <h3 className="text-xl font-bold text-gray-800 text-center">
              Who are you?
            </h3>
            <p className="text-gray-600 text-center text-sm">
              This helps us tailor your experience.
            </p>

            <div className="space-y-2.5 mt-4">
              {[
                "Student",
                "Professional",
                "Researcher",
                "Entrepreneur",
                "Hobbyist",
              ].map((type) => (
                <div key={type} className="relative">
                  <input
                    type="radio"
                    id={type}
                    name="userType"
                    className="sr-only"
                    checked={onboardingAnswers.userType === type}
                    onChange={() => handleOnboardingChange("userType", type)}
                  />
                  <label
                    htmlFor={type}
                    className={`flex items-center p-3 w-full text-left rounded-xl border cursor-pointer ${
                      onboardingAnswers.userType === type
                        ? "border-teal-600 bg-teal-50"
                        : "border-gray-200 hover:border-teal-300"
                    }`}
                  >
                    <div className="flex items-center">
                      <div
                        className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center border ${
                          onboardingAnswers.userType === type
                            ? "border-teal-600"
                            : "border-gray-300"
                        }`}
                      >
                        {onboardingAnswers.userType === type && (
                          <div className="w-2 h-2 rounded-full bg-teal-600"></div>
                        )}
                      </div>
                      <span className="text-sm font-medium">{type}</span>
                    </div>
                  </label>
                </div>
              ))}
            </div>
          </div>
        );

      case 2:
        return (
          <div className="space-y-5">
            <div className="flex justify-center mb-4">
              <div className="w-14 h-14 rounded-full bg-teal-50 flex items-center justify-center">
                <svg
                  className="w-7 h-7 text-teal-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
              </div>
            </div>

            <h3 className="text-xl font-bold text-gray-800 text-center">
              What's your AI proficiency level?
            </h3>
            <p className="text-gray-600 text-center text-sm">
              This helps us adjust the complexity of our insights.
            </p>

            <div className="space-y-2.5 mt-4">
              {[
                {
                  level: "Beginner",
                  description: "New to AI concepts and technologies",
                },
                {
                  level: "Intermediate",
                  description: "Familiar with basic AI principles",
                },
                {
                  level: "Advanced",
                  description: "Good understanding of AI systems",
                },
                {
                  level: "Expert",
                  description: "Deep technical knowledge of AI",
                },
              ].map((item) => (
                <div key={item.level} className="relative">
                  <input
                    type="radio"
                    id={item.level}
                    name="aiProficiency"
                    className="sr-only"
                    checked={onboardingAnswers.aiProficiency === item.level}
                    onChange={() =>
                      handleOnboardingChange("aiProficiency", item.level)
                    }
                  />
                  <label
                    htmlFor={item.level}
                    className={`flex items-center p-3 w-full text-left rounded-xl border cursor-pointer ${
                      onboardingAnswers.aiProficiency === item.level
                        ? "border-teal-600 bg-teal-50"
                        : "border-gray-200 hover:border-teal-300"
                    }`}
                  >
                    <div className="flex items-center">
                      <div
                        className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center border ${
                          onboardingAnswers.aiProficiency === item.level
                            ? "border-teal-600"
                            : "border-gray-300"
                        }`}
                      >
                        {onboardingAnswers.aiProficiency === item.level && (
                          <div className="w-2 h-2 rounded-full bg-teal-600"></div>
                        )}
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-medium">
                          {item.level}
                        </span>
                        <span className="text-xs text-gray-500">
                          {item.description}
                        </span>
                      </div>
                    </div>
                  </label>
                </div>
              ))}
            </div>
          </div>
        );

      case 3:
        return (
          <div className="space-y-5">
            <div className="flex justify-center mb-4">
              <div className="w-14 h-14 rounded-full bg-teal-50 flex items-center justify-center">
                <svg
                  className="w-7 h-7 text-teal-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                  />
                </svg>
              </div>
            </div>

            <h3 className="text-xl font-bold text-gray-800 text-center">
              How often will you use PRISM?
            </h3>
            <p className="text-gray-600 text-center text-sm">
              This helps us understand your engagement needs.
            </p>

            <div className="space-y-2.5 mt-4">
              {[
                "Daily",
                "Several times a week",
                "Weekly",
                "Monthly",
                "Occasionally",
              ].map((frequency) => (
                <div key={frequency} className="relative">
                  <input
                    type="radio"
                    id={frequency}
                    name="usageFrequency"
                    className="sr-only"
                    checked={onboardingAnswers.usageFrequency === frequency}
                    onChange={() =>
                      handleOnboardingChange("usageFrequency", frequency)
                    }
                  />
                  <label
                    htmlFor={frequency}
                    className={`flex items-center p-3 w-full text-left rounded-xl border cursor-pointer transition-all duration-200 ${
                      onboardingAnswers.usageFrequency === frequency
                        ? "border-teal-600 bg-teal-50"
                        : "border-gray-200 hover:border-teal-300"
                    }`}
                  >
                    <div className="flex items-center">
                      <div
                        className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center border ${
                          onboardingAnswers.usageFrequency === frequency
                            ? "border-teal-600"
                            : "border-gray-300"
                        }`}
                      >
                        {onboardingAnswers.usageFrequency === frequency && (
                          <div className="w-2 h-2 rounded-full bg-teal-600"></div>
                        )}
                      </div>
                      <span className="text-sm font-medium">{frequency}</span>
                    </div>
                  </label>
                </div>
              ))}
            </div>
          </div>
        );

      case 4:
        return (
          <div className="space-y-5">
            <div className="flex justify-center mb-4">
              <div className="w-14 h-14 rounded-full bg-teal-50 flex items-center justify-center">
                <svg
                  className="w-7 h-7 text-teal-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M21 13.255A23.931 23.931 0 0112 15c-3.183 0-6.22-.62-9-1.745M16 6V4a2 2 0 00-2-2h-4a2 2 0 00-2 2v2m4 6h.01M5 20h14a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                  />
                </svg>
              </div>
            </div>

            <h3 className="text-xl font-bold text-gray-800 text-center">
              How will you use PRISM?
            </h3>
            <p className="text-gray-600 text-center text-sm">
              This helps us understand your context.
            </p>

            <div className="space-y-2.5 mt-4">
              {[
                {
                  context: "Work/Professional",
                  description: "For business use cases",
                },
                {
                  context: "Academic Research",
                  description: "For scholarly activities",
                },
                {
                  context: "Personal Interest",
                  description: "For personal growth",
                },
                {
                  context: "Policy Development",
                  description: "For creating guidelines",
                },
                {
                  context: "Compliance",
                  description: "For regulatory adherence",
                },
              ].map(({ context, description }) => (
                <div key={context} className="relative">
                  <input
                    type="radio"
                    id={context}
                    name="usageContext"
                    className="sr-only"
                    checked={onboardingAnswers.usageContext === context}
                    onChange={() =>
                      handleOnboardingChange("usageContext", context)
                    }
                  />
                  <label
                    htmlFor={context}
                    className={`flex items-center p-3 w-full text-left rounded-xl border cursor-pointer transition-all duration-200 ${
                      onboardingAnswers.usageContext === context
                        ? "border-teal-600 bg-teal-50"
                        : "border-gray-200 hover:border-teal-300"
                    }`}
                  >
                    <div className="flex items-center">
                      <div
                        className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center border ${
                          onboardingAnswers.usageContext === context
                            ? "border-teal-600"
                            : "border-gray-300"
                        }`}
                      >
                        {onboardingAnswers.usageContext === context && (
                          <div className="w-2 h-2 rounded-full bg-teal-600"></div>
                        )}
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-medium">{context}</span>
                        <span className="text-xs text-gray-500">
                          {description}
                        </span>
                      </div>
                    </div>
                  </label>
                </div>
              ))}
            </div>
          </div>
        );

      case 5:
        return (
          <div className="space-y-5">
            <div className="flex justify-center mb-4">
              <div className="w-14 h-14 rounded-full bg-teal-50 flex items-center justify-center">
                <svg
                  className="w-7 h-7 text-teal-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M13 10V3L4 14h7v7l9-11h-7z"
                  />
                </svg>
              </div>
            </div>

            <h3 className="text-xl font-bold text-gray-800 text-center">
              What's your primary goal with PRISM?
            </h3>
            <p className="text-gray-600 text-center text-sm">
              This helps us highlight relevant features.
            </p>

            <div className="space-y-2.5 mt-4">
              {[
                {
                  goal: "Stay informed on AI regulations",
                  description: "Keep updated on governance",
                },
                {
                  goal: "Ensure compliance",
                  description: "Meet regulatory requirements",
                },
                {
                  goal: "Competitive analysis",
                  description: "Understand the industry landscape",
                },
                {
                  goal: "Research",
                  description: "Deepen knowledge in AI governance",
                },
                {
                  goal: "Education",
                  description: "Learning and teaching purposes",
                },
                {
                  goal: "Product development",
                  description: "Create compliant AI solutions",
                },
              ].map(({ goal, description }) => (
                <div key={goal} className="relative">
                  <input
                    type="radio"
                    id={goal}
                    name="primaryGoal"
                    className="sr-only"
                    checked={onboardingAnswers.primaryGoal === goal}
                    onChange={() => handleOnboardingChange("primaryGoal", goal)}
                  />
                  <label
                    htmlFor={goal}
                    className={`flex items-center p-3 w-full text-left rounded-xl border cursor-pointer transition-all duration-200 ${
                      onboardingAnswers.primaryGoal === goal
                        ? "border-teal-600 bg-teal-50"
                        : "border-gray-200 hover:border-teal-300"
                    }`}
                  >
                    <div className="flex items-center">
                      <div
                        className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center border ${
                          onboardingAnswers.primaryGoal === goal
                            ? "border-teal-600"
                            : "border-gray-300"
                        }`}
                      >
                        {onboardingAnswers.primaryGoal === goal && (
                          <div className="w-2 h-2 rounded-full bg-teal-600"></div>
                        )}
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-medium">{goal}</span>
                        <span className="text-xs text-gray-500">
                          {description}
                        </span>
                      </div>
                    </div>
                  </label>
                </div>
              ))}
            </div>
          </div>
        );

      case 6:
        return (
          <div className="space-y-5">
            <div className="flex justify-center mb-4">
              <div className="w-14 h-14 rounded-full bg-teal-50 flex items-center justify-center">
                <svg
                  className="w-7 h-7 text-teal-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4"
                  />
                </svg>
              </div>
            </div>

            <h3 className="text-xl font-bold text-gray-800 text-center">
              Which industry interests you most?
            </h3>
            <p className="text-gray-600 text-center text-sm">
              This helps us curate relevant content.
            </p>

            <div className="space-y-2.5 mt-4">
              {[
                {
                  industry: "Financial Services",
                  description: "Banking, insurance, fintech",
                },
                {
                  industry: "Healthcare",
                  description: "Medical, biotech, health systems",
                },
                {
                  industry: "Government & Public Policy",
                  description: "Public sector and policy",
                },
                {
                  industry: "Technology",
                  description: "Software, hardware, IT services",
                },
                {
                  industry: "Education",
                  description: "Schools, universities, edtech",
                },
                {
                  industry: "Manufacturing",
                  description: "Production, automotive, industrial",
                },
                { industry: "Retail", description: "Commerce, consumer goods" },
                {
                  industry: "Other",
                  description: "Other industries not listed",
                },
              ].map(({ industry, description }) => (
                <div key={industry} className="relative">
                  <input
                    type="radio"
                    id={industry}
                    name="industryFocus"
                    className="sr-only"
                    checked={onboardingAnswers.industryFocus === industry}
                    onChange={() =>
                      handleOnboardingChange("industryFocus", industry)
                    }
                  />
                  <label
                    htmlFor={industry}
                    className={`flex items-center p-3 w-full text-left rounded-xl border cursor-pointer transition-all duration-200 ${
                      onboardingAnswers.industryFocus === industry
                        ? "border-teal-600 bg-teal-50"
                        : "border-gray-200 hover:border-teal-300"
                    }`}
                  >
                    <div className="flex items-center">
                      <div
                        className={`w-4 h-4 rounded-full mr-3 flex items-center justify-center border ${
                          onboardingAnswers.industryFocus === industry
                            ? "border-teal-600"
                            : "border-gray-300"
                        }`}
                      >
                        {onboardingAnswers.industryFocus === industry && (
                          <div className="w-2 h-2 rounded-full bg-teal-600"></div>
                        )}
                      </div>
                      <div className="flex flex-col">
                        <span className="text-sm font-medium">{industry}</span>
                        <span className="text-xs text-gray-500">
                          {description}
                        </span>
                      </div>
                    </div>
                  </label>
                </div>
              ))}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-[9999] flex items-center justify-center p-4 bg-black bg-opacity-50 backdrop-blur-sm">
      <div
        className="bg-white w-full max-w-md rounded-2xl shadow-xl overflow-hidden"
        style={{ maxHeight: "85vh" }}
      >
        <div className="p-5">
          {/* Header with logo */}
          <div className="flex items-center justify-center space-x-2 mb-4">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="text-teal-600"
            >
              <path
                d="M12 2L2 7L12 12L22 7L12 2Z"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M2 17L12 22L22 17"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d="M2 12L12 17L22 12"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
            <h1 className="text-lg font-bold text-gray-900">
              PRISM by Block Convey
            </h1>
          </div>

          {/* Progress indicator */}
          <div className="mb-4">
            <div className="flex justify-between items-center mb-1.5">
              <span className="text-xs font-medium text-gray-600">
                Step {onboardingStep} of 6
              </span>
              <span className="text-xs font-medium text-teal-600">
                {Math.round((onboardingStep / 6) * 100)}% Complete
              </span>
            </div>
            <div className="h-1.5 w-full bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-1.5 bg-teal-500 rounded-full transition-all duration-300"
                style={{ width: `${(onboardingStep / 6) * 100}%` }}
              ></div>
            </div>
          </div>

          {/* Questions - optimized for better visibility with custom styling */}
          <div
            className="overflow-y-auto pr-1"
            style={{
              maxHeight: "calc(85vh - 180px)",
              scrollbarWidth: "thin",
              scrollbarColor: "#99f6e4 #f3f4f6",
            }}
          >
            {renderOnboardingStep()}
          </div>

          {/* Buttons - updated with prominent skip option */}
          <div className="mt-4 flex space-x-3">
            {onboardingStep > 1 ? (
              <button
                onClick={goToPreviousStep}
                className="flex-shrink-0 bg-white border border-gray-200 text-gray-700 py-2.5 px-4 rounded-lg text-sm font-medium hover:bg-gray-50 hover:border-gray-300 transition-colors duration-200 flex items-center shadow-sm"
              >
                <svg
                  className="mr-1.5 h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M15 19l-7-7 7-7"
                  />
                </svg>
                Back
              </button>
            ) : (
              <div></div> // Empty div for spacing when on first step
            )}

            {onboardingStep < 6 ? (
              <button
                onClick={goToNextStep}
                disabled={
                  (onboardingStep === 1 && !onboardingAnswers.userType) ||
                  (onboardingStep === 2 && !onboardingAnswers.aiProficiency) ||
                  (onboardingStep === 3 && !onboardingAnswers.usageFrequency) ||
                  (onboardingStep === 4 && !onboardingAnswers.usageContext) ||
                  (onboardingStep === 5 && !onboardingAnswers.primaryGoal) ||
                  isSubmitting
                }
                className="flex-grow bg-teal-600 text-white py-2.5 px-4 rounded-lg text-sm font-medium hover:bg-teal-700 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-sm"
              >
                Continue{" "}
                <svg
                  className="ml-2 h-4 w-4"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="2"
                    d="M9 5l7 7-7 7"
                  />
                </svg>
              </button>
            ) : (
              <button
                onClick={submitOnboardingData}
                disabled={!onboardingAnswers.industryFocus || isSubmitting}
                className="flex-grow bg-teal-600 text-white py-2.5 px-4 rounded-lg text-sm font-medium hover:bg-teal-700 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center shadow-sm"
              >
                {isSubmitting ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                      xmlns="http://www.w3.org/2000/svg"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      ></circle>
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      ></path>
                    </svg>
                    Completing...
                  </>
                ) : (
                  "Complete"
                )}
              </button>
            )}
          </div>

          {/* Clear skip button at the bottom */}
          <div className="mt-3 text-center">
            <button
              onClick={skipOnboarding}
              disabled={isSubmitting}
              className="text-sm text-gray-500 hover:text-teal-600 font-medium transition-colors duration-200 py-1 px-3"
            >
              Skip for now
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default OnboardingModal;