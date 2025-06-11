import React, { useState, useEffect } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { supabase } from "../lib/supabase";
import { useAuth } from "../contexts/AuthContext";

const SignupPage: React.FC = () => {
  const navigate = useNavigate();
  const { signInWithGoogle } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [organization, setOrganization] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Add state for the onboarding form
  const [showOnboarding, setShowOnboarding] = useState(false);
  const [onboardingStep, setOnboardingStep] = useState(1);
  const [onboardingAnswers, setOnboardingAnswers] = useState({
    userType: "",
    aiProficiency: "",
    usageFrequency: "",
    usageContext: "",
    primaryGoal: "",
    industryFocus: "",
  });

  // Remove card-related state
  const [checkingAuth, setCheckingAuth] = useState(true);

  const [searchParams] = useSearchParams();
  const showOnboardingParam = searchParams.get("onboarding");

  useEffect(() => {
    if (showOnboardingParam === "true") {
      setShowOnboarding(true);
    }
  }, [showOnboardingParam]);

  useEffect(() => {
    const checkAuth = async () => {
      const {
        data: { session },
      } = await supabase.auth.getSession();
      if (session) {
        navigate("/home");
      }
      setCheckingAuth(false);
    };

    checkAuth();
  }, [navigate]);

  // Add this early return
  if (checkingAuth) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-600"></div>
      </div>
    );
  }

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Form validation
    if (!email || !password || !confirmPassword || !fullName) {
      setError("Please fill in all required fields");
      return;
    }

    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }

    if (password.length < 8) {
      setError("Password must be at least 8 characters long");
      return;
    }

    setIsLoading(true);
    const response2 = await fetch("https://prism-backend-dtcc-dot-block-convey-p1.uc.r.appspot.com/auth/signin", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        email: "admin@gmail.com",
        password: "admin",
      }),
    });

    if (!response2.ok) {
      throw new Error("Failed to fetch local token");
    }

    const tokenData = await response2.json();
    // Store the access token in localStorage
    localStorage.setItem("access_token", tokenData.access_token);
    console.log("Access token stored in localStorage:", tokenData.access_token);
    try {
      // Create the user account
      const { data, error: signUpError } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            full_name: fullName,
            organization: organization || null,
          },
        },
      });

      if (signUpError) throw signUpError;

      if (data && data.user) {
        // Store the user ID in local storage
        localStorage.setItem("userId", data.user.id);

        // IMPORTANT: Set flag to show onboarding modal on HomePage
        localStorage.setItem("showOnboarding", "true");
        console.log("Setting showOnboarding flag to show modal on HomePage");

        // Create a record in the users table (if needed)
        await supabase.from("users").insert([
          {
            id: data.user.id,
            email: email,
            full_name: fullName,
            organization: organization || null,
            created_at: new Date().toISOString(),
          },
        ]);

        // Redirect to the homepage
        navigate("/home"); // Make sure this matches your route for HomePage.tsx
        setIsLoading(false);
      }
    } catch (error: unknown) {
      console.error("Signup error:", error);
      setError(
        error instanceof Error
          ? error.message
          : "Failed to create account. Please try again."
      );
      setIsLoading(false);
    }
  };

  // Handle onboarding questions
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
      // Submit onboarding data
      submitOnboardingData();
    }
  };

  const goToPreviousStep = () => {
    if (onboardingStep > 1) {
      setOnboardingStep(onboardingStep - 1);
    }
  };

  const submitOnboardingData = async () => {
    setIsLoading(true);
    try {
      // Get user ID from localStorage
      const userId = localStorage.getItem("userId");

      // Save the onboarding answers to the database
      await supabase.from("user_profiles").insert([
        {
          user_id: userId,
          user_type: onboardingAnswers.userType,
          ai_proficiency: onboardingAnswers.aiProficiency,
          usage_frequency: onboardingAnswers.usageFrequency,
          usage_context: onboardingAnswers.usageContext,
          primary_goal: onboardingAnswers.primaryGoal,
          industry_focus: onboardingAnswers.industryFocus,
          created_at: new Date().toISOString(),
        },
      ]);

      // Redirect to home page
      navigate("/home");
    } catch (error: unknown) {
      console.error("Onboarding error:", error);
      setError(
        error instanceof Error
          ? error.message
          : "Failed to save your preferences. You can update them later."
      );
      // Redirect anyway
      navigate("/home");
    }
  };

  // Render onboarding questions based on current step
  const renderOnboardingStep = () => {
    // Add animations to question transitions
    const fadeIn = "animate-fadeIn";

    switch (onboardingStep) {
      case 1:
        return (
          <div className={`space-y-6 ${fadeIn}`}>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-100 text-indigo-500 mb-4">
                <svg
                  className="w-8 h-8"
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
              <h3 className="text-2xl font-bold text-gray-800">Who are you?</h3>
              <p className="text-gray-600 mt-2">
                This helps us tailor your experience.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-3">
              {[
                "Student",
                "Professional",
                "Researcher",
                "Entrepreneur",
                "Hobbyist",
              ].map((type) => (
                <button
                  key={type}
                  onClick={() => handleOnboardingChange("userType", type)}
                  className={`group text-left p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                    onboardingAnswers.userType === type
                      ? "border-indigo-500 bg-indigo-50 shadow-md"
                      : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30"
                  }`}
                >
                  <div className="flex items-center">
                    <div
                      className={`w-6 h-6 rounded-full mr-3 flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                        onboardingAnswers.userType === type
                          ? "border-indigo-500 bg-indigo-500"
                          : "border-gray-300 group-hover:border-indigo-300"
                      }`}
                    >
                      {onboardingAnswers.userType === type && (
                        <svg
                          className="w-4 h-4 text-white"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <span className="text-lg font-medium">{type}</span>
                  </div>
                </button>
              ))}
            </div>
          </div>
        );

      case 2:
        return (
          <div className={`space-y-6 ${fadeIn}`}>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-100 text-indigo-500 mb-4">
                <svg
                  className="w-8 h-8"
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
              <h3 className="text-2xl font-bold text-gray-800">
                What's your AI proficiency level?
              </h3>
              <p className="text-gray-600 mt-2">
                This helps us adjust the complexity of our insights.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4">
              {[
                {
                  level: "Beginner",
                  description: "New to AI concepts and technologies",
                  icon: "🌱",
                },
                {
                  level: "Intermediate",
                  description: "Familiar with basic AI principles",
                  icon: "📊",
                },
                {
                  level: "Advanced",
                  description: "Good understanding of AI systems",
                  icon: "🔍",
                },
                {
                  level: "Expert",
                  description: "Deep technical knowledge of AI",
                  icon: "🧠",
                },
              ].map((item) => (
                <button
                  key={item.level}
                  onClick={() =>
                    handleOnboardingChange("aiProficiency", item.level)
                  }
                  className={`group text-left p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                    onboardingAnswers.aiProficiency === item.level
                      ? "border-indigo-500 bg-indigo-50 shadow-md"
                      : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30"
                  }`}
                >
                  <div className="flex items-center">
                    <div
                      className={`w-6 h-6 rounded-full mr-3 flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                        onboardingAnswers.aiProficiency === item.level
                          ? "border-indigo-500 bg-indigo-500"
                          : "border-gray-300 group-hover:border-indigo-300"
                      }`}
                    >
                      {onboardingAnswers.aiProficiency === item.level && (
                        <svg
                          className="w-4 h-4 text-white"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <div>
                      <div className="flex items-center">
                        <span className="text-lg font-medium mr-2">
                          {item.level}
                        </span>
                        <span className="text-xl">{item.icon}</span>
                      </div>
                      <span className="text-sm text-gray-500 block mt-1">
                        {item.description}
                      </span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        );

      case 3:
        return (
          <div className={`space-y-6 ${fadeIn}`}>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-100 text-indigo-500 mb-4">
                <svg
                  className="w-8 h-8"
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
              <h3 className="text-2xl font-bold text-gray-800">
                How often will you use PRISM?
              </h3>
              <p className="text-gray-600 mt-2">
                This helps us understand your engagement needs.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-3">
              {[
                { frequency: "Daily", icon: "📅" },
                { frequency: "Several times a week", icon: "🗓️" },
                { frequency: "Weekly", icon: "📆" },
                { frequency: "Monthly", icon: "📊" },
                { frequency: "Occasionally", icon: "⏰" },
              ].map(({ frequency, icon }) => (
                <button
                  key={frequency}
                  onClick={() =>
                    handleOnboardingChange("usageFrequency", frequency)
                  }
                  className={`group text-left p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                    onboardingAnswers.usageFrequency === frequency
                      ? "border-indigo-500 bg-indigo-50 shadow-md"
                      : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30"
                  }`}
                >
                  <div className="flex items-center">
                    <div
                      className={`w-6 h-6 rounded-full mr-3 flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                        onboardingAnswers.usageFrequency === frequency
                          ? "border-indigo-500 bg-indigo-500"
                          : "border-gray-300 group-hover:border-indigo-300"
                      }`}
                    >
                      {onboardingAnswers.usageFrequency === frequency && (
                        <svg
                          className="w-4 h-4 text-white"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex items-center">
                      <span className="text-lg font-medium mr-2">
                        {frequency}
                      </span>
                      <span className="text-xl">{icon}</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        );

      case 4:
        return (
          <div className={`space-y-6 ${fadeIn}`}>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-100 text-indigo-500 mb-4">
                <svg
                  className="w-8 h-8"
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
              <h3 className="text-2xl font-bold text-gray-800">
                How will you use PRISM?
              </h3>
              <p className="text-gray-600 mt-2">
                This helps us understand your context.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-3">
              {[
                {
                  context: "Work/Professional",
                  icon: "💼",
                  description: "For business use cases",
                },
                {
                  context: "Academic Research",
                  icon: "🎓",
                  description: "For scholarly activities",
                },
                {
                  context: "Personal Interest",
                  icon: "🌟",
                  description: "For personal growth",
                },
                {
                  context: "Policy Development",
                  icon: "📝",
                  description: "For creating guidelines",
                },
                {
                  context: "Compliance",
                  icon: "✅",
                  description: "For regulatory adherence",
                },
              ].map(({ context, icon, description }) => (
                <button
                  key={context}
                  onClick={() =>
                    handleOnboardingChange("usageContext", context)
                  }
                  className={`group text-left p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                    onboardingAnswers.usageContext === context
                      ? "border-indigo-500 bg-indigo-50 shadow-md"
                      : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30"
                  }`}
                >
                  <div className="flex items-center">
                    <div
                      className={`w-6 h-6 rounded-full mr-3 flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                        onboardingAnswers.usageContext === context
                          ? "border-indigo-500 bg-indigo-500"
                          : "border-gray-300 group-hover:border-indigo-300"
                      }`}
                    >
                      {onboardingAnswers.usageContext === context && (
                        <svg
                          className="w-4 h-4 text-white"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <div>
                      <div className="flex items-center">
                        <span className="text-lg font-medium mr-2">
                          {context}
                        </span>
                        <span className="text-xl">{icon}</span>
                      </div>
                      <span className="text-sm text-gray-500 block mt-1">
                        {description}
                      </span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        );

      case 5:
        return (
          <div className={`space-y-6 ${fadeIn}`}>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-100 text-indigo-500 mb-4">
                <svg
                  className="w-8 h-8"
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
              <h3 className="text-2xl font-bold text-gray-800">
                What's your primary goal with PRISM?
              </h3>
              <p className="text-gray-600 mt-2">
                This helps us highlight relevant features.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-3">
              {[
                { goal: "Stay informed on AI regulations", icon: "📣" },
                { goal: "Ensure compliance", icon: "🔒" },
                { goal: "Competitive analysis", icon: "📊" },
                { goal: "Research", icon: "🔍" },
                { goal: "Education", icon: "📚" },
                { goal: "Product development", icon: "💡" },
              ].map(({ goal, icon }) => (
                <button
                  key={goal}
                  onClick={() => handleOnboardingChange("primaryGoal", goal)}
                  className={`group text-left p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                    onboardingAnswers.primaryGoal === goal
                      ? "border-indigo-500 bg-indigo-50 shadow-md"
                      : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30"
                  }`}
                >
                  <div className="flex items-center">
                    <div
                      className={`w-6 h-6 rounded-full mr-3 flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                        onboardingAnswers.primaryGoal === goal
                          ? "border-indigo-500 bg-indigo-500"
                          : "border-gray-300 group-hover:border-indigo-300"
                      }`}
                    >
                      {onboardingAnswers.primaryGoal === goal && (
                        <svg
                          className="w-4 h-4 text-white"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex items-center">
                      <span className="text-lg font-medium mr-2">{goal}</span>
                      <span className="text-xl">{icon}</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        );

      case 6:
        return (
          <div className={`space-y-6 ${fadeIn}`}>
            <div className="text-center mb-6">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-indigo-100 text-indigo-500 mb-4">
                <svg
                  className="w-8 h-8"
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
              <h3 className="text-2xl font-bold text-gray-800">
                Which industry interests you most?
              </h3>
              <p className="text-gray-600 mt-2">
                This helps us curate relevant content.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {[
                { industry: "Financial Services", icon: "💰" },
                { industry: "Healthcare", icon: "🏥" },
                { industry: "Government & Public Policy", icon: "🏛️" },
                { industry: "Technology", icon: "💻" },
                { industry: "Education", icon: "🎓" },
                { industry: "Manufacturing", icon: "🏭" },
                { industry: "Retail", icon: "🛍️" },
                { industry: "Other", icon: "✨" },
              ].map(({ industry, icon }) => (
                <button
                  key={industry}
                  onClick={() =>
                    handleOnboardingChange("industryFocus", industry)
                  }
                  className={`group text-left p-4 rounded-xl border-2 transition-all duration-200 hover:shadow-md ${
                    onboardingAnswers.industryFocus === industry
                      ? "border-indigo-500 bg-indigo-50 shadow-md"
                      : "border-gray-200 hover:border-indigo-300 hover:bg-indigo-50/30"
                  }`}
                >
                  <div className="flex items-center">
                    <div
                      className={`w-6 h-6 rounded-full mr-3 flex-shrink-0 flex items-center justify-center border-2 transition-all duration-200 ${
                        onboardingAnswers.industryFocus === industry
                          ? "border-indigo-500 bg-indigo-500"
                          : "border-gray-300 group-hover:border-indigo-300"
                      }`}
                    >
                      {onboardingAnswers.industryFocus === industry && (
                        <svg
                          className="w-4 h-4 text-white"
                          fill="currentColor"
                          viewBox="0 0 20 20"
                        >
                          <path
                            fillRule="evenodd"
                            d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                            clipRule="evenodd"
                          />
                        </svg>
                      )}
                    </div>
                    <div className="flex items-center">
                      <span className="text-base font-medium mr-2">
                        {industry}
                      </span>
                      <span className="text-xl">{icon}</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  // Replace the handleGoogleSignup function with this version
  const handleGoogleSignup = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const response2 = await fetch("https://prism-backend-dtcc-dot-block-convey-p1.uc.r.appspot.com/auth/signin", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: "admin@gmail.com",
          password: "admin",
        }),
      });
      // Use the signInWithGoogle function from AuthContext
      await signInWithGoogle();

      const tokenData = await response2.json();
      // Store the access token in localStorage
      localStorage.setItem("access_token", tokenData.access_token);
      console.log("tokenData", tokenData.access_token);

      // The redirect will be handled by the signInWithGoogle function
    } catch (error: unknown) {
      console.error("Google signup error:", error);
      setError(
        error instanceof Error
          ? error.message
          : "Failed to sign up with Google. Please try again."
      );
      setIsLoading(false);
    }
  };

  // Render the onboarding form or the signup form
  if (showOnboarding) {
    return (
      <div className="min-h-screen flex flex-col justify-center items-center bg-gradient-to-br from-blue-50 to-indigo-50 px-4 py-12">
        <div className="max-w-md w-full bg-white p-8 rounded-2xl shadow-xl border border-gray-100">
          <div className="mb-10">
            <div className="flex items-center space-x-2 mb-4">
              <svg
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="text-indigo-600"
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
              <h1 className="text-2xl font-bold text-gray-900">
                PRISM by Block Convey
              </h1>
            </div>

            <div className="flex flex-col mb-8">
              <div className="flex justify-between items-center mb-2">
                <span className="text-sm font-medium text-gray-600">
                  Step {onboardingStep} of 6
                </span>
                <span className="text-sm font-medium text-indigo-600">
                  {Math.round((onboardingStep / 6) * 100)}% Complete
                </span>
              </div>
              <div className="w-full bg-gray-200 rounded-full h-2.5 overflow-hidden">
                <div
                  className="bg-indigo-600 h-2.5 rounded-full transition-all duration-500 ease-in-out"
                  style={{ width: `${(onboardingStep / 6) * 100}%` }}
                ></div>
              </div>
            </div>
          </div>

          {error && (
            <div className="mb-6 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-lg text-sm animate-pulse">
              <div className="flex">
                <svg
                  className="h-5 w-5 text-red-500 mr-2"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
                <p>{error}</p>
              </div>
            </div>
          )}

          <div className="bg-white transition-all duration-500 ease-in-out transform">
            {renderOnboardingStep()}
          </div>

          <div className="flex mt-10 space-x-3">
            {onboardingStep > 1 && (
              <button
                onClick={goToPreviousStep}
                className="w-1/2 border-2 border-gray-200 text-gray-700 rounded-xl py-3 px-6 hover:bg-gray-50 hover:border-gray-300 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 font-medium flex items-center justify-center"
              >
                <svg
                  className="w-5 h-5 mr-2"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
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
            )}
            <button
              onClick={goToNextStep}
              disabled={
                (onboardingStep === 1 && !onboardingAnswers.userType) ||
                (onboardingStep === 2 && !onboardingAnswers.aiProficiency) ||
                (onboardingStep === 3 && !onboardingAnswers.usageFrequency) ||
                (onboardingStep === 4 && !onboardingAnswers.usageContext) ||
                (onboardingStep === 5 && !onboardingAnswers.primaryGoal) ||
                (onboardingStep === 6 && !onboardingAnswers.industryFocus) ||
                isLoading
              }
              className={`${
                onboardingStep > 1 ? "w-1/2" : "w-full"
              } bg-indigo-600 text-white rounded-xl py-3 px-6 hover:bg-indigo-700 transition-all duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed font-medium flex items-center justify-center shadow-md`}
            >
              {onboardingStep === 6 ? (
                isLoading ? (
                  <>
                    <svg
                      className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
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
                    Completing Setup...
                  </>
                ) : (
                  <>
                    Complete Setup
                    <svg
                      className="w-5 h-5 ml-2"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                  </>
                )
              ) : (
                <>
                  Continue
                  <svg
                    className="w-5 h-5 ml-2"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth="2"
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </>
              )}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Your existing signup form render code
  return (
    <div className="min-h-screen flex justify-center items-center bg-gradient-to-br from-white to-gray-50">
      <div className="w-full max-w-md px-6 py-12">
        <div className="mb-10">
          <div className="flex items-center space-x-2 mb-6">
            <div>
              <img src="/logo.svg" alt="PRISM Logo" className="w-6 h-6" />
            </div>
            <h1 className="text-xl font-medium text-gray-900">
              PRISM by Block Convey
            </h1>
          </div>
          <h2 className="text-3xl font-bold text-gray-900 mb-2">
            Create your account
          </h2>
          <p className="text-gray-600">
            Track AI regulations and governance worldwide
          </p>
        </div>

        {/* Google Sign Up button */}
        <div className="mb-8">
          <button
            onClick={handleGoogleSignup}
            disabled={isLoading}
            className="w-full flex items-center justify-center border border-gray-300 rounded-xl py-3 px-4 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 shadow-sm transition-all duration-200"
          >
            <svg className="h-5 w-5 mr-2" viewBox="0 0 24 24">
              <g transform="matrix(1, 0, 0, 1, 27.009001, -39.238998)">
                <path
                  fill="#4285F4"
                  d="M -3.264 51.509 C -3.264 50.719 -3.334 49.969 -3.454 49.239 L -14.754 49.239 L -14.754 53.749 L -8.284 53.749 C -8.574 55.229 -9.424 56.479 -10.684 57.329 L -10.684 60.329 L -6.824 60.329 C -4.564 58.239 -3.264 55.159 -3.264 51.509 Z"
                />
                <path
                  fill="#34A853"
                  d="M -14.754 63.239 C -11.514 63.239 -8.804 62.159 -6.824 60.329 L -10.684 57.329 C -11.764 58.049 -13.134 58.489 -14.754 58.489 C -17.884 58.489 -20.534 56.379 -21.484 53.529 L -25.464 53.529 L -25.464 56.619 C -23.494 60.539 -19.444 63.239 -14.754 63.239 Z"
                />
                <path
                  fill="#FBBC05"
                  d="M -21.484 53.529 C -21.734 52.809 -21.864 52.039 -21.864 51.239 C -21.864 50.439 -21.724 49.669 -21.484 48.949 L -21.484 45.859 L -25.464 45.859 C -26.284 47.479 -26.754 49.299 -26.754 51.239 C -26.754 53.179 -26.284 54.999 -25.464 56.619 L -21.484 53.529 Z"
                />
                <path
                  fill="#EA4335"
                  d="M -14.754 43.989 C -12.984 43.989 -11.404 44.599 -10.154 45.789 L -6.734 42.369 C -8.804 40.429 -11.514 39.239 -14.754 39.239 C -19.444 39.239 -23.494 41.939 -25.464 45.859 L -21.484 48.949 C -20.534 46.099 -17.884 43.989 -14.754 43.989 Z"
                />
              </g>
            </svg>
            <span className="font-medium">
              {isLoading ? "Connecting to Google..." : "Sign up with Google"}
            </span>
          </button>
        </div>

        <div className="flex items-center mb-8">
          <div className="flex-grow border-t border-gray-200"></div>
          <span className="px-4 text-sm text-gray-500">
            Or continue with email
          </span>
          <div className="flex-grow border-t border-gray-200"></div>
        </div>

        {error && (
          <div className="mb-4 p-4 bg-red-50 border-l-4 border-red-500 text-red-700 rounded-md text-sm">
            <div className="flex">
              <svg
                className="h-5 w-5 text-red-500 mr-2"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
                />
              </svg>
              <span>{error}</span>
            </div>
          </div>
        )}

        <form onSubmit={handleSignup} className="space-y-5">
          <div className="space-y-5">
            <div>
              <label
                htmlFor="fullName"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Full Name
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-gray-400">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z"
                      clipRule="evenodd"
                    />
                  </svg>
                </span>
                <input
                  id="fullName"
                  name="fullName"
                  type="text"
                  required
                  className="block w-full pl-10 px-3 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-all duration-200"
                  placeholder="Enter your full name"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="organization"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Organization{" "}
                <span className="text-gray-400 text-xs">(Optional)</span>
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-gray-400">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M4 4a2 2 0 012-2h8a2 2 0 012 2v12a1 1 0 01-1 1h-2a1 1 0 01-1-1v-2a1 1 0 00-1-1H9a1 1 0 00-1 1v2a1 1 0 01-1 1H5a1 1 0 01-1-1V4zm3 1h2v2H7V5zm2 4H7v2h2V9zm2-4h2v2h-2V5zm2 4h-2v2h2V9z"
                      clipRule="evenodd"
                    />
                  </svg>
                </span>
                <input
                  id="organization"
                  name="organization"
                  type="text"
                  className="block w-full pl-10 px-3 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-all duration-200"
                  placeholder="Your company or organization"
                  value={organization}
                  onChange={(e) => setOrganization(e.target.value)}
                />
              </div>
            </div>

            <div>
              <label
                htmlFor="email"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Email
              </label>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-gray-400">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                    <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
                  </svg>
                </span>
                <input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="block w-full pl-10 px-3 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-all duration-200"
                  placeholder="Enter your email address"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <label
                  htmlFor="password"
                  className="block text-sm font-medium text-gray-700"
                >
                  Password
                </label>
              </div>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-gray-400">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </span>
                <input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete="new-password"
                  required
                  className="block w-full pl-10 px-3 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-all duration-200"
                  placeholder="Create a password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                />
              </div>
            </div>

            <div>
              <div className="flex justify-between mb-1">
                <label
                  htmlFor="confirmPassword"
                  className="block text-sm font-medium text-gray-700"
                >
                  Confirm Password
                </label>
              </div>
              <div className="relative">
                <span className="absolute inset-y-0 left-0 pl-3 flex items-center text-gray-400">
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    viewBox="0 0 20 20"
                    fill="currentColor"
                  >
                    <path
                      fillRule="evenodd"
                      d="M5 9V7a5 5 0 0110 0v2a2 2 0 012 2v5a2 2 0 01-2 2H5a2 2 0 01-2-2v-5a2 2 0 012-2zm8-2v2H7V7a3 3 0 016 0z"
                      clipRule="evenodd"
                    />
                  </svg>
                </span>
                <input
                  id="confirmPassword"
                  name="confirmPassword"
                  type="password"
                  autoComplete="new-password"
                  required
                  className="block w-full pl-10 px-3 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-all duration-200"
                  placeholder="Confirm your password"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                />
              </div>
            </div>

            <div className="pt-2">
              <button
                type="submit"
                disabled={isLoading}
                className="w-full bg-teal-600 text-white rounded-xl py-3 px-4 hover:bg-teal-500 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 shadow-md transition-all duration-200 font-medium"
              >
                {isLoading ? (
                  <div className="flex items-center justify-center">
                    <svg
                      className="animate-spin -ml-1 mr-2 h-5 w-5 text-white"
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
                    Creating account...
                  </div>
                ) : (
                  "Create account"
                )}
              </button>
            </div>
          </div>
        </form>

        <div className="mt-6 text-center">
          <p className="text-sm text-gray-600">
            Already have an account?{" "}
            <Link
              to="/login"
              className="font-medium text-teal-600 hover:text-teal-500 transition-colors duration-200"
            >
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
};

export default SignupPage;
