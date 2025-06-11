import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../contexts/AuthContext";

export default function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [currentCardIndex, setCurrentCardIndex] = useState(0);
  const [isCardAnimating, setIsCardAnimating] = useState(false);
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const {
    signInWithEmail,
    signInWithGoogle,
    user,
    loading: authLoading,
  } = useAuth();

  useEffect(() => {
    if (user && !authLoading) {
      navigate("/home");
    }
  }, [user, authLoading, navigate]);

  // Card data for the right side
  const cardData = [
    {
      category: "FINANCIAL REGULATIONS",
      tag: "Policy",
      title: "Apple's 'sexist' credit card investigated by US regulator",
      description:
        "Investigation highlights need for algorithmic fairness in financial services",
      color: "pink-400",
    },
    {
      category: "EU REGULATION",
      tag: "New Legislation",
      title: "EU AI Act set to establish global standards for AI governance",
      description:
        "Europe leads with comprehensive framework for artificial intelligence regulation",
      color: "blue-400",
    },
    {
      category: "TRANSPARENCY",
      tag: "Research",
      title: "Stanford study reveals gaps in AI model documentation practices",
      description:
        "Researchers call for standardized reporting to improve AI accountability",
      color: "emerald-400",
    },
    {
      category: "ETHICAL AI",
      tag: "Industry News",
      title: "Major tech companies form alliance for responsible AI deployment",
      description:
        "Coalition aims to develop shared ethical standards and best practices",
      color: "amber-400",
    },
  ];

  const nextCard = () => {
    if (!isCardAnimating) {
      setIsCardAnimating(true);
      setTimeout(() => {
        setCurrentCardIndex((prevIndex) => (prevIndex + 1) % cardData.length);
        setIsCardAnimating(false);
      }, 300);
    }
  };

  const prevCard = () => {
    if (!isCardAnimating) {
      setIsCardAnimating(true);
      setTimeout(() => {
        setCurrentCardIndex(
          (prevIndex) => (prevIndex - 1 + cardData.length) % cardData.length
        );
        setIsCardAnimating(false);
      }, 300);
    }
  };

  const handleLogin = async () => {
    if (!email || !password) {
      setError("Please enter both email and password");
      return;
    }

    try {
      setIsLoading(true);
      setError("");
      // First authenticate with your existing auth provider
      await signInWithEmail(email, password);

      // Make API calls to both token endpoints
      try {
        // First token endpoint
      
        // Second token endpoint (localhost)
        const response2 = await fetch("http://localhost:8000/auth/signin", {
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
        // Now navigate
        navigate("/home");
      } catch (tokenError) {
        console.error("Token fetch error:", tokenError);
        // You might want to handle this differently than a login error
        // Maybe still allow the user to proceed but with limited functionality
        navigate("/home");
      }
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error ? error.message : "Failed to sign in";
      setError(errorMessage);
      setIsLoading(false);
    }
  };

  const handleGoogleLogin = async () => {
    try {
      setIsLoading(true);
      setError("");
      const response2 = await fetch("http://localhost:8000/auth/signin", {
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
      await signInWithGoogle();
      

     

 
    } catch (error: unknown) {
      const errorMessage =
        error instanceof Error
          ? error.message
          : "Failed to sign in with Google";
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-row bg-gradient-to-br from-white to-gray-50">
      {/* Left column */}
      <div className="w-full lg:w-1/2 flex justify-center items-center px-6 py-12">
        <div className="max-w-md w-full">
          <div className="mb-10">
            <div className="flex items-center space-x-2 mb-6">
              <div className="bg-teal-500 p-2 rounded-lg shadow-md">
                <img src="/logo.svg" alt="PRISM Logo" className="w-6 h-6" />
              </div>
              <h1 className="text-xl font-medium text-gray-900">
                PRISM by Block Convey
              </h1>
            </div>
            <h2 className="text-3xl font-bold text-gray-900 mb-2">
              Log in to your account
            </h2>
            <p className="text-gray-600">
              Access AI regulations and governance insights
            </p>
          </div>

          {/* Google Sign In */}
          <div className="mb-8">
            <button
              onClick={handleGoogleLogin}
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
                {isLoading ? "Signing in..." : "Sign in with Google"}
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

          <div className="space-y-5">
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
                <a
                  href="/auth/reset-password"
                  className="text-sm text-teal-500 hover:text-teal-400 transition-colors duration-200"
                >
                  Forgot password?
                </a>
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
                  autoComplete="current-password"
                  required
                  className="block w-full pl-10 px-3 py-3 border border-gray-300 rounded-xl shadow-sm focus:outline-none focus:ring-2 focus:ring-teal-500 focus:border-teal-500 transition-all duration-200"
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      handleLogin();
                    }
                  }}
                />
              </div>
            </div>

            <div className="pt-2">
              <button
                onClick={handleLogin}
                disabled={isLoading}
                className="w-full bg-teal-500 text-white rounded-xl py-3 px-4 hover:bg-teal-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-400 shadow-md transition-all duration-200 font-medium"
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
                    Signing in...
                  </div>
                ) : (
                  "Sign in"
                )}
              </button>
            </div>
          </div>

          <div className="mt-6 text-center">
            <p className="text-sm text-gray-600">
              Don't have an account?{" "}
              <Link
                to="/signup"
                className="font-medium text-teal-500 hover:text-teal-400 transition-colors duration-200"
              >
                Sign up
              </Link>
            </p>
          </div>
        </div>
      </div>

      {/* Right column - Card design that matches the image */}
      <div className="lg:block lg:w-1/2 bg-white flex justify-center items-center p-8 pt-16">
        <div className="bg-teal-500 rounded-3xl shadow-2xl overflow-hidden w-full max-w-2xl border border-teal-400/30">
          <div className="p-8 md:p-12 h-full relative">
            {/* Decorative elements */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-400 opacity-20 rounded-full blur-xl -mr-10 -mt-10"></div>
            <div className="absolute bottom-0 left-0 w-32 h-32 bg-teal-300 opacity-20 rounded-full blur-xl -ml-10 -mb-10"></div>

            <div className="flex items-center space-x-3 mb-10">
              <div className="bg-teal-500/20 p-2.5 rounded-lg backdrop-blur-sm">
                <img src="/logo.svg" alt="PRISM Logo" className="w-7 h-7" />
              </div>
              <span className="text-2xl font-bold text-white tracking-wider">
                PRISM
              </span>
            </div>

            <div className="inline-block px-5 py-1.5 bg-teal-400/20 border border-teal-300/30 rounded-full text-sm font-medium text-teal-200 mb-6 transform hover:scale-105 transition-transform duration-300 cursor-pointer shadow-lg">
              AI REGULATION RADAR
            </div>

            <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
              Shaping the Future of{" "}
              <span className="text-teal-200">AI Governance</span>
            </h2>

            <p className="text-lg text-white/90 mb-10 max-w-xl">
              Stay ahead with critical developments in AI regulation, ethics,
              and policy frameworks worldwide.
            </p>

            <div
              className={`${
                isCardAnimating
                  ? "opacity-0 transform -translate-y-4"
                  : "opacity-100 transform translate-y-0"
              } transition-all duration-300 ease-in-out bg-white shadow-lg p-6 rounded-xl border ${
                cardData[currentCardIndex].color === "pink-400"
                  ? "border-pink-400"
                  : cardData[currentCardIndex].color === "blue-400"
                  ? "border-blue-400"
                  : cardData[currentCardIndex].color === "emerald-400"
                  ? "border-emerald-400"
                  : "border-amber-400"
              } hover:shadow-xl group cursor-pointer`}
              onClick={nextCard}
            >
              <div className="flex space-x-2 mb-3">
                <span
                  className={`px-3 py-1 ${
                    cardData[currentCardIndex].color === "pink-400"
                      ? "bg-pink-500 hover:bg-pink-600"
                      : cardData[currentCardIndex].color === "blue-400"
                      ? "bg-blue-500 hover:bg-blue-600"
                      : cardData[currentCardIndex].color === "emerald-400"
                      ? "bg-emerald-500 hover:bg-emerald-600"
                      : "bg-amber-500 hover:bg-amber-600"
                  } text-white rounded-full text-xs font-semibold transition-all duration-200`}
                >
                  {cardData[currentCardIndex].category}
                </span>
                <span className="px-3 py-1 bg-teal-500 text-white rounded-full text-xs font-medium hover:bg-teal-600 transition-all duration-200">
                  {cardData[currentCardIndex].tag}
                </span>
              </div>

              <h3 className="text-xl font-bold text-gray-800 mb-3 group-hover:text-teal-600 transition-colors duration-200">
                {cardData[currentCardIndex].title}
              </h3>

              <p className="text-sm text-gray-600 mb-5">
                {cardData[currentCardIndex].description}
              </p>

              <div className="flex justify-between items-center">
                <div className="flex space-x-2">
                  {cardData.map((_, index) => (
                    <div
                      key={index}
                      className={`h-3 w-3 rounded-full transition-all duration-300 cursor-pointer ${
                        index === currentCardIndex
                          ? "bg-teal-500 scale-110"
                          : "bg-gray-300 hover:bg-gray-400"
                      }`}
                      onClick={(e) => {
                        e.stopPropagation();
                        if (!isCardAnimating) {
                          setIsCardAnimating(true);
                          setTimeout(() => {
                            setCurrentCardIndex(index);
                            setIsCardAnimating(false);
                          }, 300);
                        }
                      }}
                    />
                  ))}
                </div>

                <div className="text-xs text-gray-500 font-medium">
                  {currentCardIndex + 1} of {cardData.length}
                </div>
              </div>
            </div>

            <div className="flex justify-center items-center mt-8 mb-4">
              <div className="flex space-x-5 bg-gray-100 backdrop-blur-sm px-5 py-2 rounded-full border border-gray-200">
                <button
                  onClick={prevCard}
                  className="h-10 w-10 rounded-full bg-teal-500 hover:bg-teal-600 flex items-center justify-center text-white transition-all duration-200 hover:scale-110"
                  aria-label="Previous card"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M15 19l-7-7 7-7"
                    />
                  </svg>
                </button>
                <button
                  onClick={nextCard}
                  className="h-10 w-10 rounded-full bg-teal-500 hover:bg-teal-600 flex items-center justify-center text-white transition-all duration-200 hover:scale-110"
                  aria-label="Next card"
                >
                  <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-5 w-5"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
