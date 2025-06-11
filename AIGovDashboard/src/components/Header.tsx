import React, { useState, useEffect } from "react";
import {
  User,
  LogOut,
  UserCircle,
  HelpCircle,
  ChevronDown,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Link, useNavigate } from "react-router-dom";
import { supabase } from "../lib/supabase";

interface HeaderProps {
  userName?: string;
  userEmail?: string;
  onLogout?: () => Promise<void>;
}

const Header: React.FC<HeaderProps> = ({
  userName = "John Doe",
  userEmail = "john.doe@example.com",
  onLogout,
}) => {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const navigate = useNavigate();
  const [userInfo, setUserInfo] = useState({
    name: userName,
    email: userEmail,
  });

  useEffect(() => {
    // Fetch user information when component mounts
    const getUserInfo = async () => {
      try {
        const {
          data: { user },
        } = await supabase.auth.getUser();

        if (user) {
          console.log("Found user:", user); // Debug logging

          try {
            // Get user profile data if you have a profiles table
            const { data: profileData, error: profileError } = await supabase
              .from("profiles")
              .select("*")
              .eq("id", user.id)
              .single();

            if (!profileError && profileData) {
              console.log("Found profile data:", profileData); // Debug logging

              setUserInfo({
                name:
                  profileData?.full_name ||
                  user.user_metadata?.full_name ||
                  userName,
                email: user.email || userEmail,
              });
            } else {
              // If no profile or error, just use the user data
              setUserInfo({
                name:
                  user.user_metadata?.full_name ||
                  user.email?.split("@")[0] ||
                  userName,
                email: user.email || userEmail,
              });
            }
          } catch (profileError) {
            console.error("Error fetching profile:", profileError);

            // Fallback to auth user data
            setUserInfo({
              name:
                user.user_metadata?.full_name ||
                user.email?.split("@")[0] ||
                userName,
              email: user.email || userEmail,
            });
          }
        }
      } catch (error) {
        console.error("Error fetching user info:", error);
      }
    };

    getUserInfo();
  }, [userName, userEmail]);

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  const handleLogout = async () => {
    try {
      setIsDropdownOpen(false); // Close dropdown first

      // Start loading state
      const loadingIndicator = document.createElement("div");
      loadingIndicator.className =
        "fixed inset-0 bg-white bg-opacity-50 flex items-center justify-center z-50";
      loadingIndicator.innerHTML =
        '<div class="animate-spin rounded-full h-8 w-8 border-t-2 border-b-2 border-gray-900"></div>';
      document.body.appendChild(loadingIndicator);

      try {
        if (onLogout) {
          await onLogout();
        } else {
          // Use Supabase for logout if no onLogout prop is provided
          const { error } = await supabase.auth.signOut({ scope: "global" });
          if (error) throw error;
        }

        // Clear all local storage items related to user session
        localStorage.clear(); // Clear all localStorage items

        // Clear session cookies
        document.cookie.split(";").forEach((cookie) => {
          const eqPos = cookie.indexOf("=");
          const name =
            eqPos > -1 ? cookie.substr(0, eqPos).trim() : cookie.trim();
          document.cookie =
            name + "=;expires=Thu, 01 Jan 1970 00:00:00 GMT;path=/";
        });

        // Add a small delay before redirecting
        await new Promise((resolve) => setTimeout(resolve, 500));

        // Use replace instead of push to prevent back navigation
        navigate("/login", { replace: true });
      } finally {
        // Remove loading indicator
        if (loadingIndicator && loadingIndicator.parentNode) {
          loadingIndicator.parentNode.removeChild(loadingIndicator);
        }
      }
    } catch (err) {
      console.error("Unexpected error during logout:", err);
    }
  };

  const handleClickOutside = () => {
    if (isDropdownOpen) {
      setIsDropdownOpen(false);
    }
  };

  const navigateToProfile = () => {
    setIsDropdownOpen(false);
    navigate("/profile");
  };

  return (
    <header className="bg-white border-b border-gray-200 h-16 z-50 sticky top-0">
      <div className="max-w-[1400px] mx-auto px-6 flex h-16 items-center justify-between">
        {/* Logo */}
        <div className="flex items-center">
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
          >
            <Link to="/home" className="flex items-center">
              <img
                src="/logo.svg"
                alt="PRISM Logo"
                className="h-10 w-auto mr-2"
              />
              <div className="flex flex-col">
                <h1 className="text-xl font-bold text-gray-900 leading-none">
                  PRISM
                </h1>
                <span className="text-xs text-gray-500 font-normal">
                  Block Convey<b> X </b>DTCC
                </span>
              </div>
            </Link>
          </motion.div>
        </div>

        {/* Profile Section */}
        <div className="relative">
          <motion.button
            whileHover={{ scale: 1.03 }}
            whileTap={{ scale: 0.97 }}
            onClick={toggleDropdown}
            className="flex items-center gap-2 px-3 py-1.5 rounded-full hover:bg-gray-100 hover:shadow-sm transition-all duration-300 focus:outline-none"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 overflow-hidden flex items-center justify-center shadow-md border border-white">
              <User className="w-4 h-4 text-white" />
            </div>
            <span className="text-gray-800 font-medium hidden sm:inline-block">
              {userInfo.name}
            </span>
            <ChevronDown className="w-4 h-4 text-gray-500" />
          </motion.button>

          <AnimatePresence>
            {isDropdownOpen && (
              <>
                <div
                  className="fixed inset-0 z-40"
                  onClick={handleClickOutside}
                />
                <motion.div
                  initial={{ opacity: 0, y: 10, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: 10, scale: 0.95 }}
                  transition={{ duration: 0.2 }}
                  className="absolute right-0 mt-2 w-64 bg-white rounded-xl shadow-xl border border-gray-100 z-50 overflow-hidden"
                >
                  <div className="p-4 border-b border-gray-100 bg-gray-50">
                    <p className="font-medium text-gray-900">{userInfo.name}</p>
                    <p className="text-sm text-gray-500">{userInfo.email}</p>
                  </div>
                  <div className="py-2">
                    <button
                      className="w-full px-4 py-2 text-left text-sm flex items-center space-x-2 hover:bg-gray-50 transition-colors"
                      onClick={navigateToProfile}
                    >
                      <UserCircle className="w-4 h-4 text-gray-500" />
                      <span>Profile</span>
                    </button>
                    <a
                      href="mailto:info@blockconvey.com"
                      className="w-full px-4 py-2 text-left text-sm flex items-center space-x-2 hover:bg-gray-50 transition-colors"
                      onClick={() => setIsDropdownOpen(false)}
                    >
                      <HelpCircle className="w-4 h-4 text-gray-500" />
                      <span>Help & Support</span>
                    </a>
                  </div>
                  <div className="py-2 border-t border-gray-100">
                    <button
                      className="w-full px-4 py-2 text-left text-sm flex items-center space-x-2 text-red-600 hover:bg-red-50 transition-colors"
                      onClick={handleLogout}
                    >
                      <LogOut className="w-4 h-4" />
                      <span>Logout</span>
                    </button>
                  </div>
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>
      </div>
    </header>
  );
};

export default Header;
