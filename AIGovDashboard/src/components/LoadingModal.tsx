import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";

interface LoadingModalProps {
  isOpen: boolean;
  onComplete: () => void;
}

const LoadingModal: React.FC<LoadingModalProps> = ({ isOpen, onComplete }) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isOpen) {
      // Reset progress when modal opens
      setProgress(0);

      // Simulate loading progress
      const interval = setInterval(() => {
        setProgress((prev) => {
          const next = prev + Math.random() * 15;
          if (next >= 100) {
            clearInterval(interval);
            setTimeout(onComplete, 500); // Give time for 100% to be visible
            return 100;
          }
          return next;
        });
      }, 200);

      return () => clearInterval(interval);
    }
  }, [isOpen, onComplete]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white rounded-2xl p-8 w-[90%] max-w-md">
        <div className="text-center mb-6">
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Generating Risk Assessment Report
          </h3>
          <p className="text-gray-600">
            Please wait while we analyze and compile your assessment data...
          </p>
        </div>

        {/* Progress Bar Container */}
        <div className="relative h-4 bg-gray-100 rounded-full overflow-hidden mb-3">
          {/* Animated Progress Bar */}
          <motion.div
            className="absolute left-0 top-0 h-full bg-gradient-to-r from-teal-500 to-blue-500"
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 0.3 }}
          />
        </div>

        {/* Progress Text */}
        <div className="flex justify-between text-sm">
          <span className="text-gray-600">Progress</span>
          <span className="font-medium text-gray-900">
            {Math.round(progress)}%
          </span>
        </div>

        {/* Loading Animation */}
        <div className="mt-6 flex justify-center">
          <motion.div
            className="w-12 h-12 border-4 border-teal-500 rounded-full border-t-transparent"
            animate={{ rotate: 360 }}
            transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
          />
        </div>
      </div>
    </div>
  );
};

export default LoadingModal;
