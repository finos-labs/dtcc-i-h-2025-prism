import { useParams } from "react-router-dom";
import { motion } from "framer-motion";
import { FileUp as FileUpload2, Download } from "lucide-react";
import { Button } from "../components/ui/button";
import { Breadcrumb } from "../components/ui/breadcrumb";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  Legend,
  BarChart,
  Bar,
  Cell,
  ComposedChart,
  ScatterChart,
  Scatter,
  ReferenceLine,
} from "recharts";
import AppLayout from "../components/AppLayout";
import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { InfoTooltip } from "../components/InfoTooltip";
import { supabase } from "../lib/supabase";

// Type for model info
interface ModelInfo {
  type: string;
  name: string;
  version: string;
}

// Type for data info
interface DataInfo {
  total_samples: number;
  feature_count: number;
  feature_names: string[];
  class_distribution: Record<string, number>;
}

// Type for metrics
interface PerformanceMetrics {
  metrics: {
    accuracy: number;
    f1Score: number;
    precision: number;
    recall: number;
    aucRoc: number;
    status: {
      accuracy: string;
      f1Score: string;
      precision: string;
      recall: string;
      aucRoc: string;
    };
  };
  confusionMatrix: Array<{
    name: string;
    value: number;
    fill: string;
  }>;
  precisionRecall: Array<{
    recall: number;
    precision: number;
  }>;
  rocCurve: Array<{
    fpr: number;
    tpr: number;
    random: number;
  }>;
  learningCurve: {
    trainSizes: number[];
    trainScores: number[];
    testScores: number[];
  };
  modelInfo: ModelInfo;
  dataInfo: DataInfo;
  cross_validation: {
    mean_score: number;
    std_score: number;
    scores: number[];
  };
  isRegression: boolean;
  regression_metrics: {
    mse: number;
    rmse: number;
    mae: number;
    r2: number;
  };
  residual_analysis: {
    mean_residual: number;
    std_residual: number;
    residuals: number[];
  };
}

// Demo data for the Investment Portfolio Analysis project
const performanceData = {
  metrics: {
    accuracy: 94.5,
    f1Score: 93.2,
    aucRoc: 95.8,
    status: {
      accuracy: "Above Target",
      f1Score: "Good",
      aucRoc: "Excellent",
    },
  },
  trends: Array.from({ length: 30 }, (_, i) => ({
    date: `Jan ${i + 1}`,
    accuracy: 90 + Math.random() * 10,
    precision: 88 + Math.random() * 12,
    recall: 89 + Math.random() * 11,
    f1_score: 91 + Math.random() * 9,
    auc_roc: 92 + Math.random() * 8,
  })),
  distribution: Array.from({ length: 30 }, (_, i) => ({
    date: `Jan ${i + 1}`,
    "True Positives": Math.round((0.4 + Math.random() * 0.1) * 100),
    "True Negatives": Math.round((0.3 + Math.random() * 0.1) * 100),
    "False Positives": Math.round((0.2 + Math.random() * 0.1) * 100),
    "False Negatives": Math.round((0.1 + Math.random() * 0.1) * 100),
  })),
  confusionMatrix: [
    { name: "True Negative", value: 423, fill: "#3182CE" },
    { name: "False Positive", value: 17, fill: "#F56565" },
    { name: "False Negative", value: 28, fill: "#ED8936" },
    { name: "True Positive", value: 532, fill: "#38A169" },
  ],
  precisionRecall: Array.from({ length: 11 }, (_, i) => ({
    recall: i * 0.1,
    precision: i === 10 ? 0.5 : 1 - i * 0.1 * 0.5,
  })),
  rocCurve: Array.from({ length: 11 }, (_, i) => ({
    fpr: i * 0.1,
    tpr: i === 0 ? 0 : Math.min(1, Math.pow(i * 0.1, 0.5) * 1.5),
    random: i * 0.1,
  })),
};

const MetricCard = ({
  title,
  value,
  status,
  description,
  infoData,
}: {
  title: string;
  value: number;
  status: string;
  description: string;
  infoData: any;
}) => (
  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 h-full">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-lg font-medium text-gray-900 truncate pr-2">
        {title}
      </h3>
      <InfoTooltip
        title={title}
        entityType="metric"
        entityName={title}
        data={{
          value: value,
          status: status,
          description: description,
        }}
      />
    </div>
    <div className="flex flex-wrap items-baseline gap-2">
      <span className="text-3xl font-bold text-gray-900 break-all">
        {value.toFixed(1)}%
      </span>
      <span
        className={`text-xs font-medium px-2 py-0.5 rounded-full whitespace-nowrap ${
          status === "Excellent"
            ? "bg-green-100 text-green-800"
            : status === "Good"
            ? "bg-blue-100 text-blue-800"
            : status === "Needs Improvement"
            ? "bg-red-100 text-red-800"
            : "bg-yellow-100 text-yellow-800"
        }`}
      >
        {status}
      </span>
    </div>
    <p
      className="mt-2 text-sm text-gray-500 overflow-hidden text-ellipsis"
      title={description}
    >
      {description}
    </p>
  </div>
);

const UploadModal = () => (
  <div className="bg-gradient-to-br from-white to-gray-50 rounded-xl p-12 shadow-xl border border-gray-100">
    <div className="text-center max-w-2xl mx-auto">
      <div className="bg-primary/5 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-6">
        <FileUpload2 className="h-8 w-8 text-primary" />
      </div>
      <h2 className="text-2xl font-bold text-gray-900 mb-3">
        Upload Your Model for Analysis
      </h2>
      <p className="text-gray-600 mb-8">
        Upload your trained model to start analyzing its performance metrics,
        fairness indicators, and explainability factors.
      </p>
      <Button className="bg-primary text-white hover:bg-primary/90 px-8 py-6 text-lg shadow-lg">
        Upload Model
      </Button>
      <p className="mt-4 text-sm text-gray-500">
        Supported formats: .h5, .pkl, .pt, .onnx
      </p>
    </div>
  </div>
);

// Update ConfusionMatrixChart for better responsiveness
const ConfusionMatrixChart = ({ data }: { data: any[] }) => (
  <div className="w-full h-full">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-sm font-medium text-gray-700">
        Classification Results
      </h3>
      <InfoTooltip
        title="About Confusion Matrix"
        entityType="chart"
        entityName="Confusion Matrix"
        data={{ chartData: data }}
      />
    </div>
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 20, right: 25, left: 25, bottom: 20 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            type="number"
            tickFormatter={(value) => value.toLocaleString()}
          />
          <YAxis
            dataKey="name"
            type="category"
            tick={{ fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={110}
          />
          <Tooltip
            formatter={(value) => [
              `${value.toLocaleString()} samples`,
              "Count",
            ]}
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              borderRadius: "8px",
              boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
              border: "1px solid #E5E7EB",
            }}
          />
          <Legend
            verticalAlign="top"
            wrapperStyle={{ paddingBottom: "10px" }}
          />
          <Bar
            dataKey="value"
            name="Number of Samples"
            animationDuration={1500}
            minPointSize={5}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.fill} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  </div>
);

// Similar updates for other chart components
const PrecisionRecallChart = ({ data }: { data: any[] }) => (
  <div className="w-full h-full">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-sm font-medium text-gray-700">Trade-off Analysis</h3>
      <InfoTooltip
        title="About Precision-Recall Curve"
        entityType="chart"
        entityName="Precision-Recall Curve"
        data={{ chartData: data }}
      />
    </div>
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 20, right: 20, left: 20, bottom: 30 }}
        >
          <defs>
            <linearGradient id="colorPrecision" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#10B981" stopOpacity={0.8} />
              <stop offset="95%" stopColor="#10B981" stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="recall"
            tickFormatter={(value) => value.toFixed(1)}
            label={{ value: "Recall", position: "insideBottom", offset: -10 }}
            tick={{ fontSize: 12 }}
          />
          <YAxis
            domain={[0.4, 1]}
            tickFormatter={(value) => value.toFixed(1)}
            label={{ value: "Precision", angle: -90, position: "insideLeft" }}
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            formatter={(value: any) => [value.toFixed(2), "Precision"]}
            labelFormatter={(value) => `Recall: ${value}`}
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              borderRadius: "8px",
              boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
              border: "1px solid #E5E7EB",
            }}
          />
          <Area
            type="monotone"
            dataKey="precision"
            stroke="#10B981"
            fillOpacity={1}
            fill="url(#colorPrecision)"
            strokeWidth={3}
            dot={false}
            activeDot={{ r: 6 }}
            animationDuration={1500}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  </div>
);

// Fix ROCCurveChart implementation to display AUC value properly
const ROCCurveChart = ({
  data,
  aucValue,
}: {
  data: any[];
  aucValue?: number;
}) => {
  // Ensure aucValue is properly formatted - transform to 0-1 scale if needed
  const normalizedAucValue =
    aucValue !== undefined && aucValue > 1 ? aucValue / 100 : aucValue;

  // Dynamically create the name for the legend to include the AUC value if available
  const curveLabel = `ROC Curve ${
    normalizedAucValue ? `(AUC=${normalizedAucValue.toFixed(3)})` : ""
  }`;

  return (
    <div className="w-full h-full">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-700">ROC Analysis</h3>
      </div>
      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart
            data={data}
            margin={{ top: 20, right: 30, left: 40, bottom: 40 }}
          >
            <defs>
              <linearGradient id="colorRoc" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#6366F1" stopOpacity={0.8} />
                <stop offset="95%" stopColor="#6366F1" stopOpacity={0.1} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis
              dataKey="fpr"
              type="number"
              tickFormatter={(value) => value.toFixed(1)}
              domain={[0, 1]}
              label={{
                value: "False Positive Rate",
                position: "insideBottomRight",
                offset: -5,
                dy: 10,
                fontSize: 11,
              }}
              tick={{ fontSize: 11 }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(value) => value.toFixed(1)}
              label={{
                value: "True Positive Rate",
                angle: -90,
                position: "insideLeft",
                offset: -20,
                fontSize: 11,
              }}
              tick={{ fontSize: 11 }}
            />
            <Tooltip
              formatter={(value: any) => [value.toFixed(2), "Value"]}
              contentStyle={{
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                borderRadius: "8px",
                boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
                border: "1px solid #E5E7EB",
              }}
            />
            <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
            <Area
              type="monotone"
              name={curveLabel}
              dataKey="tpr"
              stroke="#6366F1"
              fillOpacity={1}
              fill="url(#colorRoc)"
              strokeWidth={3}
              dot={false}
              activeDot={{ r: 6 }}
              animationDuration={1500}
            />
            <Line
              type="monotone"
              dataKey="random"
              name="Random Classifier"
              stroke="#D1D5DB"
              strokeWidth={2}
              strokeDasharray="5 5"
              dot={false}
              activeDot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const LearningCurveChart = ({ data }: { data: any }) => (
  <div className="w-full h-full">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-sm font-medium text-gray-700">Learning Curve</h3>
    </div>
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart
          data={data.trainSizes.map((size: number, index: number) => ({
            size,
            trainScore: data.trainScores[index] * 100 || 0,
            testScore: data.testScores[index] * 100 || 0,
          }))}
          margin={{ top: 20, right: 30, left: 40, bottom: 40 }}
        >
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="size"
            label={{
              value: "Training Examples",
              position: "insideBottomRight",
              offset: -5,
              dy: 10,
              fontSize: 11,
            }}
            tick={{ fontSize: 11 }}
            tickFormatter={(value) => value.toLocaleString()}
          />
          <YAxis
            domain={[0, 100]}
            label={{
              value: "Score (%)",
              angle: -90,
              position: "insideLeft",
              offset: -20,
              fontSize: 11,
            }}
            tick={{ fontSize: 11 }}
          />
          <Tooltip
            formatter={(value: any) => [`${value.toFixed(1)}%`, "Score"]}
            labelFormatter={(value) =>
              `Training Examples: ${value.toLocaleString()}`
            }
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              borderRadius: "8px",
              boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
              border: "1px solid #E5E7EB",
            }}
          />
          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
          <Line
            type="monotone"
            dataKey="trainScore"
            name="Training Score"
            stroke="#8884d8"
            strokeWidth={2}
            animationDuration={1500}
            dot={{ r: 2 }}
            activeDot={{ r: 5 }}
          />
          <Line
            type="monotone"
            dataKey="testScore"
            name="Validation Score"
            stroke="#82ca9d"
            strokeWidth={2}
            animationDuration={1500}
            dot={{ r: 2 }}
            activeDot={{ r: 5 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  </div>
);

const ClassDistributionChart = ({ data }: { data: any }) => {
  const classData = Object.entries(data).map(
    ([className, count]: [string, any]) => ({
      name: `Class ${className}`,
      value: count,
      fill: getClassColor(parseInt(className)),
    })
  );

  return (
    <div className="w-full h-full">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium text-gray-700">
          Class Distribution
        </h3>
      </div>
      <div className="h-[280px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={classData}
            margin={{ top: 20, right: 25, left: 25, bottom: 30 }}
          >
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="name" tick={{ fontSize: 11 }} />
            <YAxis
              tick={{ fontSize: 11 }}
              tickFormatter={(value) => value.toLocaleString()}
            />
            <Tooltip
              formatter={(value: any) => [
                `${value.toLocaleString()} samples`,
                "Count",
              ]}
              contentStyle={{
                backgroundColor: "rgba(255, 255, 255, 0.95)",
                borderRadius: "8px",
                boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
                border: "1px solid #E5E7EB",
              }}
            />
            <Bar dataKey="value" name="Samples" animationDuration={1500}>
              {classData.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

// Helper function to get colors for different classes
const getClassColor = (classIndex: number) => {
  const colors = [
    "#3182CE",
    "#10B981",
    "#F59E0B",
    "#6366F1",
    "#EC4899",
    "#8B5CF6",
  ];
  return colors[classIndex % colors.length];
};

// Helper function to format confusion matrix data
const formatConfusionMatrix = (confusionMatrix: any) => {
  if (!confusionMatrix) {
    return [
      { name: "True Negative", value: 0, fill: "#3182CE" },
      { name: "False Positive", value: 0, fill: "#F56565" },
      { name: "False Negative", value: 0, fill: "#ED8936" },
      { name: "True Positive", value: 0, fill: "#38A169" },
    ];
  }

  // For binary classification, use the provided values
  if (confusionMatrix.true_negatives !== undefined) {
    return [
      {
        name: "True Negative",
        value: confusionMatrix.true_negatives || 0,
        fill: "#3182CE",
      },
      {
        name: "False Positive",
        value: confusionMatrix.false_positives || 0,
        fill: "#F56565",
      },
      {
        name: "False Negative",
        value: confusionMatrix.false_negatives || 0,
        fill: "#ED8936",
      },
      {
        name: "True Positive",
        value: confusionMatrix.true_positives || 0,
        fill: "#38A169",
      },
    ];
  }

  // If matrix array is provided (for multi-class), extract the values from the matrix
  if (confusionMatrix.matrix && Array.isArray(confusionMatrix.matrix)) {
    const matrix = confusionMatrix.matrix;

    // This is a simplification for visualization - for multi-class, we're showing aggregated values
    // For a true multi-class confusion matrix, a heatmap would be better
    const classCount = matrix.length;

    let tp = 0,
      fp = 0,
      fn = 0,
      tn = 0;

    // Calculate TP, FP, FN, TN from the matrix
    for (let i = 0; i < classCount; i++) {
      for (let j = 0; j < classCount; j++) {
        if (i === j) {
          // True positives (diagonal elements)
          tp += matrix[i][j];
        } else {
          // Off-diagonal elements
          fp += matrix[i][j]; // From perspective of class i, these are false positives
          fn += matrix[j][i]; // From perspective of class i, these are false negatives
        }
      }
    }

    // For multi-class, TN is less meaningful, but we calculate it to complete the matrix
    // TN for class i would be all correctly classified instances of other classes
    const total = matrix
      .flat()
      .reduce((sum: number, val: number) => sum + val, 0);
    tn = total - (tp + fp + fn);

    return [
      { name: "True Negative", value: tn, fill: "#3182CE" },
      { name: "False Positive", value: fp / classCount, fill: "#F56565" },
      { name: "False Negative", value: fn / classCount, fill: "#ED8936" },
      { name: "True Positive", value: tp, fill: "#38A169" },
    ];
  }

  return [
    { name: "True Negative", value: 0, fill: "#3182CE" },
    { name: "False Positive", value: 0, fill: "#F56565" },
    { name: "False Negative", value: 0, fill: "#ED8936" },
    { name: "True Positive", value: 0, fill: "#38A169" },
  ];
};

// Helper function to format ROC curve data
const formatRocCurve = (rocCurve: any) => {
  if (!rocCurve || !Object.keys(rocCurve).length) {
    return Array.from({ length: 11 }, (_, i) => ({
      fpr: i * 0.1,
      tpr: i * 0.1,
      random: i * 0.1,
    }));
  }

  // Check if rocCurve has direct fpr, tpr arrays (top-level)
  if (Array.isArray(rocCurve.fpr) && Array.isArray(rocCurve.tpr)) {
    return rocCurve.fpr.map((fpr: number, index: number) => ({
      fpr: fpr,
      tpr: rocCurve.tpr[index] || 0,
      random: fpr, // Random baseline (diagonal line)
    }));
  }

  // Get the first class key (usually 'class_0', 'class_1', etc.)
  const classKeys = Object.keys(rocCurve);

  // If there are no class keys, use default values
  if (classKeys.length === 0) {
    return Array.from({ length: 11 }, (_, i) => ({
      fpr: i * 0.1,
      tpr: i * 0.1,
      random: i * 0.1,
    }));
  }

  // Prefer class_1 if available (typically the positive class)
  const selectedClass = classKeys.includes("class_1")
    ? "class_1"
    : classKeys[0];

  const fprArray = rocCurve[selectedClass]?.fpr || [];
  const tprArray = rocCurve[selectedClass]?.tpr || [];

  if (!fprArray.length || !tprArray.length) {
    return Array.from({ length: 11 }, (_, i) => ({
      fpr: i * 0.1,
      tpr: i * 0.1,
      random: i * 0.1,
    }));
  }

  // Combine FPR and TPR into data points for the chart
  return fprArray.map((fpr: number, index: number) => ({
    fpr: fpr,
    tpr: tprArray[index] || 0,
    random: fpr, // Random baseline (diagonal line)
  }));
};

// Format learning curve data
const formatLearningCurve = (learningCurve: any) => {
  if (
    !learningCurve ||
    !learningCurve.train_sizes ||
    !learningCurve.train_scores ||
    !learningCurve.test_scores
  ) {
    return {
      trainSizes: [],
      trainScores: [],
      testScores: [],
    };
  }

  // Average the scores across all cross-validation folds for each training size
  const trainScores = learningCurve.train_scores.map((scores: number[]) =>
    Array.isArray(scores)
      ? scores.reduce((sum: number, score: number) => sum + score, 0) /
        scores.length
      : scores
  );

  const testScores = learningCurve.test_scores.map((scores: number[]) =>
    Array.isArray(scores)
      ? scores.reduce((sum: number, score: number) => sum + score, 0) /
        scores.length
      : scores
  );

  return {
    trainSizes: learningCurve.train_sizes,
    trainScores: trainScores,
    testScores: testScores,
  };
};

// Helper function to calculate average AUC across all classes
const calculateAverageAUC = (rocCurve: any) => {
  if (!rocCurve) return 0;

  let total = 0;
  let count = 0;

  // First try to extract direct auc values if available
  for (const className in rocCurve) {
    if (rocCurve[className]?.auc !== undefined) {
      total += rocCurve[className].auc;
      count++;
    }
  }

  // If no AUC values found directly, check if there's a summary AUC
  if (count === 0 && rocCurve.auc !== undefined) {
    return rocCurve.auc;
  }

  return count > 0 ? total / count : 0;
};

// Define a function to validate and format the performance data
const formatPerformanceData = (data: any): PerformanceMetrics => {
  // Create a default structure that matches what the UI expects
  const defaultData: PerformanceMetrics = {
    metrics: {
      accuracy: 0,
      f1Score: 0,
      precision: 0,
      recall: 0,
      aucRoc: 0,
      status: {
        accuracy: "Not Available",
        f1Score: "Not Available",
        precision: "Not Available",
        recall: "Not Available",
        aucRoc: "Not Available",
      },
    },
    confusionMatrix: [
      { name: "True Negative", value: 0, fill: "#3182CE" },
      { name: "False Positive", value: 0, fill: "#F56565" },
      { name: "False Negative", value: 0, fill: "#ED8936" },
      { name: "True Positive", value: 0, fill: "#38A169" },
    ],
    precisionRecall: Array.from({ length: 11 }, (_, i) => ({
      recall: i * 0.1,
      precision: 0.5,
    })),
    rocCurve: Array.from({ length: 11 }, (_, i) => ({
      fpr: i * 0.1,
      tpr: i * 0.1,
      random: i * 0.1,
    })),
    learningCurve: {
      trainSizes: [],
      trainScores: [],
      testScores: [],
    },
    modelInfo: {
      type: "unknown",
      name: "unknown",
      version: "0.0.0",
    },
    dataInfo: {
      total_samples: 0,
      feature_count: 0,
      feature_names: [],
      class_distribution: {},
    },
    cross_validation: {
      mean_score: 0,
      std_score: 0,
      scores: [],
    },
    isRegression: false,
    regression_metrics: {
      mse: 0,
      rmse: 0,
      mae: 0,
      r2: 0,
    },
    residual_analysis: {
      mean_residual: 0,
      std_residual: 0,
      residuals: [],
    },
  };

  // If data is null or undefined, return default data
  if (!data) {
    return defaultData;
  }

  try {
    // Extract the metrics object from the API response
    const metricsData = data.metrics || {};

    // Check if this is a regression model
    const isRegression =
      data.model_info?.type === "regression" ||
      metricsData.model_info?.type === "regression";

    if (isRegression) {
      // Extract regression metrics
      const regression_metrics = {
        mse: metricsData.basic_metrics?.mse || 0,
        rmse: metricsData.basic_metrics?.rmse || 0,
        mae: metricsData.basic_metrics?.mae || 0,
        r2: metricsData.basic_metrics?.r2 || 0,
      };

      // Extract residual analysis data
      const residual_analysis = metricsData.residual_analysis || {
        mean_residual: 0,
        std_residual: 0,
        residuals: [],
      };

      return {
        // Include other common properties
        metrics: defaultData.metrics,
        confusionMatrix: defaultData.confusionMatrix,
        precisionRecall: defaultData.precisionRecall,
        rocCurve: defaultData.rocCurve,
        modelInfo: {
          type: metricsData.model_info?.type || "regression",
          name: data.model_name || metricsData.model_info?.name || "unknown",
          version:
            data.model_version || metricsData.model_info?.version || "0.0.0",
        },
        dataInfo: metricsData.data_info || defaultData.dataInfo,
        learningCurve: formatLearningCurve(metricsData.learning_curve),
        cross_validation: {
          mean_score: metricsData.cross_validation?.mean_score || 0,
          std_score: metricsData.cross_validation?.std_score || 0,
          scores: metricsData.cross_validation?.scores || [],
        },
        // Add regression-specific properties
        regression_metrics,
        residual_analysis,
        // Flag to indicate this is regression data
        isRegression: true,
      };
    } else {
      // Extract AUC-ROC value from different possible locations in the response
      let aucRocValue = 0;

      // Try direct auc field first
      if (metricsData.basic_metrics?.auc !== undefined) {
        aucRocValue = metricsData.basic_metrics.auc;
      }
      // Then try roc_auc field
      else if (metricsData.basic_metrics?.roc_auc !== undefined) {
        aucRocValue = metricsData.basic_metrics.roc_auc;
      }
      // Then try calculating from roc_curve data
      else {
        aucRocValue = calculateAverageAUC(metricsData.roc_curve);
      }

      // Make sure we convert to percentage (0-100 scale)
      aucRocValue = aucRocValue * 100;

      // Log to debug
      console.log("AUC-ROC Value:", aucRocValue);

      // Handle classification metrics
      const metrics = {
        accuracy: metricsData.basic_metrics?.accuracy * 100 || 0,
        f1Score: metricsData.basic_metrics?.f1 * 100 || 0,
        precision: metricsData.basic_metrics?.precision * 100 || 0,
        recall: metricsData.basic_metrics?.recall * 100 || 0,
        // Use the extracted AUC-ROC value
        aucRoc: aucRocValue,
        status: {
          accuracy: getMetricStatus(
            metricsData.basic_metrics?.accuracy * 100 || 0
          ),
          f1Score: getMetricStatus(metricsData.basic_metrics?.f1 * 100 || 0),
          precision: getMetricStatus(
            metricsData.basic_metrics?.precision * 100 || 0
          ),
          recall: getMetricStatus(metricsData.basic_metrics?.recall * 100 || 0),
          aucRoc: getMetricStatus(aucRocValue),
        },
      };

      return {
        metrics,
        confusionMatrix: formatConfusionMatrix(metricsData.confusion_matrix),
        precisionRecall: defaultData.precisionRecall,
        rocCurve: formatRocCurve(metricsData.roc_curve),
        learningCurve: formatLearningCurve(metricsData.learning_curve),
        modelInfo: {
          type: metricsData.model_info?.type || "unknown",
          name: data.model_name || metricsData.model_info?.name || "unknown",
          version:
            data.model_version || metricsData.model_info?.version || "0.0.0",
        },
        dataInfo: metricsData.data_info || defaultData.dataInfo,
        cross_validation: {
          mean_score: metricsData.cross_validation?.mean_score || 0,
          std_score: metricsData.cross_validation?.std_score || 0,
          scores: metricsData.cross_validation?.scores || [],
        },
        isRegression: false,
        regression_metrics: defaultData.regression_metrics,
        residual_analysis: defaultData.residual_analysis,
      };
    }
  } catch (error: any) {
    console.error("Error formatting performance data:", error);
    return defaultData;
  }
};

// Helper function to determine status based on metric value
const getMetricStatus = (value: number) => {
  if (value >= 95) return "Excellent";
  if (value >= 90) return "Good";
  if (value >= 80) return "Above Target";
  if (value >= 70) return "On Target";
  return "Needs Improvement";
};

// New component: Model Info Card
const ModelInfoCard = ({ data }: { data: any }) => (
  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 h-full">
    <h3 className="text-lg font-medium text-gray-900 mb-3">
      Model Information
    </h3>
    <div className="space-y-3">
      <div className="flex justify-between">
        <span className="text-sm text-gray-500 min-w-[90px]">Model Type:</span>
        <span
          className="text-sm font-medium capitalize ml-2 text-right truncate max-w-[65%]"
          title={data.type}
        >
          {data.type}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-sm text-gray-500 min-w-[90px]">Model Name:</span>
        <span
          className="text-sm font-medium ml-2 text-right truncate max-w-[65%]"
          title={data.name}
        >
          {data.name}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-sm text-gray-500 min-w-[90px]">Version:</span>
        <span
          className="text-sm font-medium ml-2 text-right truncate max-w-[65%]"
          title={data.version}
        >
          {data.version}
        </span>
      </div>
    </div>
  </div>
);

// New component: Data Info Card
const DataInfoCard = ({
  data,
  isRegression,
}: {
  data: any;
  isRegression: boolean;
}) => (
  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 h-full">
    <h3 className="text-lg font-medium text-gray-900 mb-3">
      Dataset Information
    </h3>
    <div className="space-y-3">
      <div className="flex justify-between">
        <span className="text-sm text-gray-500 min-w-[100px]">
          Total Samples:
        </span>
        <span
          className="text-sm font-medium ml-2 text-right truncate max-w-[60%]"
          title={data.total_samples.toString()}
        >
          {data.total_samples}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-sm text-gray-500 min-w-[100px]">Features:</span>
        <span
          className="text-sm font-medium ml-2 text-right truncate max-w-[60%]"
          title={data.feature_count.toString()}
        >
          {data.feature_count}
        </span>
      </div>
      <div className="flex justify-between">
        <span className="text-sm text-gray-500 min-w-[100px]">Classes:</span>
        <span
          className="text-sm font-medium ml-2 text-right truncate max-w-[60%]"
          title={Object.keys(data.class_distribution || {}).length.toString()}
        >
          {Object.keys(data.class_distribution || {}).length}
        </span>
      </div>
    </div>
  </div>
);

// Add this utility function
const formatMetricValue = (
  value: number,
  decimalPlaces: number = 1
): string => {
  return value.toFixed(decimalPlaces);
};

const RegressionMetricCard = ({
  title,
  value,
  status,
  description,
  unit = "",
  isPercentage = false,
  infoData,
}: {
  title: string;
  value: number;
  status: string;
  description: string;
  unit?: string;
  isPercentage?: boolean;
  infoData: any;
}) => (
  <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100 h-full">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-lg font-medium text-gray-900 truncate pr-2">
        {title}
      </h3>
      <InfoTooltip
        title={title}
        entityType="metric"
        entityName={title}
        data={{
          value: value,
          status: status,
          description: description,
          isPercentage: isPercentage,
        }}
      />
    </div>
    <div className="flex flex-wrap items-baseline gap-2">
      <span className="text-3xl font-bold text-gray-900 break-all">
        {isPercentage
          ? (value * 100).toFixed(2) + "%"
          : value.toFixed(
              Math.abs(value) < 0.01 ? 4 : Math.abs(value) < 1 ? 3 : 2
            ) + (unit ? " " + unit : "")}
      </span>
      <span
        className={`text-xs font-medium px-2 py-0.5 rounded-full whitespace-nowrap ${
          status === "Excellent"
            ? "bg-green-100 text-green-800"
            : status === "Good"
            ? "bg-blue-100 text-blue-800"
            : status === "Needs Improvement"
            ? "bg-red-100 text-red-800"
            : "bg-yellow-100 text-yellow-800"
        }`}
      >
        {status}
      </span>
    </div>
    <p
      className="mt-2 text-sm text-gray-500 overflow-hidden text-ellipsis"
      title={description}
    >
      {description}
    </p>
  </div>
);

const PerformancePage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [models, setModels] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [performanceMetrics, setPerformanceMetrics] =
    useState<PerformanceMetrics | null>(null);

  // Check if this is a dummy project
  const isDummyProject = id === "dummy-1" || id === "dummy-2";

  // New state to track if analysis exists
  const [hasAnalysis, setHasAnalysis] = useState(isDummyProject);

  useEffect(() => {
    const fetchModels = async () => {
      try {
        setLoading(true);

        // If it's a dummy project, set the performanceMetrics to the dummy data
        if (isDummyProject) {
          // Format the dummy data to match the expected structure
          const dummyFormattedData: PerformanceMetrics = {
            metrics: {
              accuracy: performanceData.metrics.accuracy,
              f1Score: performanceData.metrics.f1Score,
              precision: 92.7,
              recall: 91.5,
              aucRoc: performanceData.metrics.aucRoc,
              status: {
                accuracy: performanceData.metrics.status.accuracy,
                f1Score: performanceData.metrics.status.f1Score,
                precision: "Excellent",
                recall: "Excellent",
                aucRoc: performanceData.metrics.status.aucRoc,
              },
            },
            confusionMatrix: performanceData.confusionMatrix,
            precisionRecall: performanceData.precisionRecall,
            rocCurve: performanceData.rocCurve,
            learningCurve: {
              trainSizes: [100, 500, 1000, 1500, 2000],
              trainScores: [0.85, 0.87, 0.9, 0.92, 0.94],
              testScores: [0.82, 0.85, 0.87, 0.89, 0.91],
            },
            modelInfo: {
              type: "Random Forest",
              name: "Investment Portfolio Predictor",
              version: "1.2.3",
            },
            dataInfo: {
              total_samples: 2000,
              feature_count: 35,
              feature_names: [
                "return_ratio",
                "volatility",
                "market_cap",
                "sector_performance",
              ],
              class_distribution: {
                "0": 800,
                "1": 1200,
              },
            },
            cross_validation: {
              mean_score: 0.915,
              std_score: 0.023,
              scores: [0.91, 0.92, 0.93, 0.9, 0.91],
            },
            isRegression: false,
            regression_metrics: {
              mse: 0,
              rmse: 0,
              mae: 0,
              r2: 0,
            },
            residual_analysis: {
              mean_residual: 0,
              std_residual: 0,
              residuals: [],
            },
          };

          setPerformanceMetrics(dummyFormattedData);
          setLoading(false);
          return;
        }

        // Get access token from localStorage
        const accessToken = localStorage.getItem("access_token");

        if (!accessToken) {
          console.error("No access token found in localStorage");
          setLoading(false);
          return;
        }

        // Use the id from useParams instead of undefined projectId
        // Query Supabase for models
        const { data, error } = await supabase
          .from("modeldetails")
          .select("model_id, project_id, dataset_id, model_version")
          .eq("project_id", id);

        if (error) {
          throw error;
        }
        console.log("data", data);

        setModels(data || []);
        console.log("projectid", id);

        // If we have models for this project, fetch performance data
        if (data && data.length > 0) {
          const modelId = data[0].model_id;
          const projectId = data[0].project_id;
          const model_version = data[0].model_version;
          // Use the first model

          try {
            // Call the performance API with project_id and model_id and include the auth token
            const response = await fetch(
              `http://localhost:8000/ml/performance/${projectId}/${modelId}/${model_version}`,
              {
                method: "GET",
                headers: {
                  Authorization: `Bearer ${accessToken}`,
                  "Content-Type": "application/json",
                },
              }
            );

            if (response.ok) {
              const apiData = await response.json();
              console.log(apiData);
              // Process and validate the API response data
              const formattedData = formatPerformanceData(apiData);
              setPerformanceMetrics(formattedData);
              setHasAnalysis(true);
            } else {
              console.error(
                "Performance API returned an error:",
                response.statusText
              );
              // Still show the analysis UI if we have models but no performance data
              setHasAnalysis(data.length > 0);
            }
          } catch (apiError) {
            console.error("Error fetching performance data:", apiError);
            // If API call fails, still show analysis if we have models
            setHasAnalysis(data.length > 0);
          }
        } else {
          setHasAnalysis(false);
        }
      } catch (error) {
        console.error("Error fetching models:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
  }, [id, isDummyProject]);

  const breadcrumbSegments = [
    { title: "Projects", href: "/home" },
    { title: "Investment Portfolio Analysis", href: `/projects/${id}` },
    { title: "Performance", href: `/projects/${id}/performance` },
  ];

  // If still loading, show a loading state
  if (loading && !isDummyProject) {
    return (
      <div className="flex-1 p-8 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading performance data...</p>
        </div>
      </div>
    );
  }

  // If no analysis exists, show premium empty state
  if (!hasAnalysis) {
    return (
      <div className="flex-1 p-8">
        <div className="max-w-7xl mx-auto">
          {/* Page Header */}
          <div className="mb-10">
            <h1 className="text-3xl font-bold text-gray-900">
              Performance Analysis
            </h1>
            <p className="mt-2 text-gray-600">
              Evaluate your model's accuracy and performance metrics
            </p>
          </div>

          {/* Premium Empty State */}
          <div className="bg-gradient-to-br from-white to-gray-50 border border-gray-100 rounded-2xl shadow-sm p-12 text-center">
            <div className="mx-auto w-24 h-24 bg-teal-50 rounded-full flex items-center justify-center mb-6">
              <svg
                className="h-12 w-12 text-teal-600"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={1.5}
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
            </div>

            <h2 className="text-2xl font-bold text-gray-900 mb-3">
              No Performance Analysis Available
            </h2>
            <p className="text-gray-600 max-w-2xl mx-auto mb-8">
              Upload your model to analyze its performance metrics including
              accuracy, precision, recall, and F1 score. Our system will
              generate comprehensive visualizations to help you understand your
              model's strengths and weaknesses.
            </p>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <button
                onClick={() => navigate(`/projects/${id}`)}
                className="inline-flex items-center px-6 py-3 border border-gray-300 rounded-lg shadow-sm text-base font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-teal-500 transition-colors duration-200"
              >
                Return to Overview
              </button>
            </div>
          </div>

          {/* Features Preview Section */}
          <div className="mt-12">
            <h3 className="text-xl font-semibold text-gray-900 mb-6">
              Performance Analysis Features
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                <div className="w-12 h-12 bg-blue-50 rounded-lg flex items-center justify-center mb-4">
                  <svg
                    className="h-6 w-6 text-blue-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                    />
                  </svg>
                </div>
                <h4 className="text-lg font-medium text-gray-900 mb-2">
                  Comprehensive Metrics
                </h4>
                <p className="text-gray-600">
                  Track accuracy, precision, recall, F1 score, and AUC-ROC
                  across different thresholds and data segments.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                <div className="w-12 h-12 bg-purple-50 rounded-lg flex items-center justify-center mb-4">
                  <svg
                    className="h-6 w-6 text-purple-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"
                    />
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"
                    />
                  </svg>
                </div>
                <h4 className="text-lg font-medium text-gray-900 mb-2">
                  Interactive Visualizations
                </h4>
                <p className="text-gray-600">
                  Explore dynamic charts and plots to understand performance
                  patterns and identify areas for improvement.
                </p>
              </div>

              <div className="bg-white p-6 rounded-xl border border-gray-200 shadow-sm">
                <div className="w-12 h-12 bg-amber-50 rounded-lg flex items-center justify-center mb-4">
                  <svg
                    className="h-6 w-6 text-amber-600"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={1.5}
                      d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"
                    />
                  </svg>
                </div>
                <h4 className="text-lg font-medium text-gray-900 mb-2">
                  Actionable Insights
                </h4>
                <p className="text-gray-600">
                  Receive recommendations based on model performance to help
                  optimize and improve your AI system.
                </p>
              </div>
            </div>
          </div>

          {/* Additional information section */}
          <div className="mt-10 bg-white rounded-xl shadow-sm border border-gray-200 p-6">
            <h3 className="text-xl font-bold text-gray-900 mb-4">
              Why Performance Analysis Matters
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
              <div>
                <p className="text-gray-700 mb-4">
                  A comprehensive performance analysis helps you understand how
                  well your model works across different scenarios and data
                  distributions. By evaluating key metrics, you can:
                </p>
                <ul className="space-y-2">
                  <li className="flex items-start">
                    <svg
                      className="h-5 w-5 text-teal-500 mr-2 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-gray-600">
                      Identify prediction strengths and weaknesses
                    </span>
                  </li>
                  <li className="flex items-start">
                    <svg
                      className="h-5 w-5 text-teal-500 mr-2 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-gray-600">
                      Compare model versions to track improvements
                    </span>
                  </li>
                  <li className="flex items-start">
                    <svg
                      className="h-5 w-5 text-teal-500 mr-2 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-gray-600">
                      Optimize decision thresholds for your use case
                    </span>
                  </li>
                  <li className="flex items-start">
                    <svg
                      className="h-5 w-5 text-teal-500 mr-2 mt-0.5"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M5 13l4 4L19 7"
                      />
                    </svg>
                    <span className="text-gray-600">
                      Ensure reliability before deployment
                    </span>
                  </li>
                </ul>
              </div>
              <div className="bg-gray-50 rounded-lg p-5 border border-gray-100">
                <h4 className="text-base font-medium text-gray-900 mb-3">
                  Performance Report Preview
                </h4>
                <div className="flex flex-col space-y-3">
                  <div className="bg-white p-3 rounded border border-gray-200 flex justify-between items-center">
                    <span className="text-sm text-gray-700">
                      Accuracy Score
                    </span>
                    <div className="h-4 w-32 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-300 rounded-full"
                        style={{ width: "0%" }}
                      ></div>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border border-gray-200 flex justify-between items-center">
                    <span className="text-sm text-gray-700">
                      Precision Score
                    </span>
                    <div className="h-4 w-32 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-300 rounded-full"
                        style={{ width: "0%" }}
                      ></div>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border border-gray-200 flex justify-between items-center">
                    <span className="text-sm text-gray-700">Recall Score</span>
                    <div className="h-4 w-32 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-300 rounded-full"
                        style={{ width: "0%" }}
                      ></div>
                    </div>
                  </div>
                  <div className="bg-white p-3 rounded border border-gray-200 flex justify-between items-center">
                    <span className="text-sm text-gray-700">F1 Score</span>
                    <div className="h-4 w-32 bg-gray-200 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gray-300 rounded-full"
                        style={{ width: "0%" }}
                      ></div>
                    </div>
                  </div>
                </div>
                <div className="mt-4 pt-4 border-t border-gray-200 text-center text-sm text-gray-500">
                  Upload your model to generate a full performance report
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Use properly formatted performance data
  const dataToDisplay =
    performanceMetrics || formatPerformanceData(performanceData);

  // Return just the content without wrapping in AppLayout
  // This assumes that this page is rendered within a layout at the router level
  const content = isDummyProject ? (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="p-8 space-y-8"
    >
      <div>
        <h1 className="text-3xl font-bold text-gray-900">
          Performance Analysis
        </h1>
        <p className="text-gray-500 mt-1">
          Monitor and analyze model performance metrics
        </p>
      </div>

      {/* Model and Data Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <ModelInfoCard data={dataToDisplay.modelInfo} />
        <DataInfoCard
          data={dataToDisplay.dataInfo}
          isRegression={dataToDisplay.isRegression}
        />
      </div>

      {/* Key Metrics Section - Conditional rendering based on model type */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
        {dataToDisplay.isRegression ? (
          // Regression Metrics
          <>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <RegressionMetricCard
                title="MSE"
                value={dataToDisplay.regression_metrics.mse}
                status={getErrorMetricStatus(
                  dataToDisplay.regression_metrics.mse,
                  "MSE"
                )}
                description="Mean Squared Error - Average of squared differences between predicted and actual values"
                infoData={dataToDisplay.regression_metrics}
              />
            </motion.div>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <RegressionMetricCard
                title="RMSE"
                value={dataToDisplay.regression_metrics.rmse}
                status={getErrorMetricStatus(
                  dataToDisplay.regression_metrics.rmse,
                  "RMSE"
                )}
                description="Root Mean Squared Error - Square root of MSE"
                infoData={dataToDisplay.regression_metrics}
              />
            </motion.div>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <RegressionMetricCard
                title="MAE"
                value={dataToDisplay.regression_metrics.mae}
                status={getErrorMetricStatus(
                  dataToDisplay.regression_metrics.mae,
                  "MAE"
                )}
                description="Mean Absolute Error - Average of absolute differences between predicted and actual values"
                infoData={dataToDisplay.regression_metrics}
              />
            </motion.div>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <RegressionMetricCard
                title="R²"
                value={dataToDisplay.regression_metrics.r2}
                status={getR2Status(dataToDisplay.regression_metrics.r2)}
                description="Coefficient of determination - Proportion of variance explained by the model"
                isPercentage={true}
                infoData={dataToDisplay.regression_metrics}
              />
            </motion.div>
          </>
        ) : (
          // Classification Metrics (existing code)
          <>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <MetricCard
                title="Accuracy"
                value={dataToDisplay.metrics?.accuracy || 0}
                status={
                  dataToDisplay.metrics?.status?.accuracy || "Not Available"
                }
                description="Overall prediction accuracy"
                infoData={dataToDisplay.metrics}
              />
            </motion.div>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <MetricCard
                title="Precision"
                value={dataToDisplay.metrics?.precision || 0}
                status={
                  dataToDisplay.metrics?.status?.precision || "Not Available"
                }
                description="Positive predictive value"
                infoData={dataToDisplay.metrics}
              />
            </motion.div>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <MetricCard
                title="Recall"
                value={dataToDisplay.metrics?.recall || 0}
                status={
                  dataToDisplay.metrics?.status?.recall || "Not Available"
                }
                description="True positive rate"
                infoData={dataToDisplay.metrics}
              />
            </motion.div>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <MetricCard
                title="F1 Score"
                value={dataToDisplay.metrics?.f1Score || 0}
                status={
                  dataToDisplay.metrics?.status?.f1Score || "Not Available"
                }
                description="Harmonic mean of precision and recall"
                infoData={dataToDisplay.metrics}
              />
            </motion.div>
            <motion.div
              whileHover={{ y: -5 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <MetricCard
                title="AUC-ROC"
                value={dataToDisplay.metrics?.aucRoc || 0}
                status={
                  dataToDisplay.metrics?.status?.aucRoc || "Not Available"
                }
                description="Area under ROC curve"
                infoData={dataToDisplay.metrics}
              />
            </motion.div>
          </>
        )}
      </div>

      {/* Replace Confusion Matrix and Class Distribution with Residual Plot for regression */}
      {dataToDisplay.isRegression ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Residual Plot */}
          <motion.div
            whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
          >
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Residual Analysis
                </h2>
                <p className="text-sm text-gray-500">
                  Difference between predicted and actual values
                </p>
              </div>
            </div>
            <ResidualPlot
              residuals={dataToDisplay.residual_analysis.residuals}
            />
          </motion.div>

          {/* Residual Statistics */}
          <motion.div
            whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
          >
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Residual Statistics
                </h2>
                <p className="text-sm text-gray-500">
                  Statistical properties of model residuals
                </p>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-4">
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-700 mb-2">
                  Mean Residual
                </h3>
                <p className="text-2xl font-bold text-gray-900">
                  {dataToDisplay.residual_analysis.mean_residual.toFixed(4)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Ideally close to zero for unbiased models
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-700 mb-2">
                  Residual Standard Deviation
                </h3>
                <p className="text-2xl font-bold text-gray-900">
                  {dataToDisplay.residual_analysis.std_residual.toFixed(4)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Measure of residual dispersion
                </p>
              </div>
              <div className="bg-gray-50 p-4 rounded-lg">
                <h3 className="text-sm font-medium text-gray-700 mb-2">
                  Number of Samples
                </h3>
                <p className="text-2xl font-bold text-gray-900">
                  {dataToDisplay.residual_analysis.residuals.length}
                </p>
              </div>
            </div>
          </motion.div>
        </div>
      ) : (
        // Original Classification charts (Confusion Matrix, etc.)
        // ... existing code ...
        <>
          {/* Confusion Matrix */}
          <motion.div
            whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
          >
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Confusion Matrix
                </h2>
                <p className="text-sm text-gray-500">
                  Model prediction correctness
                </p>
              </div>
            </div>
            <ConfusionMatrixChart data={dataToDisplay.confusionMatrix || []} />
          </motion.div>

          {/* Class Distribution */}
          <motion.div
            whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
          >
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  Class Distribution
                </h2>
                <p className="text-sm text-gray-500">
                  Distribution of target classes
                </p>
              </div>
              <InfoTooltip
                title="Class Distribution"
                entityType="chart"
                entityName="Class Distribution"
                data={{
                  chartData: Object.entries(
                    dataToDisplay.dataInfo?.class_distribution || {}
                  ).map(([key, value]) => ({ name: key, value })),
                }}
              />
            </div>
            <ClassDistributionChart
              data={dataToDisplay.dataInfo?.class_distribution || {}}
            />
          </motion.div>
        </>
      )}

      {/* Performance Curves */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* ROC Curve - Only show for classification models */}
        {!dataToDisplay.isRegression && (
          <motion.div
            whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
            transition={{ duration: 0.3 }}
            className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
          >
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900">
                  ROC Curve
                </h2>
                <p className="text-sm text-gray-500">
                  True positive rate vs false positive rate
                </p>
              </div>
              <InfoTooltip
                title="ROC Curve"
                entityType="chart"
                entityName="ROC Curve"
                data={{
                  chartData: dataToDisplay.rocCurve || [],
                }}
              />
            </div>
            <ROCCurveChart
              data={dataToDisplay.rocCurve || []}
              aucValue={dataToDisplay.metrics?.aucRoc}
            />
          </motion.div>
        )}

        {/* Learning Curve - Always show */}
        <motion.div
          whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
          transition={{ duration: 0.3 }}
          className={`bg-white rounded-xl p-6 shadow-md border border-gray-100 ${
            dataToDisplay.isRegression ? "md:col-span-2" : ""
          }`}
        >
          <div className="flex justify-between items-center mb-6">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">
                Learning Curves
              </h2>
              <p className="text-sm text-gray-500">
                Training vs validation performance
              </p>
            </div>
            <InfoTooltip
              title="Learning Curves"
              entityType="chart"
              entityName="Learning Curves"
              data={{
                chartData: (dataToDisplay.learningCurve?.trainSizes || []).map(
                  (size, index) => ({
                    size,
                    trainScore:
                      (dataToDisplay.learningCurve?.trainScores || [])[index] ||
                      0,
                    testScore:
                      (dataToDisplay.learningCurve?.testScores || [])[index] ||
                      0,
                  })
                ),
              }}
            />
          </div>
          <LearningCurveChart
            data={
              dataToDisplay.learningCurve || {
                trainSizes: [],
                trainScores: [],
                testScores: [],
              }
            }
          />
        </motion.div>
      </div>

      {/* Cross-Validation Results */}
      <motion.div
        whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
        transition={{ duration: 0.3 }}
        className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
      >
        <div className="flex justify-between items-center mb-6">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">
              Cross-Validation Results
            </h2>
            <p className="text-sm text-gray-500">
              Model Stability Across Folds
            </p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Mean Score
            </h3>
            <p className="text-2xl font-bold text-gray-900 break-words">
              {formatMetricValue(
                (dataToDisplay.cross_validation?.mean_score || 0) * 100
              )}
              %
            </p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-gray-700 mb-2">
              Standard Deviation
            </h3>
            <p className="text-2xl font-bold text-gray-900 break-words">
              {formatMetricValue(
                (dataToDisplay.cross_validation?.std_score || 0) * 100
              )}
              %
            </p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Folds</h3>
            <p className="text-2xl font-bold text-gray-900 break-words">
              {dataToDisplay.cross_validation?.scores?.length || 0}
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  ) : (
    <div className="p-8">
      <div className="mt-8">
        {models.length > 0 ? (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5 }}
            className="space-y-8"
          >
            <div>
              <h1 className="text-3xl font-bold text-gray-900">
                Performance Analysis
              </h1>
              <p className="text-gray-500 mt-1">
                Monitor and analyze model performance metrics
              </p>
            </div>

            {/* Model and Data Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <ModelInfoCard data={dataToDisplay.modelInfo} />
              <DataInfoCard
                data={dataToDisplay.dataInfo}
                isRegression={dataToDisplay.isRegression}
              />
            </div>

            {/* Key Metrics Section - Conditional rendering based on model type */}
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
              {dataToDisplay.isRegression ? (
                // Regression Metrics
                <>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <RegressionMetricCard
                      title="MSE"
                      value={dataToDisplay.regression_metrics.mse}
                      status={getErrorMetricStatus(
                        dataToDisplay.regression_metrics.mse,
                        "MSE"
                      )}
                      description="Mean Squared Error - Average of squared differences between predicted and actual values"
                      infoData={dataToDisplay.regression_metrics}
                    />
                  </motion.div>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <RegressionMetricCard
                      title="RMSE"
                      value={dataToDisplay.regression_metrics.rmse}
                      status={getErrorMetricStatus(
                        dataToDisplay.regression_metrics.rmse,
                        "RMSE"
                      )}
                      description="Root Mean Squared Error - Square root of MSE"
                      infoData={dataToDisplay.regression_metrics}
                    />
                  </motion.div>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <RegressionMetricCard
                      title="MAE"
                      value={dataToDisplay.regression_metrics.mae}
                      status={getErrorMetricStatus(
                        dataToDisplay.regression_metrics.mae,
                        "MAE"
                      )}
                      description="Mean Absolute Error - Average of absolute differences between predicted and actual values"
                      infoData={dataToDisplay.regression_metrics}
                    />
                  </motion.div>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <RegressionMetricCard
                      title="R²"
                      value={dataToDisplay.regression_metrics.r2}
                      status={getR2Status(dataToDisplay.regression_metrics.r2)}
                      description="Coefficient of determination - Proportion of variance explained by the model"
                      isPercentage={true}
                      infoData={dataToDisplay.regression_metrics}
                    />
                  </motion.div>
                </>
              ) : (
                // Classification Metrics (existing code)
                <>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <MetricCard
                      title="Accuracy"
                      value={dataToDisplay.metrics?.accuracy || 0}
                      status={
                        dataToDisplay.metrics?.status?.accuracy ||
                        "Not Available"
                      }
                      description="Overall prediction accuracy"
                      infoData={dataToDisplay.metrics}
                    />
                  </motion.div>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <MetricCard
                      title="Precision"
                      value={dataToDisplay.metrics?.precision || 0}
                      status={
                        dataToDisplay.metrics?.status?.precision ||
                        "Not Available"
                      }
                      description="Positive predictive value"
                      infoData={dataToDisplay.metrics}
                    />
                  </motion.div>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <MetricCard
                      title="Recall"
                      value={dataToDisplay.metrics?.recall || 0}
                      status={
                        dataToDisplay.metrics?.status?.recall || "Not Available"
                      }
                      description="True positive rate"
                      infoData={dataToDisplay.metrics}
                    />
                  </motion.div>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <MetricCard
                      title="F1 Score"
                      value={dataToDisplay.metrics?.f1Score || 0}
                      status={
                        dataToDisplay.metrics?.status?.f1Score ||
                        "Not Available"
                      }
                      description="Harmonic mean of precision and recall"
                      infoData={dataToDisplay.metrics}
                    />
                  </motion.div>
                  <motion.div
                    whileHover={{ y: -5 }}
                    transition={{ type: "spring", stiffness: 300 }}
                  >
                    <MetricCard
                      title="AUC-ROC"
                      value={dataToDisplay.metrics?.aucRoc || 0}
                      status={
                        dataToDisplay.metrics?.status?.aucRoc || "Not Available"
                      }
                      description="Area under ROC curve"
                      infoData={dataToDisplay.metrics}
                    />
                  </motion.div>
                </>
              )}
            </div>

            {/* Replace Confusion Matrix and Class Distribution with Residual Plot for regression */}
            {dataToDisplay.isRegression ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Residual Plot */}
                <motion.div
                  whileHover={{
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)",
                  }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
                >
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">
                        Residual Analysis
                      </h2>
                      <p className="text-sm text-gray-500">
                        Difference between predicted and actual values
                      </p>
                    </div>
                  </div>
                  <ResidualPlot
                    residuals={dataToDisplay.residual_analysis.residuals}
                  />
                </motion.div>

                {/* Residual Statistics */}
                <motion.div
                  whileHover={{
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)",
                  }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
                >
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">
                        Residual Statistics
                      </h2>
                      <p className="text-sm text-gray-500">
                        Statistical properties of model residuals
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-4">
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-gray-700 mb-2">
                        Mean Residual
                      </h3>
                      <p className="text-2xl font-bold text-gray-900">
                        {dataToDisplay.residual_analysis.mean_residual.toFixed(
                          4
                        )}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Ideally close to zero for unbiased models
                      </p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-gray-700 mb-2">
                        Residual Standard Deviation
                      </h3>
                      <p className="text-2xl font-bold text-gray-900">
                        {dataToDisplay.residual_analysis.std_residual.toFixed(
                          4
                        )}
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Measure of residual dispersion
                      </p>
                    </div>
                    <div className="bg-gray-50 p-4 rounded-lg">
                      <h3 className="text-sm font-medium text-gray-700 mb-2">
                        Number of Samples
                      </h3>
                      <p className="text-2xl font-bold text-gray-900">
                        {dataToDisplay.residual_analysis.residuals.length}
                      </p>
                    </div>
                  </div>
                </motion.div>
              </div>
            ) : (
              // Original Classification charts (Confusion Matrix, etc.)
              // ... existing code ...
              <>
                {/* Confusion Matrix */}
                <motion.div
                  whileHover={{
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)",
                  }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
                >
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">
                        Confusion Matrix
                      </h2>
                      <p className="text-sm text-gray-500">
                        Model prediction correctness
                      </p>
                    </div>
                  </div>
                  <ConfusionMatrixChart
                    data={dataToDisplay.confusionMatrix || []}
                  />
                </motion.div>

                {/* Class Distribution */}
                <motion.div
                  whileHover={{
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)",
                  }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
                >
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">
                        Class Distribution
                      </h2>
                      <p className="text-sm text-gray-500">
                        Distribution of target classes
                      </p>
                    </div>
                    <InfoTooltip
                      title="Class Distribution"
                      entityType="chart"
                      entityName="Class Distribution"
                      data={{
                        chartData: Object.entries(
                          dataToDisplay.dataInfo?.class_distribution || {}
                        ).map(([key, value]) => ({ name: key, value })),
                      }}
                    />
                  </div>
                  <ClassDistributionChart
                    data={dataToDisplay.dataInfo?.class_distribution || {}}
                  />
                </motion.div>
              </>
            )}

            {/* Performance Curves */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* ROC Curve - Only show for classification models */}
              {!dataToDisplay.isRegression && (
                <motion.div
                  whileHover={{
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)",
                  }}
                  transition={{ duration: 0.3 }}
                  className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
                >
                  <div className="flex justify-between items-center mb-6">
                    <div>
                      <h2 className="text-xl font-semibold text-gray-900">
                        ROC Curve
                      </h2>
                      <p className="text-sm text-gray-500">
                        True positive rate vs false positive rate
                      </p>
                    </div>
                    <InfoTooltip
                      title="ROC Curve"
                      entityType="chart"
                      entityName="ROC Curve"
                      data={{
                        chartData: dataToDisplay.rocCurve || [],
                      }}
                    />
                  </div>
                  <ROCCurveChart
                    data={dataToDisplay.rocCurve}
                    aucValue={dataToDisplay.metrics?.aucRoc}
                  />
                </motion.div>
              )}

              {/* Learning Curve - Always show */}
              <motion.div
                whileHover={{
                  boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)",
                }}
                transition={{ duration: 0.3 }}
                className={`bg-white rounded-xl p-6 shadow-md border border-gray-100 ${
                  dataToDisplay.isRegression ? "md:col-span-2" : ""
                }`}
              >
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h2 className="text-xl font-semibold text-gray-900">
                      Learning Curves
                    </h2>
                    <p className="text-sm text-gray-500">
                      Training vs validation performance
                    </p>
                  </div>
                  <InfoTooltip
                    title="Learning Curves"
                    entityType="chart"
                    entityName="Learning Curves"
                    data={{
                      chartData: (
                        dataToDisplay.learningCurve?.trainSizes || []
                      ).map((size, index) => ({
                        size,
                        trainScore:
                          (dataToDisplay.learningCurve?.trainScores || [])[
                            index
                          ] || 0,
                        testScore:
                          (dataToDisplay.learningCurve?.testScores || [])[
                            index
                          ] || 0,
                      })),
                    }}
                  />
                </div>
                <LearningCurveChart
                  data={
                    dataToDisplay.learningCurve || {
                      trainSizes: [],
                      trainScores: [],
                      testScores: [],
                    }
                  }
                />
              </motion.div>
            </div>

            {/* Cross-Validation Results */}
            <motion.div
              whileHover={{ boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1)" }}
              transition={{ duration: 0.3 }}
              className="bg-white rounded-xl p-6 shadow-md border border-gray-100"
            >
              <div className="flex justify-between items-center mb-6">
                <div>
                  <h2 className="text-xl font-semibold text-gray-900">
                    Cross-Validation Results
                  </h2>
                  <p className="text-sm text-gray-500">
                    Model Stability Across Folds
                  </p>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">
                    Mean Score
                  </h3>
                  <p className="text-2xl font-bold text-gray-900 break-words">
                    {formatMetricValue(
                      (dataToDisplay.cross_validation?.mean_score || 0) * 100
                    )}
                    %
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">
                    Standard Deviation
                  </h3>
                  <p className="text-2xl font-bold text-gray-900 break-words">
                    {formatMetricValue(
                      (dataToDisplay.cross_validation?.std_score || 0) * 100
                    )}
                    %
                  </p>
                </div>
                <div className="bg-gray-50 p-4 rounded-lg">
                  <h3 className="text-sm font-medium text-gray-700 mb-2">
                    Folds
                  </h3>
                  <p className="text-2xl font-bold text-gray-900 break-words">
                    {dataToDisplay.cross_validation?.scores?.length || 0}
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        ) : (
          <UploadModal />
        )}
      </div>
    </div>
  );

  return (
    <AppLayout showSidebar={false} showHeader={false}>
      {content}
    </AppLayout>
  );
};

export default PerformancePage;

// Get status for R² (higher is better)
const getR2Status = (value: number) => {
  if (value >= 0.95) return "Excellent";
  if (value >= 0.9) return "Good";
  if (value >= 0.8) return "Above Target";
  if (value >= 0.7) return "On Target";
  return "Needs Improvement";
};

// Get status for error metrics (lower is better)
const getErrorMetricStatus = (value: number, metricType: string) => {
  // These thresholds would ideally be dynamic based on the dataset
  // Using placeholder values that should be adjusted for specific use cases

  // For MSE
  if (metricType === "MSE") {
    if (value < 50) return "Excellent";
    if (value < 100) return "Good";
    if (value < 200) return "Above Target";
    if (value < 400) return "On Target";
    return "Needs Improvement";
  }

  // For RMSE
  if (metricType === "RMSE") {
    if (value < 5) return "Excellent";
    if (value < 10) return "Good";
    if (value < 15) return "Above Target";
    if (value < 20) return "On Target";
    return "Needs Improvement";
  }

  // For MAE
  if (metricType === "MAE") {
    if (value < 4) return "Excellent";
    if (value < 8) return "Good";
    if (value < 12) return "Above Target";
    if (value < 16) return "On Target";
    return "Needs Improvement";
  }

  return "Not Available";
};

const ResidualPlot = ({ residuals }: { residuals: number[] }) => (
  <div className="w-full h-full">
    <div className="flex items-center justify-between mb-2">
      <h3 className="text-sm font-medium text-gray-700">Residual Analysis</h3>
      <InfoTooltip
        title="About Residual Analysis"
        entityType="chart"
        entityName="Residual Analysis"
        data={{
          chartData: residuals.map((residual, index) => ({ index, residual })),
        }}
      />
    </div>
    <div className="h-[280px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 20, right: 20, bottom: 30, left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            type="number"
            dataKey="index"
            name="Sample Index"
            label={{
              value: "Sample Index",
              position: "insideBottomRight",
              offset: -10,
            }}
          />
          <YAxis
            type="number"
            dataKey="residual"
            name="Residual"
            label={{ value: "Residual", angle: -90, position: "insideLeft" }}
          />
          <Tooltip
            formatter={(value: any) => [value.toFixed(2), "Residual"]}
            labelFormatter={(value) => `Sample: ${value}`}
            contentStyle={{
              backgroundColor: "rgba(255, 255, 255, 0.95)",
              borderRadius: "8px",
              boxShadow: "0px 4px 12px rgba(0, 0, 0, 0.1)",
              border: "1px solid #E5E7EB",
            }}
          />
          <ReferenceLine y={0} stroke="#666" strokeDasharray="3 3" />
          <Scatter
            name="Residuals"
            data={residuals.map((residual, index) => ({ index, residual }))}
            fill="#8884d8"
            opacity={0.6}
          />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  </div>
);
