import { InfoTooltip } from './InfoTooltip';

export const MetricCard = ({ title, value, status, description, isPercentage = true }: { 
  title: string;
  value: number;
  status: string;
  description: string;
  isPercentage?: boolean;
}) => {
  // Ensure all values are primitive types before passing to InfoTooltip
  const numericValue = Number(value);
  const stringStatus = String(status);
  const stringDescription = String(description);
  const booleanIsPercentage = Boolean(isPercentage);
  
  // Create metric data object with primitive values
  const metricData = {
    value: isNaN(numericValue) ? 0 : numericValue,
    status: stringStatus || 'Unknown',
    description: stringDescription || '',
    isPercentage: booleanIsPercentage
  };

  // Log the metric data to verify it's correct
  console.log('MetricCard props:', { title, value, status, description, isPercentage });
  console.log('Metric data being passed to InfoTooltip:', JSON.stringify(metricData, null, 2));

  return (
    <div className="bg-white rounded-xl p-6 shadow-sm border border-gray-100">
      <div className="flex items-center">
        <h3 className="text-lg font-medium text-gray-900 mb-2">{title}</h3>
        <InfoTooltip 
          title={`About ${title}`} 
          entityType="metric" 
          entityName={title}
          data={metricData}
        />
      </div>
      <div className="flex items-baseline space-x-2">
        <span className="text-4xl font-bold text-gray-900">
          {isPercentage ? `${value}%` : value}
        </span>
        <span className={`text-sm font-medium px-2.5 py-0.5 rounded-full ${
          status === 'Excellent' ? 'bg-green-100 text-green-800' :
          status === 'Good' ? 'bg-blue-100 text-blue-800' :
          'bg-yellow-100 text-yellow-800'
        }`}>
          {status}
        </span>
      </div>
      <p className="mt-2 text-sm text-gray-500">{description}</p>
    </div>
  );
}; 