import { Link } from 'react-router-dom';
import { ChevronRight } from 'lucide-react';

interface BreadcrumbSegment {
  title: string;
  href: string;
}

interface BreadcrumbProps {
  segments: BreadcrumbSegment[];
}

export function Breadcrumb({ segments }: BreadcrumbProps) {
  return (
    <nav className="flex" aria-label="Breadcrumb">
      <ol className="flex items-center space-x-2">
        {segments.map((segment, index) => (
          <li key={segment.href} className="flex items-center">
            {index > 0 && <ChevronRight className="h-4 w-4 text-gray-400 mx-2" />}
            <Link
              to={segment.href}
              className={`text-sm font-medium ${
                index === segments.length - 1
                  ? 'text-gray-900'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              {segment.title}
            </Link>
          </li>
        ))}
      </ol>
    </nav>
  );
}