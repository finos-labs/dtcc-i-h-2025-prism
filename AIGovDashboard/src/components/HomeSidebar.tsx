import React from "react";
import { NavLink } from "react-router-dom";
import { Home } from "lucide-react";

const HomeSidebar: React.FC = () => {
  // Navigation items for the home sidebar
  const mainNavItems = [
    { label: "Dashboard", icon: <Home className="h-5 w-5" />, href: "/home" },
  ];

  // Common NavItem component for consistent styling
  const NavItem = ({
    item,
  }: {
    item: { label: string; icon: React.ReactNode; href: string };
  }) => (
    <NavLink
      to={item.href}
      className={({ isActive }) => `
        flex items-center px-4 py-3 text-sm font-medium rounded-lg
        ${
          isActive
            ? "bg-primary/10 text-primary"
            : "text-gray-700 hover:bg-gray-100 hover:text-gray-900"
        }
        transition-colors duration-200
      `}
    >
      <span className="mr-3">{item.icon}</span>
      {item.label}
    </NavLink>
  );

  return (
    <aside className="w-64 bg-white border-r border-gray-200 h-full flex flex-col">
      <div className="flex-1 px-3 py-4 space-y-6 overflow-y-auto">
        {/* Main Navigation */}
        <div>
          <div className="mt-3 space-y-1">
            {mainNavItems.map((item) => (
              <NavItem key={item.href} item={item} />
            ))}
          </div>
        </div>
      </div>
    </aside>
  );
};

export default HomeSidebar;
