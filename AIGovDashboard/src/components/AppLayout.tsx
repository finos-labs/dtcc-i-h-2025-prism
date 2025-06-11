import React, { ReactNode } from "react";
import { Outlet } from "react-router-dom";
import Header from "./Header";
import AppSidebar from "./AppSidebar";
import { useAuth } from "../contexts/AuthContext";

interface AppLayoutProps {
  children?: ReactNode;
  showSidebar?: boolean;
  showHeader?: boolean;
}

const AppLayout: React.FC<AppLayoutProps> = ({
  children,
  showSidebar = true,
  showHeader = true,
}) => {
  const { signOut } = useAuth();

  const handleLogout = async () => {
    await signOut();
  };

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {showHeader && <Header onLogout={handleLogout} />}
      <div className="flex flex-1">
        {showSidebar && <AppSidebar />}
        <main className="flex-1 overflow-auto">{children || <Outlet />}</main>
      </div>
    </div>
  );
};

export default AppLayout;
