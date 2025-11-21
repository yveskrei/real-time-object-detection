import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { LayoutDashboard, Tv, Video } from 'lucide-react';
import clsx from 'clsx';

interface LayoutProps {
    children: React.ReactNode;
}

export const Layout: React.FC<LayoutProps> = ({ children }) => {
    const location = useLocation();

    const tabs = [
        { name: 'Management', path: '/', icon: LayoutDashboard },
        { name: 'Stream Viewer', path: '/viewer', icon: Tv },
    ];

    return (
        <div className="min-h-screen flex flex-col">
            {/* Top Navigation Bar */}
            <header className="bg-surface/50 backdrop-blur-md border-b border-border sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center justify-between h-16">
                        {/* Logo */}
                        <div className="flex items-center space-x-3">
                            <div className="bg-primary/10 p-2 rounded-lg">
                                <Video className="w-6 h-6 text-primary" />
                            </div>
                            <h1 className="text-lg font-bold tracking-tight text-white">
                                Video<span className="text-primary">Streamer</span>
                            </h1>
                        </div>

                        {/* Navigation */}
                        <nav className="flex space-x-1">
                            {tabs.map((tab) => {
                                const Icon = tab.icon;
                                const isActive = location.pathname === tab.path;
                                return (
                                    <Link
                                        key={tab.path}
                                        to={tab.path}
                                        className={clsx(
                                            'flex items-center px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                                            isActive
                                                ? 'bg-primary/10 text-primary'
                                                : 'text-zinc-400 hover:text-white hover:bg-white/5'
                                        )}
                                    >
                                        <Icon className="w-4 h-4 mr-2" />
                                        {tab.name}
                                    </Link>
                                );
                            })}
                        </nav>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {children}
            </main>
        </div>
    );
};
