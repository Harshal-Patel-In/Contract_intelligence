import React, { useEffect, useState } from 'react';
import { API_BASE_URL } from '../config';

// Professional SVG logo — purple scales
const LogoIcon: React.FC = () => (
  <svg className="logo-svg-icon" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="logoGrad" x1="0" y1="0" x2="48" y2="48">
        <stop offset="0%" stopColor="#8b5cf6" />
        <stop offset="50%" stopColor="#d946ef" />
        <stop offset="100%" stopColor="#6366f1" />
      </linearGradient>
      <linearGradient id="scaleGrad" x1="0" y1="0" x2="48" y2="48">
        <stop offset="0%" stopColor="#c084fc" />
        <stop offset="100%" stopColor="#818cf8" />
      </linearGradient>
    </defs>
    {/* Base/Pillar */}
    <path d="M22 10V38H18L16 42H32L30 38H26V10H22Z" fill="url(#logoGrad)" />
    {/* Top Beam - Animated */}
    <path className="logo-beam" d="M8 10H40V12H8V10Z" fill="url(#logoGrad)" />
    {/* Center Pivot Pin */}
    <circle cx="24" cy="11" r="3" fill="url(#logoGrad)" />
    {/* Left Scale Assembly - Animated */}
    <g className="logo-left-scale">
      <path d="M10 12L6 26M10 12L14 26" stroke="url(#scaleGrad)" strokeWidth="1" />
      <path d="M4 26C4 29.3 6.7 32 10 32C13.3 32 16 29.3 16 26H4Z" fill="url(#scaleGrad)" opacity="0.9" />
    </g>
    {/* Right Scale Assembly - Animated */}
    <g className="logo-right-scale">
      <path d="M38 12L34 26M38 12L42 26" stroke="url(#scaleGrad)" strokeWidth="1" />
      <path d="M32 26C32 29.3 34.7 32 38 32C41.3 32 44 29.3 44 26H32Z" fill="url(#scaleGrad)" opacity="0.9" />
    </g>
    {/* Pulsing AI Scanner Core Node (Contract Intelligence / Analysis symbol) */}
    <circle className="logo-scanner-node" cx="24" cy="25" r="2.5" fill="#06b6d4" />
    {/* Sweeping AI Laser Scan Line */}
    <line className="logo-scan-line" x1="4" y1="12" x2="44" y2="12" stroke="#06b6d4" strokeWidth="1.5" />
  </svg>
);

// Theme toggle icons
const SunIcon: React.FC = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="5" />
    <line x1="12" y1="1" x2="12" y2="3" />
    <line x1="12" y1="21" x2="12" y2="23" />
    <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
    <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
    <line x1="1" y1="12" x2="3" y2="12" />
    <line x1="21" y1="12" x2="23" y2="12" />
    <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
    <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
  </svg>
);

const MoonIcon: React.FC = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
  </svg>
);

export const Header: React.FC = () => {
  const [health, setHealth] = useState<'active' | 'degraded' | 'offline'>('active');
  const [healthText, setHealthText] = useState('Services: Active');
  const [theme, setTheme] = useState<'dark' | 'light'>(() => {
    const saved = localStorage.getItem('zaalima_theme');
    return (saved === 'light' ? 'light' : 'dark');
  });

  // Apply theme to HTML element
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('zaalima_theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  };

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE_URL}/health`);
      if (!res.ok) throw new Error('Health check error');
      const data = await res.json();

      if (data.status === 'healthy') {
        setHealth('active');
        setHealthText('Services: Active');
      } else if (data.status === 'degraded') {
        setHealth('active');
        setHealthText('Services: Active');
      } else {
        setHealth('degraded');
        setHealthText('Services: Degraded');
      }
    } catch (err) {
      setHealth('offline');
      setHealthText('Services: Offline');
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header>
      <div className="logo-group">
        <LogoIcon />
        <div className="logo-text">
          <h1>ClauseForge</h1>
          <p>Contract Intelligence & Risk Platform</p>
        </div>
      </div>

      <div className="health-panel">
        <div className={`health-dot ${health}`} />
        <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
          {healthText}
        </span>
        <button
          type="button"
          className="theme-toggle-btn"
          onClick={toggleTheme}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
        <a
          href={`${API_BASE_URL}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary"
          style={{ padding: '6px 12px', fontSize: '0.8rem', marginLeft: '4px' }}
        >
          API Docs
        </a>
      </div>
    </header>
  );
};

