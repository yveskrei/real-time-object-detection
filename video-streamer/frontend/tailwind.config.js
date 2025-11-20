/** @type {import('tailwindcss').Config} */
const colors = require('tailwindcss/colors')

export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Modern Dark Theme Palette (Zinc based)
        background: '#09090b', // Zinc 950
        surface: '#18181b',    // Zinc 900
        'surface-hover': '#27272a', // Zinc 800
        border: '#27272a',     // Zinc 800

        // Primary Accent (Indigo)
        primary: {
          DEFAULT: '#6366f1', // Indigo 500
          hover: '#4f46e5',   // Indigo 600
          foreground: '#ffffff',
        },

        // Semantic Colors
        success: '#22c55e', // Green 500
        warning: '#f59e0b', // Amber 500
        danger: '#ef4444',  // Red 500
        info: '#3b82f6',    // Blue 500
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
