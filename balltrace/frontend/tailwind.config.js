/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      fontFamily: {
        sans: ['Sora', 'IBM Plex Sans', 'system-ui', 'sans-serif'],
        display: ['Sora', 'system-ui', 'sans-serif'],
        body: ['IBM Plex Sans', 'system-ui', 'sans-serif'],
        mono: ['IBM Plex Mono', 'ui-monospace', 'monospace'],
      },
      colors: {
        base: {
          950: '#0a0c10',
          900: '#0e1116',
          850: '#12161d',
          800: '#161b23',
          700: '#1d232d',
          600: '#272f3b',
        },
        neon: {
          cyan: '#00ffd0',
          green: '#39ff14',
        },
      },
      boxShadow: {
        glow: '0 0 12px rgba(0, 255, 208, 0.55)',
        'glow-sm': '0 0 6px rgba(0, 255, 208, 0.45)',
      },
      backgroundImage: {
        'grid-fade':
          'radial-gradient(circle at 1px 1px, rgba(255,255,255,0.06) 1px, transparent 0)',
      },
    },
  },
  plugins: [],
}
