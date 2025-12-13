/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'buy': '#22c55e',
        'sell': '#ef4444',
        'hold': '#6b7280',
        'strong-buy': '#16a34a',
        'strong-sell': '#dc2626',
      }
    },
  },
  plugins: [],
}
