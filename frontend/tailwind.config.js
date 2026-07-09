/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class', // support dark mode
  theme: {
    extend: {
      colors: {
        wa: {
          green: "#00a884",
          lightgreen: "#53bdeb",
          darkBg: "#111b21",
          sidebarBg: "#202c33",
          chatBg: "#0b141a",
          chatBubbleOut: "#005c4b",
          chatBubbleIn: "#202c33",
          textActive: "#e9edef",
          textMuted: "#8696a0",
          border: "#374248"
        }
      }
    },
  },
  plugins: [],
}
