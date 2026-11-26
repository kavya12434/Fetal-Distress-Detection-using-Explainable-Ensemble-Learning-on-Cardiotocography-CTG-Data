/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class", ".dark"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        "surface-2": "var(--surface-2)",
        border: "var(--border)",
        text: {
          DEFAULT: "var(--text)",
          muted: "var(--text-muted)",
        },
        accent: "var(--accent)",
        // Non-semantic amber for out-of-distribution warnings only.
        warning: "var(--warning)",
        // Semantic colors: reserved EXCLUSIVELY for class identity.
        normal: "var(--normal)",
        suspect: "var(--suspect)",
        pathological: "var(--pathological)",
      },
      fontFamily: {
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "monospace"],
      },
      borderRadius: {
        // Max radius is 6px per the design system.
        none: "0px",
        sm: "3px",
        DEFAULT: "4px",
        md: "5px",
        lg: "6px",
      },
      letterSpacing: {
        tightheading: "-0.01em",
        section: "0.08em",
      },
      fontSize: {
        section: ["11px", { lineHeight: "1.2", letterSpacing: "0.08em" }],
        nav: ["13px", { lineHeight: "1.3" }],
      },
      transitionTimingFunction: {
        "out-clinical": "cubic-bezier(0.16, 1, 0.3, 1)",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
