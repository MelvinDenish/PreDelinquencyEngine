import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Barclays PDI Engine | Pre-Delinquency Intelligence Platform",
  description: "Barclays Pre-Delinquency Intervention Engine — Real-time AI-driven credit risk scoring, 4-model ensemble, SHAP explainability, automated multi-channel interventions and portfolio what-if stress testing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap"
          rel="stylesheet"
        />
        {/* Security headers via meta tags */}
        <meta httpEquiv="X-Content-Type-Options" content="nosniff" />
        <meta httpEquiv="X-Frame-Options" content="DENY" />
        <meta httpEquiv="Referrer-Policy" content="strict-origin-when-cross-origin" />
        {/* Barclays brand colour for browser chrome */}
        <meta name="theme-color" content="#002C6C" />
        <meta name="application-name" content="Barclays PDI Engine" />
        {/* Prevent search engine indexing in production */}
        <meta name="robots" content="noindex, nofollow" />
      </head>
      <body className="antialiased">{children}</body>
    </html>
  );
}
