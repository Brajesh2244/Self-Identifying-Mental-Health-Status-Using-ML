import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "AI-Powered HealthTech Intelligence",
  description: "Predict, Analyze, and Improve Your Health Using Artificial Intelligence.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${inter.variable} antialiased`}>
      <body className="font-sans bg-[#0B1120] text-slate-50 min-h-screen">
        {children}
      </body>
    </html>
  );
}
