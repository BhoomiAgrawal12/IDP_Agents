// About.tsx
"use client"; // Important for interactivity

import React from "react";
import { Brain, FileText, Database, Shield, ExternalLink } from "lucide-react";

// FeatureCard Component
const FeatureCard: React.FC<{ icon: React.ReactNode; title: string; description: string }> = ({ icon, title, description }) => {
  return (
    <div className="p-8 rounded-2xl bg-gray-800/50 border border-gray-700 hover:border-cyan-500/50 transition-all duration-300 group min-h-[300px]">
      <div className="mb-6">{icon}</div>
      <h3 className="text-2xl font-bold text-white mb-3 group-hover:text-cyan-400 transition-colors duration-200">
        {title}
      </h3>
      <p className="text-gray-400 text-lg">
        {description}
      </p>
    </div>
  );
};

// About Component
const About: React.FC<{ onTryChat: () => void }> = ({ onTryChat }) => {
  return (
    <div
      id="about"
      className="bg-gradient-to-b from-[#29153D] to-[#000000] min-h-screen py-5 px-5 md:py-9 md:px-16 lg:py-24 lg:px-40"
    >
      {/* Main Content */}
      <div className="flex flex-col lg:flex-row justify-start">
        <div className="flex-1">

          {/* Hero Section */}
          <section className="container mx-auto px-4 pt-20 pb-32 text-center">
            <h1 className="text-5xl md:text-6xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 mb-6">
              Turn Documents into Intelligence
            </h1>
            <p className="text-xl md:text-2xl text-gray-300 mb-12 max-w-3xl mx-auto">
              Ask your data in natural language. Get structured datasets — effortlessly.
            </p>

            {/* Hero Image */}
            <div className="flex justify-center items-center">
              <picture className="w-[30%] md:w-[30%] lg:w-[30%] floating-image">
                <source type="image/avif" srcSet="/2.avif" />
                <source type="image/webp" srcSet="/2.webp" />
                <img src="/2.png" alt="Mascot" loading="lazy" />
              </picture>
            </div>
          </section>

          {/* Try Chat Button */}
          {/* <button
            onClick={onTryChat}
            className="px-8 py-4 text-lg font-semibold text-white bg-gradient-to-r from-blue-600 to-cyan-500 rounded-full hover:from-blue-700 hover:to-cyan-600 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-cyan-500/25 mb-12"
          >
            Try AI Chat
          </button> */}
            <div className="flex justify-center mb-12">
            <button
              onClick={onTryChat}
              className="px-8 py-4 text-lg font-semibold text-white bg-gradient-to-r from-blue-600 to-cyan-500 rounded-full hover:from-blue-700 hover:to-cyan-600 transform hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-cyan-500/25"
            >
              Try AI Chat
            </button>
          </div>

          {/* Features Section */}
          <section id="features" className="py-24">
            <div className="container mx-auto px-4">
              <div className="grid gap-10 md:grid-cols-2 lg:grid-cols-4">
                <FeatureCard
                  icon={<Brain className="w-10 h-10 text-cyan-400" />}
                  title="Natural Language Understanding"
                  description="Understand queries exactly like a human agent would."
                />
                <FeatureCard
                  icon={<FileText className="w-10 h-10 text-cyan-400" />}
                  title="Intelligent Parsing & OCR"
                  description="Extract data from scanned documents and PDFs with precision."
                />
                <FeatureCard
                  icon={<Database className="w-10 h-10 text-cyan-400" />}
                  title="Automatic Dataset Building"
                  description="Create structured, ready-to-use datasets and files."
                />
                <FeatureCard
                  icon={<Shield className="w-10 h-10 text-cyan-400" />}
                  title="Privacy & Quality Control"
                  description="Ensure data consistency, accuracy, and optional de-identification."
                />
              </div>
            </div>
          </section>

          {/* Footer */}
          <footer className="py-8">
            <div className="container mx-auto px-4">
              <div className="flex justify-center items-center gap-8 text-gray-400">
                {["About", "Privacy", "Contact"].map((item) => (
                  <a
                    key={item}
                    href="#"
                    className="hover:text-white transition-colors duration-200 flex items-center gap-1"
                  >
                    {item} <ExternalLink className="w-4 h-4" />
                  </a>
                ))}
              </div>
            </div>
          </footer>

        </div>
      </div>
    </div>
  );
};

export default About;
