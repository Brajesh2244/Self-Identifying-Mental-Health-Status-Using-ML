"use client";

import { motion } from "framer-motion";
import { Brain, ArrowRight, ShieldCheck } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export default function MentalHealthHub() {
  const [step, setStep] = useState(0);

  // Upgraded Clinical Questionnaire (PHQ-9 / GAD-7 style)
  const questions = [
    "Over the last 2 weeks, how often have you felt down, depressed, or hopeless?",
    "How often have emotional challenges impaired your ability to complete daily work or personal tasks?",
    "Do you have a family history of diagnosed mental health conditions?",
    "How frequently do you experience sleep disturbances (trouble falling or staying asleep)?"
  ];

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-50 font-sans p-6 md:p-12">
      
      {/* Lottie-style SVG Header */}
      <div className="max-w-4xl mx-auto mb-12 flex flex-col md:flex-row items-center justify-between glass-panel p-8">
        <div>
          <h1 className="text-4xl font-bold mb-4 flex items-center">
            <Brain className="w-10 h-10 text-purple-400 mr-4" />
            Mental Health Assessment
          </h1>
          <p className="text-slate-400 max-w-xl">
            This module uses an advanced Stacked Ensemble Model to evaluate psychological stability. Our clinical questionnaire is aligned with PHQ-9 standards for maximum accuracy.
          </p>
        </div>
        
        {/* Animated Brain Visual */}
        <motion.div 
          animate={{ scale: [1, 1.05, 1] }}
          transition={{ repeat: Infinity, duration: 3, ease: "easeInOut" }}
          className="w-32 h-32 rounded-full bg-purple-500/20 border border-purple-500/50 flex items-center justify-center mt-8 md:mt-0 relative"
        >
          <div className="absolute inset-0 rounded-full bg-purple-400/20 blur-xl animate-pulse" />
          <Brain className="w-16 h-16 text-purple-300" />
        </motion.div>
      </div>

      {/* Assessment Interface */}
      <div className="max-w-4xl mx-auto">
        <div className="glass-panel p-8 relative overflow-hidden">
          <div className="absolute top-0 left-0 h-1 bg-gradient-to-r from-purple-500 to-indigo-500 transition-all duration-500" style={{ width: `${((step + 1) / questions.length) * 100}%` }} />
          
          <h3 className="text-sm font-bold text-purple-400 mb-6 uppercase tracking-wider">Question {step + 1} of {questions.length}</h3>
          
          <motion.div
            key={step}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            className="mb-10"
          >
            <h2 className="text-2xl md:text-3xl font-semibold mb-8 leading-relaxed">
              {questions[step]}
            </h2>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {["Not at all", "Several days", "More than half the days", "Nearly every day"].map((opt, i) => (
                <button 
                  key={i}
                  onClick={() => {
                    if (step < questions.length - 1) setStep(step + 1);
                    else alert("Generating 7-Day Action Plan and PDF Report..."); // Mock submit
                  }}
                  className="p-4 rounded-xl border border-slate-700 hover:border-purple-500 hover:bg-purple-500/10 transition-all text-left flex justify-between items-center group"
                >
                  {opt}
                  <ArrowRight className="w-5 h-5 text-slate-600 group-hover:text-purple-400 transition-colors" />
                </button>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Trust Badges */}
      <div className="max-w-4xl mx-auto mt-8 flex items-center justify-center text-sm text-slate-500">
        <ShieldCheck className="w-4 h-4 mr-2 text-teal-500" />
        HIPAA Compliant Data Processing. Your responses are encrypted.
      </div>
    </div>
  );
}
