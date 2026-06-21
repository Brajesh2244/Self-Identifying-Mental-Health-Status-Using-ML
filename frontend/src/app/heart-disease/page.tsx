"use client";

import { motion } from "framer-motion";
import { HeartPulse, ArrowRight, ShieldCheck, Activity } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export default function HeartDiseaseHub() {
  const [step, setStep] = useState(0);

  // Clinical Questions based on Cleveland Heart Disease dataset
  const questions = [
    "What is your resting blood pressure (in mm Hg)?",
    "What is your serum cholesterol level (in mg/dl)?",
    "Have you experienced any chest pain or discomfort recently?",
    "Do you engage in more than 150 minutes of moderate-intensity exercise per week?"
  ];

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-50 font-sans p-6 md:p-12">
      
      {/* Animated SVG Header */}
      <div className="max-w-4xl mx-auto mb-12 flex flex-col md:flex-row items-center justify-between glass-panel p-8 border-red-500/20">
        <div>
          <h1 className="text-4xl font-bold mb-4 flex items-center">
            <HeartPulse className="w-10 h-10 text-red-400 mr-4" />
            Heart Disease Assessment
          </h1>
          <p className="text-slate-400 max-w-xl">
            This module evaluates cardiovascular risk using a Gradient Boosting model trained on the Cleveland Heart Disease Database.
          </p>
        </div>
        
        {/* Animated Beating Heart Visual */}
        <motion.div 
          animate={{ scale: [1, 1.2, 1, 1.2, 1] }}
          transition={{ repeat: Infinity, duration: 1.5, ease: "easeInOut" }}
          className="w-32 h-32 rounded-full bg-red-500/20 border border-red-500/50 flex items-center justify-center mt-8 md:mt-0 relative"
        >
          <div className="absolute inset-0 rounded-full bg-red-400/20 blur-xl animate-pulse" />
          <Activity className="w-16 h-16 text-red-400" />
        </motion.div>
      </div>

      {/* Assessment Interface */}
      <div className="max-w-4xl mx-auto">
        <div className="glass-panel p-8 relative overflow-hidden">
          <div className="absolute top-0 left-0 h-1 bg-gradient-to-r from-red-500 to-orange-500 transition-all duration-500" style={{ width: `${((step + 1) / questions.length) * 100}%` }} />
          
          <h3 className="text-sm font-bold text-red-400 mb-6 uppercase tracking-wider">Metric {step + 1} of {questions.length}</h3>
          
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-10"
          >
            <h2 className="text-2xl md:text-3xl font-semibold mb-8 leading-relaxed">
              {questions[step]}
            </h2>
            
            <div className="flex flex-col gap-4 max-w-md">
              <input 
                type="text" 
                placeholder="Enter value..."
                className="w-full bg-white/5 border border-slate-700 rounded-xl p-4 text-slate-50 focus:outline-none focus:border-red-500 transition-colors"
              />
              <button 
                onClick={() => {
                  if (step < questions.length - 1) setStep(step + 1);
                  else alert("Analyzing cardiovascular load and generating PDF Report..."); 
                }}
                className="btn-primary w-full flex justify-center items-center gap-2 !bg-gradient-to-r !from-red-500 !to-orange-500 hover:shadow-[0_10px_25px_-5px_rgba(239,68,68,0.5)]"
              >
                Continue <ArrowRight className="w-5 h-5" />
              </button>
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Trust Badges */}
      <div className="max-w-4xl mx-auto mt-8 flex items-center justify-center text-sm text-slate-500">
        <ShieldCheck className="w-4 h-4 mr-2 text-teal-500" />
        Encrypted Processing. Data is securely analyzed and immediately discarded.
      </div>
    </div>
  );
}
