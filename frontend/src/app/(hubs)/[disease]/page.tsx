"use client";

import { useState, useEffect } from "react";
import { useParams, useRouter } from "next/navigation";
import { questionPools, Question } from "../../../data/questionPools";
import { motion } from "framer-motion";
import { Activity, ArrowLeft, RefreshCw, BarChart2, CheckCircle } from "lucide-react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function HubAssessmentPage() {
  const params = useParams();
  const router = useRouter();
  const diseaseId = params.disease as string;

  const [questions, setQuestions] = useState<Question[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [score, setScore] = useState(0);
  const [isFinished, setIsFinished] = useState(false);

  // Initialize random 10 questions on mount
  useEffect(() => {
    const pool = questionPools[diseaseId];
    if (!pool) {
      router.push("/dashboard"); // Invalid disease, go back
      return;
    }
    
    // Shuffle and pick 10
    const shuffled = [...pool].sort(() => 0.5 - Math.random());
    const selected = shuffled.slice(0, 10);
    setQuestions(selected);
  }, [diseaseId, router]);

  if (questions.length === 0) return <div className="min-h-screen bg-[#0B1120] text-white flex items-center justify-center">Loading Data...</div>;

  const currentQ = questions[currentIndex];
  
  // Calculate probability and chart data
  const maxScore = questions.length * 3;
  const probability = Math.round((score / maxScore) * 100);
  
  const chartData = [
    { name: "Low Risk", value: 30, fill: "#10b981" },
    { name: "Mod Risk", value: 30, fill: "#f59e0b" },
    { name: "High Risk", value: 40, fill: "#ef4444" },
    { name: "Your Score", value: probability, fill: "#0ea5e9" }
  ];

  const handleOptionClick = (optionScore: number) => {
    setScore((prev) => prev + optionScore);
    
    if (currentIndex < questions.length - 1) {
      setCurrentIndex((prev) => prev + 1);
    } else {
      setIsFinished(true);
    }
  };

  const getRiskLevel = () => {
    if (probability < 30) return { text: "Low Risk", color: "text-green-400" };
    if (probability < 60) return { text: "Moderate Risk", color: "text-yellow-400" };
    return { text: "High Risk", color: "text-red-400" };
  };

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-50 font-sans p-6">
      <div className="max-w-4xl mx-auto pt-10">
        
        {/* Header */}
        <button 
          onClick={() => router.push("/dashboard")}
          className="flex items-center text-slate-400 hover:text-white mb-8 transition-colors"
        >
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back to Dashboard
        </button>

        <h1 className="text-4xl font-bold mb-2 capitalize text-gradient">
          {diseaseId.replace("-", " ")} Assessment
        </h1>
        <p className="text-slate-400 mb-10">
          Complete the following 10 dynamic questions to generate your real-time probability graph.
        </p>

        {!isFinished ? (
          <motion.div 
            key={currentIndex}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="glass-panel p-8"
          >
            <div className="mb-6 flex justify-between items-center text-sm font-medium text-sky-400">
              <span>Question {currentIndex + 1} of 10</span>
              <span>{Math.round(((currentIndex) / 10) * 100)}% Completed</span>
            </div>
            
            <div className="w-full h-2 bg-slate-800 rounded-full mb-8 overflow-hidden">
              <div 
                className="h-full bg-gradient-to-r from-sky-400 to-indigo-500 transition-all duration-500"
                style={{ width: `${((currentIndex) / 10) * 100}%` }}
              />
            </div>

            <h2 className="text-2xl font-semibold mb-8 leading-relaxed">
              {currentQ.text}
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {currentQ.options.map((opt, i) => (
                <button
                  key={i}
                  onClick={() => handleOptionClick(opt.score)}
                  className="p-4 rounded-xl border border-slate-700/50 bg-white/5 hover:bg-sky-500/20 hover:border-sky-500/50 transition-all text-left group"
                >
                  <div className="flex items-center">
                    <div className="w-8 h-8 rounded-full bg-slate-800 group-hover:bg-sky-500 flex items-center justify-center mr-4 transition-colors">
                      {String.fromCharCode(65 + i)}
                    </div>
                    <span className="text-lg">{opt.text}</span>
                  </div>
                </button>
              ))}
            </div>
          </motion.div>
        ) : (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-panel p-8 md:p-12"
          >
            <div className="flex flex-col md:flex-row items-center gap-12">
              
              {/* Results Text */}
              <div className="flex-1 text-center md:text-left">
                <div className="inline-flex items-center justify-center p-4 rounded-full bg-sky-500/10 mb-6">
                  <Activity className="w-10 h-10 text-sky-400" />
                </div>
                <h2 className="text-3xl font-bold mb-4">Assessment Complete</h2>
                <p className="text-slate-300 text-lg mb-6">
                  Based on your responses across the 10 parameters, our ML-derived logic has calculated your risk profile.
                </p>
                
                <div className="bg-slate-900/50 rounded-2xl p-6 border border-slate-700/50 mb-8">
                  <p className="text-sm text-slate-400 mb-2 uppercase tracking-widest">Calculated Probability</p>
                  <h3 className={`text-6xl font-bold ${getRiskLevel().color} mb-2`}>
                    {probability}%
                  </h3>
                  <p className={`text-xl font-medium ${getRiskLevel().color}`}>
                    {getRiskLevel().text}
                  </p>
                </div>

                <button 
                  onClick={() => window.location.reload()}
                  className="btn-primary w-full flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-5 h-5" /> Retake Assessment
                </button>
              </div>

              {/* Chart */}
              <div className="flex-1 w-full h-[400px]">
                <h3 className="text-xl font-bold mb-6 flex items-center gap-2">
                  <BarChart2 className="w-6 h-6 text-sky-400" />
                  Risk Distribution
                </h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip 
                      cursor={{fill: 'rgba(255,255,255,0.05)'}}
                      contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px' }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {
                        chartData.map((entry, index) => (
                          <cell key={`cell-${index}`} fill={entry.fill} />
                        ))
                      }
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

            </div>
          </motion.div>
        )}
      </div>
    </div>
  );
}
