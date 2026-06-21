"use client";

import { motion } from "framer-motion";
import { Activity, Calendar, TrendingDown, ArrowUpRight, Award } from "lucide-react";
import Link from "next/link";

export default function Timeline() {
  const journey = [
    { date: "Oct 15, 2025", score: "High Risk", detail: "Stress levels critical. Insomnia detected.", color: "text-red-400 border-red-500/50" },
    { date: "Nov 22, 2025", score: "Moderate Risk", detail: "Began 7-Day Action Plan. Sleep improved.", color: "text-yellow-400 border-yellow-500/50" },
    { date: "Today", score: "Low Risk", detail: "Stress managed effectively. Wellness Explorer badge earned.", color: "text-teal-400 border-teal-500/50" }
  ];

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-50 font-sans p-6 md:p-12">
      <div className="max-w-5xl mx-auto">
        
        {/* Header & AI Insights Card */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="md:col-span-2 glass-panel p-8">
            <h1 className="text-4xl font-bold mb-2">Smart Health Timeline</h1>
            <p className="text-slate-400">Track your longitudinal wellness journey and assessment history.</p>
          </div>
          
          <div className="glass-panel p-8 bg-indigo-500/10 border-indigo-500/30 flex flex-col justify-center">
            <div className="flex items-center text-indigo-400 font-bold mb-2">
              <TrendingDown className="w-5 h-5 mr-2" /> AI Insight
            </div>
            <p className="text-sm text-slate-300">"Your overall psychological stability has improved by <span className="text-white font-bold">42%</span> since October. Consistent meditation logging has been a major contributing factor."</p>
          </div>
        </div>

        {/* Timeline Visualization */}
        <div className="glass-panel p-8 md:p-12">
          <div className="relative border-l border-slate-700 ml-4 md:ml-6">
            
            {journey.map((entry, index) => (
              <motion.div 
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.2 }}
                className="mb-10 ml-8 relative"
              >
                {/* Timeline Node */}
                <div className={`absolute -left-[41px] top-1 w-5 h-5 rounded-full bg-[#0B1120] border-4 ${entry.color}`} />
                
                <div className="glass-panel p-6 hover:-translate-y-1 transition-transform cursor-pointer group">
                  <div className="flex justify-between items-start mb-2">
                    <span className="flex items-center text-sm font-semibold text-slate-400 uppercase tracking-wider">
                      <Calendar className="w-4 h-4 mr-2" /> {entry.date}
                    </span>
                    <ArrowUpRight className="w-5 h-5 text-slate-600 group-hover:text-sky-400 transition-colors" />
                  </div>
                  
                  <h3 className={`text-2xl font-bold mb-2 ${entry.color.split(' ')[0]}`}>{entry.score}</h3>
                  <p className="text-slate-300">{entry.detail}</p>
                </div>
              </motion.div>
            ))}

          </div>
        </div>
        
        {/* Achievements Section */}
        <div className="mt-12 glass-panel p-8">
          <h2 className="text-2xl font-bold mb-6 flex items-center"><Award className="w-6 h-6 mr-3 text-yellow-500" /> Recent Achievements</h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="w-12 h-12 rounded-full bg-yellow-500/20 mx-auto flex items-center justify-center mb-3">
                <span className="text-2xl">🌟</span>
              </div>
              <h4 className="font-bold text-sm">Wellness Explorer</h4>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/10 text-center">
              <div className="w-12 h-12 rounded-full bg-teal-500/20 mx-auto flex items-center justify-center mb-3">
                <span className="text-2xl">🔥</span>
              </div>
              <h4 className="font-bold text-sm">7-Day Streak</h4>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
