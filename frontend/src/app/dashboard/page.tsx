"use client";

import { motion } from "framer-motion";
import { Brain, HeartPulse, Stethoscope, Droplet, Activity, Moon, Salad, User, Award, FileText, Settings, ShieldAlert, Thermometer, Apple } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export default function PersonalDashboard() {
  const [activeTab, setActiveTab] = useState("hubs");

  const diseases = [
    { title: "Mental Health", icon: Brain, color: "text-purple-400", bg: "bg-purple-500/20", path: "/mental-health" },
    { title: "Heart Disease", icon: HeartPulse, color: "text-red-400", bg: "bg-red-500/20", path: "/heart-disease" },
    { title: "Diabetes Risk", icon: Droplet, color: "text-blue-400", bg: "bg-blue-500/20", path: "#" },
    { title: "Kidney Health", icon: Activity, color: "text-amber-400", bg: "bg-amber-500/20", path: "#" },
    { title: "Liver Disease", icon: ShieldAlert, color: "text-orange-400", bg: "bg-orange-500/20", path: "#" },
    { title: "Obesity Risk", icon: Apple, color: "text-emerald-400", bg: "bg-emerald-500/20", path: "#" },
    { title: "Sleep Disorders", icon: Moon, color: "text-indigo-400", bg: "bg-indigo-500/20", path: "#" },
    { title: "Lifestyle Risk", icon: Salad, color: "text-lime-400", bg: "bg-lime-500/20", path: "#" },
    { title: "Hypertension", icon: Thermometer, color: "text-rose-400", bg: "bg-rose-500/20", path: "#" },
    { title: "General Wellness", icon: Stethoscope, color: "text-teal-400", bg: "bg-teal-500/20", path: "#" }
  ];

  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-50 font-sans p-6 md:p-12 relative overflow-hidden">
      
      {/* Background Blurs */}
      <div className="absolute top-[-10%] right-[-5%] w-[500px] h-[500px] bg-sky-500/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] left-[-5%] w-[600px] h-[600px] bg-indigo-500/10 rounded-full blur-[150px] pointer-events-none" />

      <div className="max-w-7xl mx-auto z-10 relative">
        
        {/* Profile Header */}
        <div className="glass-panel p-8 mb-10 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-6">
            <div className="w-20 h-20 rounded-full bg-gradient-to-br from-sky-400 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
              <User className="w-10 h-10 text-white" />
            </div>
            <div>
              <h1 className="text-3xl font-bold">Welcome back, Alex!</h1>
              <p className="text-slate-400">Your health journey is looking great. Language: <span className="text-white font-semibold">English (EN)</span></p>
            </div>
          </div>
          <div className="flex gap-4">
            <Link href="/chat" className="btn-primary">Ask HealthGPT</Link>
            <button className="btn-secondary flex items-center gap-2"><Settings className="w-4 h-4" /> Preferences</button>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-4 border-b border-white/10 mb-8 pb-4 overflow-x-auto scrollbar-hide">
          <button onClick={() => setActiveTab("hubs")} className={`px-4 py-2 font-semibold transition-colors ${activeTab === 'hubs' ? 'text-sky-400 border-b-2 border-sky-400' : 'text-slate-400 hover:text-white'}`}>10 Disease Hubs</button>
          <button onClick={() => setActiveTab("achievements")} className={`px-4 py-2 font-semibold transition-colors ${activeTab === 'achievements' ? 'text-yellow-400 border-b-2 border-yellow-400' : 'text-slate-400 hover:text-white'}`}>Achievements</button>
          <button onClick={() => setActiveTab("reports")} className={`px-4 py-2 font-semibold transition-colors ${activeTab === 'reports' ? 'text-indigo-400 border-b-2 border-indigo-400' : 'text-slate-400 hover:text-white'}`}>Saved PDF Reports</button>
        </div>

        {/* Dynamic Content based on Tabs */}
        {activeTab === "hubs" && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-4">
            {diseases.map((disease, idx) => (
              <Link href={disease.path} key={idx}>
                <motion.div whileHover={{ scale: 1.05 }} className="glass-panel p-6 flex flex-col items-center justify-center text-center h-48 cursor-pointer group hover:bg-white/5 transition-all">
                  <div className={`w-16 h-16 rounded-2xl ${disease.bg} flex items-center justify-center mb-4 group-hover:shadow-[0_0_20px_rgba(255,255,255,0.1)] transition-shadow`}>
                    <disease.icon className={`w-8 h-8 ${disease.color}`} />
                  </div>
                  <h3 className="font-bold text-sm tracking-wide">{disease.title}</h3>
                </motion.div>
              </Link>
            ))}
          </motion.div>
        )}

        {activeTab === "achievements" && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Gamification / Achievements */}
            <div className="glass-panel p-6 border-yellow-500/30 bg-yellow-500/5 relative overflow-hidden">
              <div className="absolute top-[-20px] right-[-20px] text-yellow-500/10"><Award className="w-32 h-32" /></div>
              <div className="w-12 h-12 rounded-full bg-yellow-500/20 flex items-center justify-center mb-4 relative z-10"><span className="text-2xl">🌟</span></div>
              <h3 className="text-xl font-bold text-yellow-400 mb-2 relative z-10">Wellness Explorer</h3>
              <p className="text-slate-300 text-sm relative z-10">Completed your first general wellness and mental health assessment.</p>
            </div>
            
            <div className="glass-panel p-6 border-teal-500/30 bg-teal-500/5">
              <div className="w-12 h-12 rounded-full bg-teal-500/20 flex items-center justify-center mb-4"><span className="text-2xl">🔥</span></div>
              <h3 className="text-xl font-bold text-teal-400 mb-2">7-Day Streak</h3>
              <p className="text-slate-300 text-sm">Followed the AI Action Plan consecutively for 7 days.</p>
            </div>

            <div className="glass-panel p-6 opacity-50 border-dashed border-white/20 flex flex-col items-center justify-center text-center">
              <div className="w-12 h-12 rounded-full bg-slate-800 flex items-center justify-center mb-4"><span className="text-2xl text-slate-500">🔒</span></div>
              <h3 className="text-lg font-bold text-slate-400 mb-2">Heart Healthy</h3>
              <p className="text-slate-500 text-xs">Complete the Heart Disease Assessment to unlock.</p>
            </div>
          </motion.div>
        )}

        {activeTab === "reports" && (
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-panel p-12 flex flex-col items-center justify-center text-center">
            {/* Premium Empty State */}
            <div className="w-24 h-24 rounded-full bg-indigo-500/10 flex items-center justify-center mb-6">
              <FileText className="w-12 h-12 text-indigo-400" />
            </div>
            <h2 className="text-2xl font-bold mb-2">No Reports Generated Yet</h2>
            <p className="text-slate-400 max-w-md mb-8">Take your first disease assessment to automatically generate your premium PDF Health Report and 7-Day Action Plan.</p>
            <button onClick={() => setActiveTab("hubs")} className="btn-primary">Browse Disease Hubs</button>
          </motion.div>
        )}

      </div>
    </div>
  );
}
