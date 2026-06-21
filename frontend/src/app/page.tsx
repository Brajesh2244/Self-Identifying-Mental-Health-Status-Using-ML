"use client";

import { motion } from "framer-motion";
import { Activity, Brain, HeartPulse, Stethoscope, Video, ArrowRight, Play, CheckCircle } from "lucide-react";
import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-[#0B1120] text-slate-50 font-sans">
      
      {/* 1. Hero Video Section */}
      <section className="relative h-screen flex items-center justify-center overflow-hidden">
        {/* Placeholder for Video Background - currently using dynamic gradient */}
        <div className="absolute inset-0 z-0 bg-gradient-to-br from-[#0B1120] via-[#0EA5E9]/20 to-[#0B1120] opacity-80" />
        <div className="absolute inset-0 z-0 bg-[url('https://images.unsplash.com/photo-1576091160399-112ba8d25d1d?q=80&w=2070&auto=format&fit=crop')] bg-cover bg-center opacity-20 mix-blend-overlay" />
        
        <div className="relative z-10 text-center max-w-4xl px-4 flex flex-col items-center">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            <h1 className="text-5xl md:text-7xl font-bold tracking-tight mb-6">
              AI-Powered <br />
              <span className="text-gradient">Healthcare Intelligence</span>
            </h1>
          </motion.div>
          
          <motion.p 
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
            className="text-lg md:text-xl text-slate-300 mb-10 max-w-2xl"
          >
            Predict, Analyze, and Improve Your Health Using Artificial Intelligence. Experience a premium, multi-lingual AI companion designed to provide evidence-based lifestyle tracking and predictive analytics.
          </motion.p>

          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
            className="flex flex-col sm:flex-row gap-4"
          >
            <button className="btn-primary flex items-center justify-center gap-2 text-lg">
              <Activity className="w-5 h-5" />
              Start Assessment
            </button>
            <button className="btn-secondary flex items-center justify-center gap-2 text-lg">
              <Video className="w-5 h-5" />
              Recruiter Demo Mode
            </button>
          </motion.div>
        </div>
        
        {/* Animated Background Particles */}
        <motion.div 
          animate={{ y: [0, -20, 0] }}
          transition={{ repeat: Infinity, duration: 4, ease: "easeInOut" }}
          className="absolute bottom-10 z-10 opacity-50 flex flex-col items-center"
        >
          <span className="text-sm tracking-widest uppercase mb-2 text-sky-400">Scroll to Explore</span>
          <div className="w-[1px] h-12 bg-gradient-to-b from-sky-400 to-transparent" />
        </motion.div>
      </section>

      {/* 2. Trusted Statistics Section */}
      <section className="py-20 px-4 md:px-12 bg-[#0B1120] relative z-20">
        <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-8">
          {[
            { label: "Predictive Accuracy", value: "94.2%" },
            { label: "Diseases Analyzed", value: "10+" },
            { label: "AI Response Time", value: "<1s" },
            { label: "Data Sources", value: "WHO & CDC" }
          ].map((stat, i) => (
            <motion.div 
              key={i}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ delay: i * 0.1 }}
              className="glass-panel p-6 text-center"
            >
              <h3 className="text-3xl md:text-5xl font-bold text-gradient mb-2">{stat.value}</h3>
              <p className="text-slate-400 font-medium">{stat.label}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* 3. Disease Coverage Section */}
      <section className="py-24 px-4 md:px-12 relative overflow-hidden">
        {/* Glow effect */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-teal-500/10 rounded-full blur-[120px] pointer-events-none" />
        
        <div className="max-w-6xl mx-auto relative z-10">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold mb-4">Comprehensive <span className="text-gradient">Disease Hubs</span></h2>
            <p className="text-slate-400 max-w-2xl mx-auto">Our advanced models analyze vital signs and lifestyle choices to assess risks across multiple health domains.</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { title: "Mental Health", icon: Brain, desc: "PHQ-9 integrated assessments for depression, PTSD, and anxiety.", color: "text-purple-400" },
              { title: "Heart Disease", icon: HeartPulse, desc: "Cardiovascular load prediction using resting BP and cholesterol levels.", color: "text-red-400" },
              { title: "General Wellness", icon: Stethoscope, desc: "Holistic evaluation of sleep, diet, and lifestyle habits.", color: "text-teal-400" }
            ].map((hub, i) => (
              <motion.div 
                key={i}
                whileHover={{ y: -5 }}
                className="glass-panel p-8 group cursor-pointer"
              >
                <div className={`p-4 rounded-2xl bg-white/5 w-fit mb-6 ${hub.color}`}>
                  <hub.icon className="w-8 h-8" />
                </div>
                <h3 className="text-2xl font-bold mb-3">{hub.title}</h3>
                <p className="text-slate-400 leading-relaxed mb-6">{hub.desc}</p>
                <div className="flex items-center text-sky-400 font-medium group-hover:translate-x-2 transition-transform">
                  Explore Hub <ArrowRight className="w-4 h-4 ml-2" />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* 4. AI Assistant Showcase */}
      <section className="py-24 px-4 md:px-12 bg-gradient-to-b from-[#0B1120] to-[#0A0F1D]">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center gap-16">
          <div className="flex-1">
            <h2 className="text-4xl md:text-5xl font-bold mb-6">Your Multilingual <br/><span className="text-gradient">HealthGPT Companion</span></h2>
            <p className="text-slate-300 text-lg mb-8 leading-relaxed">
              Don't just get a static report. Our voice-enabled AI companion remembers your health journey, asks intelligent follow-up questions, and dynamically generates personalized 7-Day Action Plans.
            </p>
            
            <ul className="space-y-4 mb-8">
              {["English, Hindi & Kannada Voice Support", "Long-term Conversational Memory", "WHO & CDC Backed Knowledge", "Lottie-Animated AI Avatar"].map((feature, i) => (
                <li key={i} className="flex items-center text-slate-300">
                  <CheckCircle className="w-5 h-5 text-teal-400 mr-3" />
                  {feature}
                </li>
              ))}
            </ul>

            <button className="btn-primary">Try the Voice AI</button>
          </div>
          
          <div className="flex-1 w-full max-w-md relative">
            {/* Mockup of Chat UI */}
            <div className="glass-panel p-6 border-slate-700/50 shadow-2xl relative overflow-hidden">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-sky-400 to-indigo-500" />
              <div className="flex items-center gap-4 mb-6">
                <div className="w-12 h-12 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center animate-pulse">
                  <Brain className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h4 className="font-bold">HealthGPT</h4>
                  <p className="text-xs text-green-400 flex items-center"><span className="w-2 h-2 rounded-full bg-green-400 mr-1"/> Online & Listening</p>
                </div>
              </div>
              <div className="bg-white/5 rounded-2xl p-4 mb-4 text-sm text-slate-300">
                "Welcome back. Your previous stress score was Moderate. Would you like to compare today's assessment with your last report?"
              </div>
              <div className="bg-sky-500/20 text-sky-100 rounded-2xl p-4 ml-8 text-sm">
                "Yes, please show me my weekly trend."
              </div>
            </div>
            {/* Decorative background blurs */}
            <div className="absolute -inset-4 bg-indigo-500/20 blur-2xl -z-10 rounded-full" />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 border-t border-slate-800 text-center text-slate-500">
        <div className="max-w-6xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center">
          <p>© 2026 HealthTech Intelligence SaaS. All rights reserved.</p>
          <div className="flex space-x-6 mt-4 md:mt-0">
            <span className="hover:text-slate-300 cursor-pointer transition-colors">Privacy</span>
            <span className="hover:text-slate-300 cursor-pointer transition-colors">Terms</span>
            <span className="hover:text-slate-300 cursor-pointer transition-colors">HIPAA Compliance</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
