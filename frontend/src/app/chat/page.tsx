"use client";

import { motion } from "framer-motion";
import { Mic, Send, Brain, Paperclip, ChevronLeft, MoreVertical, Volume2 } from "lucide-react";
import { useState, useEffect } from "react";
import Link from "next/link";
import { chatbotKnowledge } from "../../data/chatbotKnowledge";

export default function CompanionChat() {
  const [messages, setMessages] = useState([
    { role: "assistant", content: "Welcome! I am equipped with a medical knowledge base of over 25 comprehensive health topics. Ask me anything about your symptoms, diseases, or health metrics.", time: new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'}) }
  ]);
  const [input, setInput] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [isListening, setIsListening] = useState(false);

  const suggestedPrompts = [
    "Review my Action Plan",
    "Symptoms of Diabetes",
    "I'm feeling stressed",
    "Heart Health Tips"
  ];

  const handleSend = (text: string) => {
    if (!text.trim()) return;
    
    const now = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
    
    // Add user message
    setMessages(prev => [...prev, { role: "user", content: text, time: now }]);
    setInput("");
    setIsTyping(true);

    // AI Logic (Find best match from dataset)
    setTimeout(() => {
      let aiResponse = "I'm sorry, I don't have enough data on that specific query. Could you try asking about diabetes, heart health, sleep, or liver disease?";
      
      const lowerText = text.toLowerCase();
      // Search the dataset for matches
      const match = chatbotKnowledge.find(qa => 
        lowerText.includes(qa.question.toLowerCase().replace("?", "")) || 
        qa.question.toLowerCase().split(" ").some(word => word.length > 4 && lowerText.includes(word))
      );
      
      if (match) {
        aiResponse = match.answer;
      } else if (lowerText.includes("hello") || lowerText.includes("hi")) {
        aiResponse = "Hello! How can I assist you with your health today?";
      }

      setMessages(prev => [...prev, { 
        role: "assistant", 
        content: aiResponse, 
        time: now 
      }]);
      setIsTyping(false);
    }, 1500);
  };

  return (
    <div className="flex flex-col h-screen bg-[#0B1120] text-slate-50 font-sans relative overflow-hidden">
      {/* Dynamic Background Gradients */}
      <div className="absolute top-0 left-0 w-full h-[500px] bg-gradient-to-br from-indigo-500/20 via-purple-500/10 to-transparent blur-[120px] pointer-events-none" />
      
      {/* Header */}
      <header className="glass-panel rounded-none border-t-0 border-l-0 border-r-0 border-b border-white/10 p-4 flex items-center justify-between z-10 sticky top-0">
        <div className="flex items-center gap-4">
          <Link href="/" className="p-2 hover:bg-white/5 rounded-full transition-colors">
            <ChevronLeft className="w-6 h-6 text-slate-400" />
          </Link>
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
                <Brain className="w-5 h-5 text-white" />
              </div>
              <span className="absolute bottom-0 right-0 w-3 h-3 bg-green-400 border-2 border-[#0B1120] rounded-full"></span>
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight">HealthGPT</h1>
              <p className="text-xs text-green-400">Online • Context Aware</p>
            </div>
          </div>
        </div>
        <button className="p-2 hover:bg-white/5 rounded-full transition-colors">
          <MoreVertical className="w-6 h-6 text-slate-400" />
        </button>
      </header>

      {/* Chat Area */}
      <div className="flex-1 overflow-y-auto p-4 md:p-8 z-10 space-y-6 flex flex-col">
        {/* Memory Indicator */}
        <div className="flex justify-center mb-4">
          <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-full px-4 py-1 text-xs text-indigo-300 font-medium tracking-wide">
            Memory Synced: Profile & Assessment History Loaded
          </div>
        </div>

        {messages.map((msg, idx) => (
          <motion.div 
            key={idx}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex-shrink-0 flex items-center justify-center mr-3 mt-auto mb-1">
                <Brain className="w-4 h-4 text-white" />
              </div>
            )}
            
            <div className={`max-w-[85%] md:max-w-[70%] flex flex-col ${msg.role === 'user' ? 'items-end' : 'items-start'}`}>
              <div className={`p-4 rounded-2xl shadow-sm ${
                msg.role === 'user' 
                  ? 'bg-gradient-to-br from-sky-500 to-indigo-600 text-white rounded-br-sm' 
                  : 'glass-panel bg-white/5 border-white/10 text-slate-200 rounded-bl-sm'
              }`}>
                <p className="leading-relaxed text-sm md:text-base">{msg.content}</p>
                
                {/* Assistant Toolbar */}
                {msg.role === 'assistant' && (
                  <div className="mt-3 flex items-center gap-3 border-t border-white/10 pt-3">
                    <button className="text-slate-400 hover:text-white transition-colors"><Volume2 className="w-4 h-4" /></button>
                    <span className="text-xs text-slate-500 italic">Medical Info. Not diagnostic.</span>
                  </div>
                )}
              </div>
              <span className="text-xs text-slate-500 mt-1 px-1">{msg.time}</span>
            </div>
          </motion.div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <motion.div 
            initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="flex justify-start"
          >
            <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex-shrink-0 flex items-center justify-center mr-3 mt-auto mb-1">
              <Brain className="w-4 h-4 text-white" />
            </div>
            <div className="glass-panel p-4 rounded-2xl rounded-bl-sm flex items-center gap-1">
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
              <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
            </div>
          </motion.div>
        )}
      </div>

      {/* Input Area */}
      <div className="z-10 p-4 md:p-6 bg-gradient-to-t from-[#0B1120] via-[#0B1120] to-transparent">
        {/* Suggested Prompts */}
        <div className="flex overflow-x-auto gap-2 pb-4 scrollbar-hide mb-2 px-2">
          {suggestedPrompts.map((prompt, i) => (
            <button 
              key={i} 
              onClick={() => handleSend(prompt)}
              className="whitespace-nowrap px-4 py-2 rounded-full border border-white/10 bg-white/5 text-xs text-slate-300 hover:bg-white/10 hover:text-white transition-all flex-shrink-0"
            >
              {prompt}
            </button>
          ))}
        </div>

        <div className="max-w-4xl mx-auto relative flex items-end gap-2 bg-white/5 border border-white/10 rounded-3xl p-2 backdrop-blur-md focus-within:border-indigo-500/50 focus-within:bg-white/10 transition-all">
          <button className="p-3 text-slate-400 hover:text-white transition-colors rounded-full hover:bg-white/5">
            <Paperclip className="w-5 h-5" />
          </button>
          
          <textarea 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => { if(e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(input); } }}
            placeholder="Ask about your health, diseases, or action plans..."
            className="w-full bg-transparent border-none focus:outline-none text-slate-50 resize-none max-h-32 min-h-[44px] py-3 text-sm md:text-base scrollbar-hide"
            rows={1}
          />

          {/* Voice Wave Animation when listening */}
          {isListening && (
            <div className="absolute right-16 top-1/2 -translate-y-1/2 flex items-center gap-1">
               <motion.div animate={{ height: [10, 20, 10] }} transition={{ repeat: Infinity, duration: 0.6 }} className="w-1 bg-red-400 rounded-full" />
               <motion.div animate={{ height: [10, 30, 10] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.1 }} className="w-1 bg-red-400 rounded-full" />
               <motion.div animate={{ height: [10, 15, 10] }} transition={{ repeat: Infinity, duration: 0.6, delay: 0.2 }} className="w-1 bg-red-400 rounded-full" />
            </div>
          )}

          <button 
            onClick={() => setIsListening(!isListening)}
            className={`p-3 transition-all rounded-full ${isListening ? 'bg-red-500/20 text-red-400' : 'text-slate-400 hover:text-white hover:bg-white/5'}`}
          >
            <Mic className="w-5 h-5" />
          </button>

          <button 
            onClick={() => handleSend(input)}
            className={`p-3 rounded-full transition-all flex items-center justify-center ${input.trim() ? 'bg-indigo-500 text-white hover:bg-indigo-600 shadow-lg shadow-indigo-500/25' : 'bg-white/5 text-slate-500 cursor-not-allowed'}`}
          >
            <Send className="w-5 h-5" />
          </button>
        </div>
        <p className="text-center text-[10px] text-slate-500 mt-3">HealthGPT uses CDC/WHO data for education. It does not replace professional medical advice.</p>
      </div>
    </div>
  );
}
