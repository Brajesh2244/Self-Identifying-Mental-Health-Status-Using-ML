import { useState, useEffect } from 'react';

export function useVoiceAI() {
  const [isListening, setIsListening] = useState(false);
  const [language, setLanguage] = useState<'en-US' | 'hi-IN' | 'kn-IN'>('en-US');

  // Simulated Web Speech API integration
  // In a full production environment, this would initialize window.SpeechRecognition
  // and window.speechSynthesis to provide Text-To-Speech and Speech-To-Text.
  
  const startListening = () => {
    setIsListening(true);
    console.log(`Voice AI: Started listening in ${language}...`);
    // Mock auto-stop after 3 seconds for UI demo purposes
    setTimeout(() => {
      setIsListening(false);
      console.log("Voice AI: Stopped listening.");
    }, 3000);
  };

  const stopListening = () => {
    setIsListening(false);
  };

  const speak = (text: string) => {
    console.log(`Voice AI Speaking [${language}]: ${text}`);
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = language;
      window.speechSynthesis.speak(utterance);
    }
  };

  return {
    isListening,
    language,
    setLanguage,
    startListening,
    stopListening,
    speak
  };
}
