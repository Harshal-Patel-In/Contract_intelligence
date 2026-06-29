import React, { useState, useRef, useEffect } from 'react';
import { Send, X, Bot, Loader2 } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { API_BASE_URL } from '../config';

interface ChatMessage {
  role: 'user' | 'ai';
  text: string;
}

interface ChatBoxProps {
  jobId: string;
  onClose: () => void;
}

export const ChatBox: React.FC<ChatBoxProps> = ({ jobId, onClose }) => {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: 'ai', text: "Hello! I've analyzed your contract. What would you like to ask me about it?" }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;
    
    const userMsg = input.trim();
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setInput('');
    setIsLoading(true);

    // Add empty AI message placeholder
    setMessages(prev => [...prev, { role: 'ai', text: '' }]);

    // Extract history, excluding the first generic greeting. Limit to last 6 messages to save context space.
    const historyPayload = messages.slice(1).map(m => ({ role: m.role, text: m.text })).slice(-6);

    try {
      const res = await fetch(`${API_BASE_URL}/analyze/chat/${jobId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, history: historyPayload })
      });

      if (!res.ok) throw new Error('Failed to fetch from chat endpoint');
      if (!res.body) throw new Error('No stream in response');

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        
        let splitIndex;
        while ((splitIndex = buffer.indexOf('\n\n')) >= 0) {
          const chunkStr = buffer.slice(0, splitIndex);
          buffer = buffer.slice(splitIndex + 2);
          
          if (chunkStr.startsWith('data: ')) {
            try {
              const data = JSON.parse(chunkStr.substring(6));
              
              if (data.status === 'chunk' && data.text) {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const last = { ...newMsgs[newMsgs.length - 1] };
                  if (last.role === 'ai') {
                    last.text += data.text;
                  }
                  newMsgs[newMsgs.length - 1] = last;
                  return newMsgs;
                });
              } else if (data.status === 'error') {
                setMessages(prev => {
                  const newMsgs = [...prev];
                  const last = { ...newMsgs[newMsgs.length - 1] };
                  if (last.role === 'ai') last.text += "\n\n[Error: " + data.message + "]";
                  newMsgs[newMsgs.length - 1] = last;
                  return newMsgs;
                });
              }
            } catch (e) {
              console.error("JSON parse error for SSE chunk:", e);
            }
          }
        }
      }
    } catch (e: any) {
      console.error("Chat streaming error:", e);
      setMessages(prev => {
        const newMsgs = [...prev];
        const last = newMsgs[newMsgs.length - 1];
        if (last.role === 'ai' && !last.text) {
            last.text = "Sorry, I encountered an error connecting to the AI. " + e.message;
        }
        return newMsgs;
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="chat-container glass-card">
      <div className="chat-header">
        <div className="chat-header-left">
          <Bot size={20} className="chat-bot-icon" />
          <h3>Chat with Contract</h3>
        </div>
        <button className="icon-btn" onClick={onClose}>
          <X size={18} />
        </button>
      </div>

      <div className="chat-messages">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble-wrapper ${msg.role}`}>
            <div className={`chat-bubble chat-bubble-${msg.role}`}>
              {msg.text ? (
                <div className="markdown-body">
                  <ReactMarkdown>{msg.text}</ReactMarkdown>
                </div>
              ) : (
                isLoading && i === messages.length - 1 ? <span className="typing-dot">...</span> : ''
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask a question..." 
          className="chat-input"
          disabled={isLoading}
        />
        <button 
          className="chat-send-btn" 
          onClick={handleSend}
          disabled={!input.trim() || isLoading}
        >
          {isLoading ? <Loader2 size={18} className="spin" /> : <Send size={18} />}
        </button>
      </div>
    </div>
  );
};
