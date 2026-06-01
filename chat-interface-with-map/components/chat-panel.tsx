"use client"

import { useState } from "react"
import { Send, MessageSquare } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import LoadingSteps, { type LoadingStep } from "@/components/loading-steps"
import type { Message, Source } from "@/app/page"

interface ChatPanelProps {
  messages: Message[]
  onSendMessage: (content: string) => void
  onShowSources: (sources: Source[]) => void
  isLoading: boolean
  loadingSteps: LoadingStep[]
}

export default function ChatPanel({ messages, onSendMessage, onShowSources, isLoading, loadingSteps }: ChatPanelProps) {
  const [input, setInput] = useState("")

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (input.trim()) {
      onSendMessage(input.trim())
      setInput("")
    }
  }

  return (
    <div className="relative flex flex-col h-full overflow-hidden">
      {/* Background layers */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-100 via-slate-50 to-white" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-blue-50/40 via-transparent to-transparent" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_bottom_left,_var(--tw-gradient-stops))] from-indigo-50/30 via-transparent to-transparent" />
      
      {/* Dot pattern */}
      <div 
        className="absolute inset-0 opacity-[0.4]"
        style={{
          backgroundImage: `radial-gradient(circle at 1px 1px, rgb(203 213 225 / 0.4) 1px, transparent 0)`,
          backgroundSize: '24px 24px',
        }}
      />

      {/* Header */}
      <div className="relative z-10 px-6 py-4 border-b border-slate-200/60 bg-white/80 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <div className="relative group cursor-default">
            <div className="bg-gradient-to-br from-slate-800 to-slate-900 text-white font-bold text-sm px-3.5 py-2 rounded-lg tracking-wide shadow-md shadow-slate-300/50">
              GNEM
            </div>
            <div className="absolute left-0 top-full mt-3 px-4 py-2.5 bg-slate-900 text-white text-xs rounded-lg shadow-xl opacity-0 group-hover:opacity-100 transition-all duration-200 whitespace-nowrap z-50 pointer-events-none">
              Georgia Network for Electric Mobility
              <div className="absolute -top-1.5 left-5 w-3 h-3 bg-slate-900 rotate-45 rounded-sm" />
            </div>
          </div>
          <div>
            <h1 className="text-base font-semibold text-slate-800">Chat Assistant</h1>
            <p className="text-xs text-slate-500">Georgia EV Supply Chain Intelligence</p>
          </div>
        </div>
      </div>

      {/* Messages area */}
      <div className="relative z-10 flex-1 overflow-y-auto px-6 py-6">
        <div className="max-w-2xl mx-auto space-y-5">
          {messages.length === 0 && (
            <div className="text-center py-20 animate-fade-in">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-slate-100 to-slate-200 shadow-inner mb-5">
                <MessageSquare className="w-7 h-7 text-slate-400" />
              </div>
              <h2 className="text-xl font-semibold text-slate-700 mb-2">Start a conversation</h2>
              <p className="text-sm text-slate-500 max-w-sm mx-auto leading-relaxed">
                Ask me about companies in Georgia&apos;s automotive and EV supply chain.
              </p>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div
              key={message.id}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} animate-fade-in-up`}
              style={{ animationDelay: `${index * 50}ms` }}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-5 py-4 ${
                  message.role === "user"
                    ? "bg-gradient-to-br from-slate-800 to-slate-900 text-white shadow-lg shadow-slate-300/30"
                    : "bg-white text-slate-700 border border-slate-200/80 shadow-md shadow-slate-200/50"
                }`}
              >
                <div className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</div>
                {message.role === "assistant" && message.sources && message.sources.length > 0 && (
                  <Button
                    variant="outline"
                    size="sm"
                    className="mt-4 h-9 text-xs bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100 hover:text-slate-900 hover:border-slate-300 transition-all"
                    onClick={() => onShowSources(message.sources!)}
                  >
                    <svg className="w-3.5 h-3.5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                    </svg>
                    View Sources ({message.sources.length})
                  </Button>
                )}
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="animate-fade-in-up">
              <LoadingSteps steps={loadingSteps} />
            </div>
          )}
        </div>
      </div>

      {/* Input area */}
      <div className="relative z-10 px-6 py-5 border-t border-slate-200/60 bg-white/80 backdrop-blur-md">
        <form onSubmit={handleSubmit} className="max-w-2xl mx-auto">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about EV companies in Georgia..."
                className="h-12 bg-white border-slate-200 focus:border-slate-400 focus:ring-2 focus:ring-slate-200 rounded-xl pl-4 pr-4 text-sm shadow-sm"
              />
            </div>
            <Button 
              type="submit" 
              size="icon" 
              disabled={isLoading || !input.trim()}
              className="h-12 w-12 rounded-xl bg-gradient-to-br from-slate-800 to-slate-900 hover:from-slate-700 hover:to-slate-800 disabled:opacity-50 disabled:cursor-not-allowed shadow-md shadow-slate-300/50 transition-all"
            >
              <Send className="h-4 w-4" />
            </Button>
          </div>
        </form>
      </div>
    </div>
  )
}
