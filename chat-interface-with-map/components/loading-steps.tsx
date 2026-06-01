"use client"

import { useEffect, useState } from "react"
import { Search, Database, ListFilter, Sparkles } from "lucide-react"

export type LoadingStep = {
  id: number
  label: string
  status: "pending" | "active" | "complete"
}

interface LoadingStepsProps {
  steps: LoadingStep[]
}

const stepIcons = [Search, Database, ListFilter, Sparkles]
const stepDescriptions = [
  "Scanning knowledge base...",
  "Fetching relevant documents...",
  "Ranking by relevance...",
  "Composing response...",
]

export default function LoadingSteps({ steps }: LoadingStepsProps) {
  const [fadeState, setFadeState] = useState<"in" | "out">("in")
  const [displayedStep, setDisplayedStep] = useState<number>(0)
  
  // Find the current active step index
  const activeStepIndex = steps.findIndex((s) => s.status === "active")
  const completedCount = steps.filter((s) => s.status === "complete").length

  useEffect(() => {
    if (activeStepIndex !== -1 && activeStepIndex !== displayedStep) {
      // Fade out current step
      setFadeState("out")
      
      // After fade out, switch to new step and fade in
      const timeout = setTimeout(() => {
        setDisplayedStep(activeStepIndex)
        setFadeState("in")
      }, 300)
      
      return () => clearTimeout(timeout)
    }
  }, [activeStepIndex, displayedStep])

  const currentStepIndex = displayedStep
  const currentStep = steps[currentStepIndex]
  const Icon = stepIcons[currentStepIndex]

  if (!currentStep) return null

  return (
    <div className="flex justify-start">
      <div className="bg-white border border-slate-200 rounded-2xl shadow-lg shadow-slate-200/50 overflow-hidden w-full max-w-sm">
        {/* Single step display with fade animation */}
        <div className="p-5">
          <div
            className={`
              flex items-center gap-4 transition-all duration-300 ease-in-out
              ${fadeState === "in" ? "opacity-100 translate-y-0" : "opacity-0 translate-y-2"}
            `}
          >
            {/* Animated icon */}
            <div className="relative flex-shrink-0">
              <div className="flex items-center justify-center w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-indigo-600 shadow-lg shadow-blue-200">
                <Icon className="w-5 h-5 text-white" />
              </div>
              {/* Pulse ring */}
              <div className="absolute -inset-1 rounded-xl border-2 border-blue-300 animate-pulse opacity-60" />
            </div>

            {/* Content */}
            <div className="flex-1">
              <p className="text-sm font-semibold text-slate-800">
                {currentStep.label}
              </p>
              <p className="text-xs text-slate-500 mt-0.5">
                {stepDescriptions[currentStepIndex]}
              </p>
            </div>

            {/* Loading dots */}
            <div className="flex gap-1 items-center">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"
                  style={{ animationDelay: `${i * 150}ms` }}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Progress indicator */}
        <div className="px-5 pb-4">
          {/* Step dots */}
          <div className="flex items-center justify-center gap-2 mb-3">
            {steps.map((step, index) => (
              <div
                key={step.id}
                className={`
                  w-2 h-2 rounded-full transition-all duration-300
                  ${index < completedCount ? "bg-emerald-500" : ""}
                  ${index === currentStepIndex ? "bg-blue-500 w-6" : ""}
                  ${index > currentStepIndex ? "bg-slate-200" : ""}
                `}
              />
            ))}
          </div>
          
          {/* Progress bar */}
          <div className="h-1 bg-slate-100 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500 ease-out bg-gradient-to-r from-blue-500 to-indigo-500"
              style={{ width: `${((completedCount + 0.5) / steps.length) * 100}%` }}
            />
          </div>
          <p className="text-[11px] text-slate-400 text-center mt-2">
            Step {currentStepIndex + 1} of {steps.length}
          </p>
        </div>
      </div>
    </div>
  )
}
