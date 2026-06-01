"use client"

import { useState, useRef, useCallback, useEffect } from "react"
import ChatPanel from "@/components/chat-panel"
import MapPanel from "@/components/map-panel"
import SourcesPanel from "@/components/sources-panel"
import type { LoadingStep } from "@/components/loading-steps"

export interface Source {
  id: string
  company: string
  category: string
  industryGroup: string
  location: string
  address: string
  latitude: number
  longitude: number
  facilityType: string
  evRole: string
  primaryOEMs: string
  supplierType: string
  employment: number
  productService: string
  evRelevant: string
  classificationMethod: string
}

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: Source[]
}

const mockSources: Source[] = [
  {
    id: "KB_ROW_0086_e2388e6d634f",
    company: "IMS Gear Georgia Inc.",
    category: "Tier 1/2",
    industryGroup: "Electronic and Other Electrical Equipment and Components",
    location: "Suwanee, Gwinnett County",
    address: "300 Satellite Blvd NE, Suwanee, GA 30024",
    latitude: 34.035115,
    longitude: -84.064265,
    facilityType: "Manufacturing Plant",
    evRole: "General Automotive",
    primaryOEMs: "Multiple OEMs",
    supplierType: "Automotive supply chain participant",
    employment: 100,
    productService: "Car audio systems",
    evRelevant: "No",
    classificationMethod: "Supplier",
  },
  {
    id: "KB_ROW_0087_f3489f7e745g",
    company: "Hyundai Transys Georgia Seating Systems",
    category: "Tier 1",
    industryGroup: "Motor Vehicle Parts and Accessories",
    location: "West Point, Troup County",
    address: "550 Kia Pkwy, West Point, GA 31833",
    latitude: 32.877,
    longitude: -85.183,
    facilityType: "Manufacturing Plant",
    evRole: "EV Components",
    primaryOEMs: "Hyundai/Kia",
    supplierType: "OEM subsidiary",
    employment: 500,
    productService: "Automotive seating systems",
    evRelevant: "Yes",
    classificationMethod: "Tier 1",
  },
  {
    id: "KB_ROW_0088_g4590h8f856h",
    company: "Hyundai Transys Georgia Powertrain",
    category: "Tier 1",
    industryGroup: "Motor Vehicle Parts and Accessories",
    location: "West Point, Troup County",
    address: "450 Kia Pkwy, West Point, GA 31833",
    latitude: 32.875,
    longitude: -85.185,
    facilityType: "Manufacturing Plant",
    evRole: "EV Powertrain",
    primaryOEMs: "Hyundai/Kia",
    supplierType: "OEM subsidiary",
    employment: 300,
    productService: "Powertrain components",
    evRelevant: "Yes",
    classificationMethod: "Tier 1",
  },
  {
    id: "KB_ROW_0089_h5601i9g967i",
    company: "Novelis Inc.",
    category: "Tier 2",
    industryGroup: "Primary Metal Industries",
    location: "Atlanta, Fulton County",
    address: "3560 Lenox Rd NE, Atlanta, GA 30326",
    latitude: 33.8486,
    longitude: -84.3625,
    facilityType: "Corporate HQ",
    evRole: "EV Materials",
    primaryOEMs: "Multiple OEMs",
    supplierType: "Materials supplier",
    employment: 1000,
    productService: "Aluminum rolled products",
    evRelevant: "Yes",
    classificationMethod: "Tier 2",
  },
  {
    id: "KB_ROW_0090_i6712j0h078j",
    company: "Joon Georgia, Inc.",
    category: "Tier 1",
    industryGroup: "Motor Vehicle Parts and Accessories",
    location: "LaGrange, Troup County",
    address: "100 Lower Big Springs Rd, LaGrange, GA 30241",
    latitude: 33.0362,
    longitude: -85.0322,
    facilityType: "Manufacturing Plant",
    evRole: "General Automotive",
    primaryOEMs: "Kia",
    supplierType: "Automotive supply chain participant",
    employment: 200,
    productService: "Automotive stampings",
    evRelevant: "No",
    classificationMethod: "Tier 1",
  },
]

const initialLoadingSteps: LoadingStep[] = [
  { id: 1, label: "Searching KB", status: "pending" },
  { id: 2, label: "Retrieval", status: "pending" },
  { id: 3, label: "Reranking", status: "pending" },
  { id: 4, label: "Final answer generation", status: "pending" },
]

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([])
  const [showSources, setShowSources] = useState(false)
  const [currentSources, setCurrentSources] = useState<Source[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [loadingSteps, setLoadingSteps] = useState<LoadingStep[]>(initialLoadingSteps)
  const [chatWidth, setChatWidth] = useState(50) // percentage
  const isDragging = useRef(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const handleMouseDown = useCallback(() => {
    isDragging.current = true
    document.body.style.cursor = "col-resize"
    document.body.style.userSelect = "none"
  }, [])

  const handleMouseUp = useCallback(() => {
    isDragging.current = false
    document.body.style.cursor = ""
    document.body.style.userSelect = ""
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isDragging.current || !containerRef.current) return
    
    const containerRect = containerRef.current.getBoundingClientRect()
    const newWidth = ((e.clientX - containerRect.left) / containerRect.width) * 100
    
    // Limit the width between 20% and 80%
    if (newWidth >= 20 && newWidth <= 80) {
      setChatWidth(newWidth)
    }
  }, [])

  // Add and remove event listeners
  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => handleMouseMove(e)
    const onMouseUp = () => handleMouseUp()
    
    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
    
    return () => {
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
    }
  }, [handleMouseMove, handleMouseUp])
  const handleSendMessage = (content: string) => {
    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content,
    }

    setMessages((prev) => [...prev, userMessage])
    setIsLoading(true)
    setLoadingSteps(initialLoadingSteps.map((step, index) => 
      index === 0 ? { ...step, status: "active" } : step
    ))

    // Step 1: Searching KB (after 800ms)
    setTimeout(() => {
      setLoadingSteps((prev) =>
        prev.map((step) =>
          step.id === 1
            ? { ...step, status: "complete" }
            : step.id === 2
            ? { ...step, status: "active" }
            : step
        )
      )
    }, 800)

    // Step 2: Retrieval (after 1600ms)
    setTimeout(() => {
      setLoadingSteps((prev) =>
        prev.map((step) =>
          step.id === 2
            ? { ...step, status: "complete" }
            : step.id === 3
            ? { ...step, status: "active" }
            : step
        )
      )
    }, 1600)

    // Step 3: Reranking (after 2400ms)
    setTimeout(() => {
      setLoadingSteps((prev) =>
        prev.map((step) =>
          step.id === 3
            ? { ...step, status: "complete" }
            : step.id === 4
            ? { ...step, status: "active" }
            : step
        )
      )
    }, 2400)

    // Step 4: Final answer generation and response (after 3200ms)
    setTimeout(() => {
      setLoadingSteps((prev) =>
        prev.map((step) =>
          step.id === 4 ? { ...step, status: "complete" } : step
        )
      )

      // Small delay before showing the response
      setTimeout(() => {
        const assistantMessage: Message = {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: `Based on your query about "${content}", I found several companies in Georgia's automotive and EV supply chain. Here are some relevant facilities:\n\n1. **IMS Gear Georgia Inc.** - Manufacturing car audio systems in Suwanee\n2. **Hyundai Transys Georgia Seating Systems** - Producing automotive seating in West Point\n3. **Hyundai Transys Georgia Powertrain** - Manufacturing powertrain components\n4. **Novelis Inc.** - Supplying aluminum rolled products from Atlanta\n5. **Joon Georgia, Inc.** - Producing automotive stampings in LaGrange\n\nThese companies represent various tiers of the automotive supply chain, with several being directly involved in EV component manufacturing.`,
          sources: mockSources,
        }

        setMessages((prev) => [...prev, assistantMessage])
        setCurrentSources(mockSources)
        setIsLoading(false)
        setLoadingSteps(initialLoadingSteps)
      }, 300)
    }, 3200)
  }

  const handleShowSources = (sources: Source[]) => {
    setCurrentSources(sources)
    setShowSources(true)
  }

  const handleCloseSources = () => {
    setShowSources(false)
  }

  return (
    <div ref={containerRef} className="flex h-screen bg-background">
      {/* Left side - Chat Panel */}
      <div style={{ width: `${chatWidth}%` }} className="flex-shrink-0">
        <ChatPanel
          messages={messages}
          onSendMessage={handleSendMessage}
          onShowSources={handleShowSources}
          isLoading={isLoading}
          loadingSteps={loadingSteps}
        />
      </div>

      {/* Draggable Divider */}
      <div
        onMouseDown={handleMouseDown}
        className="w-1 bg-border hover:bg-primary/50 cursor-col-resize flex-shrink-0 transition-colors relative group"
      >
        <div className="absolute inset-y-0 -left-1 -right-1 group-hover:bg-primary/10" />
      </div>

      {/* Right side - Map and Sources */}
      <div style={{ width: `${100 - chatWidth}%` }} className="flex flex-col flex-shrink-0">
        <div className={showSources ? "h-1/2" : "h-full"}>
          <MapPanel sources={currentSources} />
        </div>
        {showSources && (
          <div className="h-1/2 border-t border-border">
            <SourcesPanel sources={currentSources} onClose={handleCloseSources} />
          </div>
        )}
      </div>
    </div>
  )
}
