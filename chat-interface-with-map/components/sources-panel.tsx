"use client"

import { useState } from "react"
import { ChevronRight, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import type { Source } from "@/app/page"

interface SourcesPanelProps {
  sources: Source[]
  onClose: () => void
}

export default function SourcesPanel({ sources, onClose }: SourcesPanelProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const toggleExpand = (id: string) => {
    setExpandedId(expandedId === id ? null : id)
  }

  return (
    <div className="h-full flex flex-col bg-muted/30">
      {/* Header */}
      <div className="p-4 border-b border-border">
        <h2 className="font-bold text-foreground text-lg">Sources</h2>
        <p className="text-sm text-muted-foreground">{sources.length} sources found</p>
      </div>

      {/* Close panel button */}
      <div className="px-4 pt-3">
        <Button
          variant="outline"
          className="w-full justify-center gap-2"
          onClick={onClose}
        >
          <X className="h-4 w-4" />
          Close panel
        </Button>
      </div>

      {/* Sources list */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {sources.map((source) => (
          <div key={source.id} className="border border-border rounded-lg bg-background">
            <button
              onClick={() => toggleExpand(source.id)}
              className="w-full flex items-center gap-3 p-4 text-left hover:bg-muted/50 transition-colors"
            >
              <ChevronRight
                className={`h-5 w-5 text-muted-foreground transition-transform ${
                  expandedId === source.id ? "rotate-90" : ""
                }`}
              />
              <span className="font-medium text-foreground">{source.company}</span>
            </button>

            {expandedId === source.id && (
              <div className="px-4 pb-4 pl-12 space-y-2 text-sm">
                <div className="grid grid-cols-2 gap-x-4 gap-y-2">
                  <div>
                    <span className="text-muted-foreground">Record ID:</span>
                    <p className="text-foreground">{source.id}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Category:</span>
                    <p className="text-foreground">{source.category}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Industry Group:</span>
                    <p className="text-foreground">{source.industryGroup}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Location:</span>
                    <p className="text-foreground">{source.location}</p>
                  </div>
                  <div className="col-span-2">
                    <span className="text-muted-foreground">Address:</span>
                    <p className="text-foreground">{source.address}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Latitude:</span>
                    <p className="text-foreground">{source.latitude}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Longitude:</span>
                    <p className="text-foreground">{source.longitude}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Facility Type:</span>
                    <p className="text-foreground">{source.facilityType}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">EV Supply Chain Role:</span>
                    <p className="text-foreground">{source.evRole}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Primary OEMs:</span>
                    <p className="text-foreground">{source.primaryOEMs}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Supplier Type:</span>
                    <p className="text-foreground">{source.supplierType}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Employment:</span>
                    <p className="text-foreground">{source.employment}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Product/Service:</span>
                    <p className="text-foreground">{source.productService}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">EV/Battery Relevant:</span>
                    <p className="text-foreground">{source.evRelevant}</p>
                  </div>
                  <div>
                    <span className="text-muted-foreground">Classification:</span>
                    <p className="text-foreground">{source.classificationMethod}</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
