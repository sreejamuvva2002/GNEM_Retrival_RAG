"use client"

import dynamic from "next/dynamic"
import type { Source } from "@/app/page"

const MapContainer = dynamic(
  () => import("react-leaflet").then((mod) => mod.MapContainer),
  { ssr: false }
)
const TileLayer = dynamic(
  () => import("react-leaflet").then((mod) => mod.TileLayer),
  { ssr: false }
)
const Marker = dynamic(
  () => import("react-leaflet").then((mod) => mod.Marker),
  { ssr: false }
)
const Popup = dynamic(
  () => import("react-leaflet").then((mod) => mod.Popup),
  { ssr: false }
)

interface MapPanelProps {
  sources: Source[]
}

export default function MapPanel({ sources }: MapPanelProps) {
  // Center on Georgia
  const georgiaCenter: [number, number] = [33.2, -84.3]

  return (
    <div className="h-full w-full relative">
      <link
        rel="stylesheet"
        href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        integrity="sha256-p4NxAoJBhIIN+hmNHrzRCf9tD/miZyoHS5obTRR9BMY="
        crossOrigin=""
      />
      <MapContainer
        center={georgiaCenter}
        zoom={7}
        style={{ height: "100%", width: "100%" }}
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {sources.map((source) => (
          <Marker key={source.id} position={[source.latitude, source.longitude]}>
            <Popup>
              <div className="text-sm">
                <p className="font-semibold">{source.company}</p>
                <p className="text-muted-foreground">{source.address}</p>
                <p className="text-muted-foreground">{source.productService}</p>
              </div>
            </Popup>
          </Marker>
        ))}
      </MapContainer>
      {sources.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50 pointer-events-none">
          <p className="text-muted-foreground text-sm">
            Ask a question to see company locations on the map
          </p>
        </div>
      )}
    </div>
  )
}
