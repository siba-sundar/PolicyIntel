"use client"
import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"

export default function AdminPage() {
  const [file, setFile] = useState<File | null>(null)
  const router = useRouter()

  useEffect(() => {
    const role = localStorage.getItem("role")
    if (role !== "admin") router.push("/login")
  }, [router])

  const handleUpload = () => {
    if (file) alert(`File uploaded: ${file.name}`)
  }

  return (
    <div className="p-6 flex flex-col gap-4">
      <h1 className="text-xl font-bold">Admin Dashboard</h1>
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleUpload} className="bg-purple-600 text-white p-2 rounded">
        Upload
      </button>
    </div>
  )
}
