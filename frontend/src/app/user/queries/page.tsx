"use client"
import { useEffect } from "react"
import { useRouter } from "next/navigation"

export default function UserQueriesPage() {
  const router = useRouter()

  useEffect(() => {
    const role = localStorage.getItem("role")
    if (role !== "user") router.push("/login")
  }, [router])

  return (
    <div className="p-6">
      <h1 className="text-xl font-bold">User Queries</h1>
      <textarea placeholder="Ask your question..." className="border w-full p-2 mt-4" />
      <button className="bg-blue-600 text-white p-2 rounded mt-2">Submit</button>
    </div>
  )
}
