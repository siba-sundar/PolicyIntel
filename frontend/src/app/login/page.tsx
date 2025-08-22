"use client"
import { useRouter } from "next/navigation"
import { useState } from "react"

export default function LoginPage() {
  const [role, setRole] = useState("user")
  const router = useRouter()

  const handleLogin = () => {
    localStorage.setItem("role", role)
    if (role === "admin") {
      router.push("/admin")
    } else {
      router.push("/user/queries")
    }
  }

  return (
    <div className="p-6 flex flex-col gap-4">
      <h1 className="text-xl font-bold">Login</h1>
      <select value={role} onChange={(e) => setRole(e.target.value)} className="border p-2">
        <option value="user">User</option>
        <option value="admin">Admin</option>
      </select>
      <button onClick={handleLogin} className="bg-blue-600 text-white p-2 rounded">
        Login
      </button>
    </div>
  )
}
