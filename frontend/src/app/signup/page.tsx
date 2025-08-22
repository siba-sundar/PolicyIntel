"use client"
import { useRouter } from "next/navigation"

export default function SignupPage() {
  const router = useRouter()

  const handleSignup = () => {
    router.push("/login")
  }

  return (
    <div className="p-6 flex flex-col gap-4">
      <h1 className="text-xl font-bold">Signup</h1>
      <input placeholder="Email" className="border p-2" />
      <input placeholder="Password" type="password" className="border p-2" />
      <button onClick={handleSignup} className="bg-green-600 text-white p-2 rounded">
        Signup
      </button>
    </div>
  )
}
