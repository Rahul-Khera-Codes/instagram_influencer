"use client"

import { LoginForm } from "./Login"
import { RegistrationForm } from "./Registration"
import { useState } from "react"
import { Button } from "@/components/ui/button"

export default function Home() {
  const [showLogin, setShowLogin] = useState(true)

  return (
    <main className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="flex justify-center mb-6">
          <div className="flex bg-muted rounded-lg p-1">
            <Button
              variant={showLogin ? "default" : "ghost"}
              size="sm"
              onClick={() => setShowLogin(true)}
              className="rounded-md"
            >
              Login
            </Button>
            <Button
              variant={!showLogin ? "default" : "ghost"}
              size="sm"
              onClick={() => setShowLogin(false)}
              className="rounded-md"
            >
              Register
            </Button>
          </div>
        </div>

        {showLogin ? <LoginForm /> : <RegistrationForm />}
      </div>
    </main>
  )
}
