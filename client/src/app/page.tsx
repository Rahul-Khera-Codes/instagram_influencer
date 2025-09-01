import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Search, Users, TrendingUp, Zap, Shield, BarChart3 } from "lucide-react"
import Link from "next/link"


export default function HomePage() {
  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-accent-foreground" />
              </div>
              <span className="text-xl font-bold text-foreground">InfluenceHub</span>
            </div>

            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-muted-foreground hover:text-foreground transition-colors">
                Features
              </a>
              <a href="#pricing" className="text-muted-foreground hover:text-foreground transition-colors">
                Pricing
              </a>
              <a href="#about" className="text-muted-foreground hover:text-foreground transition-colors">
                About
              </a>
              <Link href="/auth">
                <Button variant="outline" size="sm">
                  Sign In
                </Button>
              </Link>
              <Link href="/auth">
              <Button size="sm" className="bg-accent hover:bg-accent/90">
                Get Started
              </Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto text-center">
          <div className="max-w-3xl mx-auto">
            <Badge variant="secondary" className="mb-4">
              AI-Powered Influencer Marketing
            </Badge>
            <h1 className="text-4xl md:text-6xl font-bold text-foreground mb-6 leading-tight">
              Find the Perfect Influencers with <span className="text-accent">AI Precision</span>
            </h1>
            <p className="text-xl text-muted-foreground mb-8 leading-relaxed">
              Connect with authentic influencers using our AI-powered platform. Get transparent pricing, performance
              predictions, and explainable matching scores to maximize your campaign ROI.
            </p>

            {/* Search Bar */}
            {/* <div className="max-w-2xl mx-auto mb-8">
              <div className="relative">
                <Search className="absolute left-4 top-1/2 transform -translate-y-1/2 text-muted-foreground w-5 h-5" />
                <Input
                  placeholder="Search for influencers by category, location, or niche..."
                  className="pl-12 pr-4 py-4 text-lg border-2 border-border focus:border-accent shadow-lg"
                />
                <Link href="/search">
                  <Button className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-accent hover:bg-accent/90">
                    Search
                  </Button>
                </Link>
              </div>
            </div> */}

            {/* <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/campaign/create">
                <Button size="lg" className="bg-accent hover:bg-accent/90 text-lg px-8">
                  Create Campaign
                </Button>
              </Link>
              <Button variant="outline" size="lg" className="text-lg px-8 bg-transparent">
                Watch Demo
              </Button>
            </div> */}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 sm:px-6 lg:px-8 bg-muted/30">
        <div className="max-w-7xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-4">Why Choose InfluenceHub?</h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Our AI-powered platform gives you the competitive edge with transparent pricing, performance predictions,
              and explainable matching.
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="border-2 hover:border-accent/50 transition-colors">
              <CardHeader>
                <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4">
                  <Shield className="w-6 h-6 text-accent" />
                </div>
                <CardTitle>AI-Powered Pricing Fairness</CardTitle>
                <CardDescription>
                  Get instant insights on whether influencer quotes are reasonable based on market data and performance
                  metrics.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-2 hover:border-accent/50 transition-colors">
              <CardHeader>
                <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4">
                  <TrendingUp className="w-6 h-6 text-accent" />
                </div>
                <CardTitle>Performance Prediction</CardTitle>
                <CardDescription>
                  Forecast campaign results before launch with our advanced AI models trained on thousands of successful
                  campaigns.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-2 hover:border-accent/50 transition-colors">
              <CardHeader>
                <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4">
                  <Users className="w-6 h-6 text-accent" />
                </div>
                <CardTitle>Explainable Matching</CardTitle>
                <CardDescription>
                  Understand exactly why each influencer is recommended with detailed compatibility scores and
                  reasoning.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-2 hover:border-accent/50 transition-colors">
              <CardHeader>
                <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4">
                  <BarChart3 className="w-6 h-6 text-accent" />
                </div>
                <CardTitle>Cost Transparency</CardTitle>
                <CardDescription>
                  Clear CPM/CPA predictions with detailed breakdowns so you know exactly what you're paying for.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-2 hover:border-accent/50 transition-colors">
              <CardHeader>
                <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4">
                  <Search className="w-6 h-6 text-accent" />
                </div>
                <CardTitle>Advanced Search</CardTitle>
                <CardDescription>
                  Find influencers by category, location, engagement rate, audience demographics, and more with powerful
                  filters.
                </CardDescription>
              </CardHeader>
            </Card>

            <Card className="border-2 hover:border-accent/50 transition-colors">
              <CardHeader>
                <div className="w-12 h-12 bg-accent/10 rounded-lg flex items-center justify-center mb-4">
                  <Zap className="w-6 h-6 text-accent" />
                </div>
                <CardTitle>Real-time Analytics</CardTitle>
                <CardDescription>
                  Track campaign performance in real-time with detailed analytics and actionable insights for
                  optimization.
                </CardDescription>
              </CardHeader>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground mb-6">
            Ready to Transform Your Influencer Marketing?
          </h2>
          <p className="text-xl text-muted-foreground mb-8">
            Join hundreds of brands already using AI to find better influencers and achieve higher ROI.
          </p>
          {/* <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/campaign/create">
              <Button size="lg" className="bg-accent hover:bg-accent/90 text-lg px-8">
                Start Your Free Trial
              </Button>
            </Link>
            <Button variant="outline" size="lg" className="text-lg px-8 bg-transparent">
              Schedule a Demo
            </Button>
          </div> */}
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/30 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-7xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center">
            <div className="flex items-center space-x-2 mb-4 md:mb-0">
              <div className="w-8 h-8 bg-accent rounded-lg flex items-center justify-center">
                <Zap className="w-5 h-5 text-accent-foreground" />
              </div>
              <span className="text-xl font-bold text-foreground">InfluenceHub</span>
            </div>
            <div className="flex space-x-6 text-muted-foreground">
              <a href="#" className="hover:text-foreground transition-colors">
                Privacy
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Terms
              </a>
              <a href="#" className="hover:text-foreground transition-colors">
                Support
              </a>
            </div>
          </div>
          <div className="mt-8 pt-8 border-t border-border text-center text-muted-foreground">
            <p>&copy; 2025 InfluenceHub. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
