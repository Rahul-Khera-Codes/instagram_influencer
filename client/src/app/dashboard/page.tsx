"use client"

import { useState, useEffect } from "react"
import { ProtectedRoute, useAuth } from "@/lib/auth"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { Badge } from "@/components/ui/badge"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from "@/components/ui/dialog"
import { Search, LayoutDashboard, Compass, Users, Megaphone, Bookmark, BarChart3 } from "lucide-react"
import Image from "next/image"


const sidebarItems = [
  // { icon: LayoutDashboard, label: "Dashboard", active: false },
  // { icon: Compass, label: "Discover", active: false },
  { icon: Users, label: "Find Influencers", active: true },
  // { icon: Megaphone, label: "Campaigns", active: false },
]

export default function Dashboard() {
  const { logout, user } = useAuth()
  const [searchKeywords, setSearchKeywords] = useState("")
  const [selectedPlatform, setSelectedPlatform] = useState("Instagram")
  const [currentPage, setCurrentPage] = useState(1)
  const [rowsPerPage, setRowsPerPage] = useState(10)
  const [searchLimit, setSearchLimit] = useState(5)  // default 5


  const [results, setResults] = useState<any[]>([])
  const [isSemanticLoading, setIsSemanticLoading] = useState(false)
  const [isProfilesLoading, setIsProfilesLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://116.202.210.102:8006"
  const [summaryOpen, setSummaryOpen] = useState(false)
  const [currentSummary, setCurrentSummary] = useState('')
  const [loadingSummary, setLoadingSummary] = useState(false)
  const [selectedInfluencer, setSelectedInfluencer] = useState<any>(null)

  const [lastSearchKey, setLastSearchKey] = useState<string | null>(null)
  const [lastSearchResults, setLastSearchResults] = useState<any[] | null>(null)

  // suggestedTags come from backend `expanded_terms` (semantic search). Start with defaults.
  const [suggestedTags, setSuggestedTags] = useState<string[]>(["fashion hyderabad", "food", "tech"])


    // 🔹 Fetch search results
  const handleSearch = async (overrideKeywords?: string) => {
    setIsProfilesLoading(true)
    setError(null)

    try {
      const raw = (overrideKeywords !== undefined ? String(overrideKeywords) : (searchKeywords || "")).trim()
      const key = `${raw.toLowerCase()}|${searchLimit}`

      // Client-side short-circuit if same query+limit already fetched
      if (key && lastSearchKey === key && lastSearchResults) {
        setResults(lastSearchResults)
        setIsProfilesLoading(false)
        return
      }

      const q = encodeURIComponent(raw)
      // include user id in the request so backend can scope cached results per-user
      const userId = (user as any)?.id || (user as any)?._id || null
      const searchUrl = `${API_BASE}/influencers/search/top?keyword=${q}&limit=${searchLimit}${userId ? `&user_id=${encodeURIComponent(String(userId))}` : ""}`
      const res = await fetch(searchUrl, { credentials: "omit" })
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      const data = await res.json()
      console.log("Search data:", data)
      
      const baseResults = data.results || []

      // If backend returned cached results (and you saved enriched fields server-side),
      // they should already contain metrics — avoid extra /insights calls.
      const backendHasMetrics =
  baseResults.length > 0 &&
  baseResults.every(
    (item: { post_count?: number; avg_likes?: number; total_posts?: number }) =>
      item.post_count !== undefined ||
      item.avg_likes !== undefined ||
      item.total_posts !== undefined
  )

      let finalResults = baseResults
      if (!backendHasMetrics) {
        finalResults = await enrichInfluencers(baseResults)
      }


      setResults(finalResults)
      // store client-side cache for quick re-use
      setLastSearchKey(key)
      setLastSearchResults(finalResults)
      console.log("Enriched results:", finalResults)

    } catch (err) {
      setError((err as Error).message)
    } finally {
      setIsProfilesLoading(false)
    }
  }

  // 🔹 Fetch semantic expansion for a query and update suggested tags
  const fetchSemantic = async (query?: string) => {
    const raw = (query !== undefined ? String(query) : (searchKeywords || "")).trim()
    if (!raw) return
    setIsSemanticLoading(true)
    try {
      const q = encodeURIComponent(raw)
      const res = await fetch(`${API_BASE}/influencers/search/semantic?query=${q}&top_k=${searchLimit}&expand=true`, { credentials: "omit" })
      if (!res.ok) throw new Error(`API error: ${res.status}`)
      const data = await res.json()
      console.log("Semantic data:", data)
      if (Array.isArray(data.expanded_terms) && data.expanded_terms.length > 0) {
        // keep the original expanded_terms array intact and set to state
        setSuggestedTags(data.expanded_terms.map((t: string) => String(t)))
      }
      // do not fetch top results yet; wait for user to click a suggested tag
    } catch (err) {
      console.error("Semantic search error", err)
    } finally {
      setIsSemanticLoading(false)
    }
  }

  // 🔹 Enrich influencers with insights + followers
  const enrichInfluencers = async (influencers: any[]) => {
    const results: any[] = []
    const insightsCache: Record<string, any> = {}

    for (const r of influencers) {
      const userId = r.pk || r.id
      let out = { ...r }

      // skip enrichment if metrics already present
      if (userId && (out.post_count === undefined && out.avg_likes === undefined && out.total_posts === undefined)) {
        try {
          // memoize per-run to avoid duplicate /insights calls
          if (!insightsCache[userId]) {
            const res = await fetch(`${API_BASE}/influencers/insights?user_id=${encodeURIComponent(String(userId))}`)
            if (res.ok) {
              const insights = await res.json()
              insightsCache[userId] = insights
            } else {
              insightsCache[userId] = null
            }
            // small delay between requests to avoid rate limits
            await new Promise(resolve => setTimeout(resolve, 1200))
          }

          if (insightsCache[userId]) {
            out = { ...out, ...insightsCache[userId] }
          }
        } catch (err) {
          console.error(`Failed to fetch insights for user ${r.username || r.full_name}`, err)
        }
      }

      results.push(out)
    }
    return results
  }



  // 🔹 Generate AI Summary
  const handleGenerateSummary = async (influencer: any) => {
    setSelectedInfluencer(influencer)
    setSummaryOpen(true)
    setLoadingSummary(true)

    try {
      const payload = {
        username: influencer.username || influencer.handle,
        text: influencer.bio || "",
        followers: influencer.followers || 0,
        avg_likes: influencer.avg_likes || 0,
        eng_rate: influencer.eng_rate || 0,
        niches: influencer.niches || [],
        platform: influencer.full || "Instagram",

      }

      const res = await fetch(`${API_BASE}/influencers/summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!res.ok) throw new Error(`API error: ${res.status}`)
      const data = await res.json()
      setCurrentSummary(data.summary || "No summary available")
    } catch (err) {
      console.error("Summary generation error:", err)
      setCurrentSummary("Failed to generate summary. Please try again.")
    } finally {
      setLoadingSummary(false)
    }
  }

  const sanitizeImageUrl = (u: any) => {
    if (!u) return null
    try {
      let s = String(u).trim()
      // remove wrapping quotes if any
      s = s.replace(/^"+|"+$/g, "")
      s = s.replace(/^'+|'+$/g, "")
      // prefer https to avoid mixed-content blocking
      if (s.startsWith("http://")) s = s.replace(/^http:\/\//i, "https://")
      return s
    } catch {
      return null
    }
  }

  // helpers to format numbers and percents
  const formatNumber = (val: any) => {
    if (val === null || val === undefined || val === "") return "-"
    const n = Number(val)
    if (Number.isNaN(n)) return String(val)
    if (n >= 1_000_000_000) return (n / 1_000_000_000).toFixed(2) + "B"
    if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + "M"
    if (n >= 1_000) return (n / 1_000).toFixed(1) + "K"
    return n.toLocaleString()
  }

  // parse follower value (e.g. 1.2K, 2,300, "2300", null) -> numeric value
  const parseFollowerCount = (val: any): number => {
    if (val === null || val === undefined || val === "") return 0
    if (typeof val === "number") return val
    const s = String(val).replace(/,/g, "").trim()
    const m = s.match(/^([\d,.]+)\s*([kKmMbB])?$/)
    if (!m) {
      const n = Number(s)
      return Number.isFinite(n) ? n : 0
    }
    let num = Number(m[1])
    const suffix = (m[2] || "").toUpperCase()
    if (suffix === "K") num = num * 1_000
    if (suffix === "M") num = num * 1_000_000
    if (suffix === "B") num = num * 1_000_000_000
    return Math.round(num)
  }

  const formatPercent = (val: any) => {
    if (val === null || val === undefined || val === "") return "-"
    const n = Number(val)
    if (Number.isNaN(n)) return String(val)
    return `${n.toFixed(2)}%`
  }

  // sort utility: descending by followers (numerical)
  const sortByFollowersDesc = (arr: any[]) => {
    return arr.slice().sort((a, b) => parseFollowerCount(b.followers) - parseFollowerCount(a.followers))
  }

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-gray-50 flex">

      {/* Sidebar */}
      <div className="w-64 bg-white border-r border-gray-200 flex flex-col">
        {/* Logo and User */}
        <div className="p-6 border-b border-gray-200">
          <div className="flex items-center gap-3 mb-6">
            <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center">
              <Users className="w-4 h-4 text-gray-600" />
            </div>
            <div>
              <h2 className="font-semibold text-gray-900">InfluencePilot</h2>
              <p className="text-sm text-gray-500">Creator CRM</p>
            </div>
          </div>
          <div className="mt-4">
            
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4">
          <ul className="space-y-2">
            {sidebarItems.map((item, index) => (
              <li key={index}>
                <button
                  className={`w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left transition-colors ${
                    item.active ? "bg-blue-50 text-blue-700" : "text-gray-600 hover:bg-gray-50"
                  }`}
                >
                  <item.icon className="w-5 h-5" />
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        </nav>

        {/* New Campaign Button */}
        {/* <div className="p-4 border-t border-gray-200">
          <Button className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700">
            New Campaign
          </Button>
        </div> */}
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-white border-b border-gray-200 px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-semibold text-gray-900">Welcome, {user?.username || user?.email || 'User'}</h1>
         
            <Button variant="outline" size="sm" onClick={logout} className="text-base cursor-pointer">
                Logout
              </Button>
        </header>

        {/* Content */}
        <main className="flex-1 p-6">
          <div className="max-w-7xl mx-auto">
            <h2 className="text-2xl font-semibold text-gray-900 mb-6">Find Influencers</h2>

            {/* Search Section */}
            <Card className="p-6 mb-6">
              <div className="flex gap-4 mb-4">
                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Platform</label>
                  <Select value={selectedPlatform} onValueChange={setSelectedPlatform}>
                    <SelectTrigger className="w-48">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="Instagram">Instagram</SelectItem>
                      {/* <SelectItem value="YouTube">YouTube</SelectItem>
                      <SelectItem value="TikTok">TikTok</SelectItem>
                      <SelectItem value="Twitter">Twitter</SelectItem> */}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex-1">
  <label className="block text-sm font-medium text-gray-700 mb-2">Limit</label>
  <Input
    type="number"
    min={1}
    max={50}
    value={searchLimit}
    onChange={(e) => setSearchLimit(Number(e.target.value))}
    className="pl-3"
  />
</div>


                <div className="flex-1">
                  <label className="block text-sm font-medium text-gray-700 mb-2">Keywords</label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                    <Input
                      placeholder="e.g. fashion hyderabad"
                      value={searchKeywords}
                      onChange={(e) => setSearchKeywords(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                </div>

                <div className="flex items-end">
                    <Button
  className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700"
  onClick={() => fetchSemantic()}
  disabled={isSemanticLoading}
>
  {isSemanticLoading ? "Searching..." : "See Influencers"}
</Button>



                </div>
              </div>

              {/* Suggested Tags */}
              <div className="flex gap-2 w-full overflow-x-scroll">
                {isSemanticLoading ? (
                  <div className="flex items-center px-4">
                    <div className="animate-pulse text-sm text-gray-500">Loading suggestions...</div>
                  </div>
                ) : (
                  suggestedTags.map((tag, index) => (
                    <Badge
                      key={index}
                      variant="secondary"
                      className="cursor-pointer hover:bg-gray-200"
                      onClick={() => {
                        setSearchKeywords(tag)
                        handleSearch(tag)
                      }}
                    >
                      {tag}
                    </Badge>
                  ))
                )}
              </div>
            </Card>

            {/* Results Table */}
            <Card>
              {isProfilesLoading ? (
                <div className="flex items-center justify-center p-12">
                  <div className="flex flex-col items-center gap-4">
                    <div className="animate-spin rounded-full h-12 w-12 border-4 border-gray-200 border-t-4 border-t-blue-600"></div>
                    <div className="text-sm text-gray-600">Searching...</div>
                  </div>
                </div>
              ) : (
                <>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead className="border-b border-gray-200">
                        <tr>
                          <th className="text-left py-4 px-6 font-medium text-gray-700">Influencer</th>
                          <th className="text-left py-4 px-6 font-medium text-gray-700">Follower</th>
                          <th className="text-left py-4 px-6 font-medium text-gray-700">Engagement</th>
                          <th className="text-left py-4 px-6 font-medium text-gray-700">Avg Likes</th>
                          <th className="text-left py-4 px-6 font-medium text-gray-700">Eng. Rate</th>
                          <th className="text-left py-4 px-6 font-medium text-gray-700">Posts</th>
                          <th className="text-left py-4 px-6 font-medium text-gray-700">Full Name</th>
                        </tr>
                      </thead>
                      <tbody>
                        {isProfilesLoading ? (
                          <tr><td colSpan={7} className="py-6 text-center text-gray-500">Loading profiles...</td></tr>
                        ) : (results && results.length > 0 ? (
                  results.map((influencer: any, idx: number) => (
                          <tr key={influencer.id || idx} className="border-b border-gray-100 hover:bg-gray-50">
                            <td className="py-4 px-6">
                                  {/* <p>{influencer?.profile_pic}</p> */}
                              <div className="flex items-center gap-3 flex-col">
                                <Image alt="image" src={influencer?.profile_pic} width={40} height={40} className=" rounded-full"/>
                          

                                <div className="flex">
                                  <a
  href={`https://instagram.com/${influencer.username}`}
  target="_blank"
  rel="noopener noreferrer"
  className="text-gray-600 text-sm hover:underline hover:text-blue-600"
>
  @{influencer.username}
</a>
                                  <div className="text-sm text-black">{influencer.handle} • <span className="pl-1">Instagram</span></div>
                                </div>
                              </div>
                            </td>
                            <td className="py-4 px-6 font-medium text-gray-900">{formatNumber(influencer.followers)}</td>
                            <td className="py-4 px-6 text-gray-700">{formatNumber(influencer.engagement)}</td>
                            <td className="py-4 px-6 text-gray-700">{formatNumber(influencer.avg_likes)}</td>
                            <td className="py-4 px-6 text-gray-700">{formatPercent(influencer.engagement_rate_percent)}</td>
                            <td className="py-4 px-6 text-gray-700">{formatNumber(influencer.total_posts)}</td>
                            <td className="py-4 px-6">
                              <div className="flex gap-1">
                                {(influencer.niches || influencer.category || influencer.full_name || []).toString().split(",").slice(0,3).map((niche: string, index: number) => (
                                  <Badge key={index} variant="secondary" className="text-xs">{niche}</Badge>
                                ))}
                              </div>
                            </td>
                            <td className="py-4 px-6">
                              <div className="flex gap-2">
                                <Button
                                  className="cursor-pointer bg-blue-600/60 outline-1 outline-gray-300"
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleGenerateSummary(influencer)}
                                >
                                  Generate Summary
                                </Button>
                              </div>
                            </td>
                          </tr>
                        ))):(
                          <tr>
    <td colSpan={7} className="py-6 text-center text-gray-500">
      No data found
    </td>
  </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {/* Pagination */}
                  {/* <div className="flex items-center justify-between px-6 py-4 border-t border-gray-200">
                    <div className="flex items-center gap-2 text-sm text-gray-600">
                      <span>Rows per page:</span>
                      <Select value={rowsPerPage.toString()} onValueChange={(value) => setRowsPerPage(Number(value))}>
                        <SelectTrigger className="w-16 h-8"><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="10">10</SelectItem>
                          <SelectItem value="25">25</SelectItem>
                          <SelectItem value="50">50</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="text-sm text-gray-600">1-4 of 4</div>
                  </div> */}
                </>
              )}
            </Card>
          </div>
        </main>
      </div>
    </div>

    {/* Summary Dialog */}
    <Dialog open={summaryOpen} onOpenChange={setSummaryOpen}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>
            {selectedInfluencer ? 
              `Summary for ${selectedInfluencer.full_name || selectedInfluencer.name || selectedInfluencer.username || selectedInfluencer.handle}` : 
              'Influencer Summary'}
          </DialogTitle>
          <DialogDescription>
            AI-generated summary based on profile data
          </DialogDescription>
        </DialogHeader>
        <div className="p-4 bg-gray-50 rounded-md overflow-y-scroll h-full">
          {loadingSummary ? (
            <div className="flex items-center justify-center py-8">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
            </div>
          ) : (
            <div className="whitespace-pre-line">{currentSummary}</div>
          )}
        </div>
      </DialogContent>
    </Dialog>
    </ProtectedRoute>
  )
}
