import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ContextSelector } from '@/components/context/ContextSelector'
import { ProductCard } from '@/components/product/ProductCard'
import { apiClient } from '@/lib/api-client'
import { Loader2, Sparkles, TrendingUp, Info } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'

export function RecommendationsPage() {
  const [timeSlot, setTimeSlot] = useState('morning')
  const [isWeekend, setIsWeekend] = useState(false)

  const handleContextChange = (newTimeSlot: string, newIsWeekend: boolean) => {
    setTimeSlot(newTimeSlot)
    setIsWeekend(newIsWeekend)
  }

  const { data: recommendations, isLoading, error } = useQuery({
    queryKey: ['context-recommendations', timeSlot, isWeekend],
    queryFn: async () => {
      const { data } = await apiClient.get('/recommendations/context-aware', {
        params: { timeSlot, isWeekend }
      })
      return data
    },
  })

  const { data: userProfile } = useQuery({
    queryKey: ['profile'],
    queryFn: async () => {
      const { data } = await apiClient.get('/users/profile')
      return data
    },
  })

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-bold mb-2">AI-Powered Recommendations</h1>
        <p className="text-muted-foreground">
          Get personalized product suggestions based on CoFARS-Sparse machine learning model
        </p>
      </div>

      {/* User Segment Info */}
      {userProfile && (
        <Card className="bg-gradient-to-r from-primary/10 to-primary/5 border-primary/20">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <TrendingUp className="h-5 w-5 text-primary" />
              <div>
                <p className="font-medium">
                  Your Profile: <span className="text-primary">{userProfile.segment}</span> User
                </p>
                <p className="text-sm text-muted-foreground">
                  {userProfile.interactionCount} interactions • 
                  {userProfile.segment === 'COLD_START' && ' Getting to know your preferences'}
                  {userProfile.segment === 'REGULAR' && ' Building your personalized experience'}
                  {userProfile.segment === 'POWER' && ' Full sequence modeling with GRU network'}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Context Selector */}
      <ContextSelector 
        onContextChange={handleContextChange}
        currentTimeSlot={timeSlot}
        currentIsWeekend={isWeekend}
      />

      {/* How It Works */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="h-5 w-5" />
            How Context-Aware Recommendations Work
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div className="space-y-2">
              <div className="font-semibold text-primary">1. Context Detection</div>
              <p className="text-muted-foreground">
                System identifies your shopping context (time of day, day type) to understand your current needs
              </p>
            </div>
            <div className="space-y-2">
              <div className="font-semibold text-primary">2. Pattern Analysis</div>
              <p className="text-muted-foreground">
                CoFARS-Sparse analyzes similar users' behavior in the same context using Jensen-Shannon divergence
              </p>
            </div>
            <div className="space-y-2">
              <div className="font-semibold text-primary">3. Smart Recommendations</div>
              <p className="text-muted-foreground">
                Model predicts products you're most likely to be interested in based on context patterns
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Recommendations Grid */}
      <div>
        <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
          <Sparkles className="h-6 w-6 text-primary" />
          Recommended for You
        </h2>

        {isLoading ? (
          <div className="flex justify-center py-12">
            <Loader2 className="h-8 w-8 animate-spin text-primary" />
          </div>
        ) : error ? (
          <Card className="p-8 text-center">
            <p className="text-muted-foreground">
              Unable to load recommendations. Please try again later.
            </p>
          </Card>
        ) : recommendations && recommendations.length > 0 ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {recommendations.map((product: any, index: number) => (
              <div key={product.id} className="relative">
                <ProductCard product={product} />
                {index < 3 && (
                  <div className="absolute -top-2 -right-2 bg-primary text-primary-foreground text-xs font-bold rounded-full w-8 h-8 flex items-center justify-center shadow-lg">
                    #{index + 1}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <Card className="p-8 text-center">
            <Sparkles className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <p className="text-muted-foreground">
              No recommendations available for this context yet.
              <br />
              Try interacting with more products to improve recommendations!
            </p>
          </Card>
        )}
      </div>

      {/* Context Statistics */}
      {recommendations && recommendations.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Context Statistics</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <p className="text-muted-foreground">Context</p>
                <p className="font-semibold">{timeSlot.replace('_', ' ')} • {isWeekend ? 'Weekend' : 'Weekday'}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Recommendations</p>
                <p className="font-semibold">{recommendations.length} products</p>
              </div>
              <div>
                <p className="text-muted-foreground">Model</p>
                <p className="font-semibold">CoFARS-Sparse</p>
              </div>
              <div>
                <p className="text-muted-foreground">User Segment</p>
                <p className="font-semibold">{userProfile?.segment || 'Loading...'}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
