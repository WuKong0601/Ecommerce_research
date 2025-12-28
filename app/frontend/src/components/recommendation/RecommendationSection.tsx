import { useQuery } from '@tanstack/react-query'
import { apiClient } from '@/lib/api-client'
import { ProductCard } from './ProductCard'
import { Loader2 } from 'lucide-react'

export function RecommendationSection() {
  const { data: recommendations, isLoading } = useQuery({
    queryKey: ['recommendations'],
    queryFn: async () => {
      const { data } = await apiClient.get('/recommendations/for-you')
      return data
    },
  })

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (!recommendations || recommendations.length === 0) {
    return null
  }

  return (
    <section className="py-12">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-3xl font-bold">Recommended For You</h2>
        <p className="text-sm text-muted-foreground">
          Powered by CoFARS-Sparse AI
        </p>
      </div>
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        {recommendations.map((product: any) => (
          <ProductCard key={product.id} product={product} />
        ))}
      </div>
    </section>
  )
}
