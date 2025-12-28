import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { RecommendationSection } from '@/components/recommendation/RecommendationSection'
import { ProductCard } from '@/components/product/ProductCard'
import { apiClient } from '@/lib/api-client'
import { useAuthStore } from '@/store/auth'
import { Sparkles, TrendingUp, Clock } from 'lucide-react'

export function HomePage() {
  const { isAuthenticated } = useAuthStore()

  const { data: featuredProducts } = useQuery({
    queryKey: ['featured-products'],
    queryFn: async () => {
      const { data } = await apiClient.get('/products?limit=8')
      return data.products
    },
  })

  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative rounded-lg bg-gradient-to-r from-primary to-primary/80 text-primary-foreground p-12 overflow-hidden">
        <div className="relative z-10 max-w-2xl">
          <h1 className="text-5xl font-bold mb-4">
            Smart Shopping with AI Recommendations
          </h1>
          <p className="text-lg mb-6 opacity-90">
            Experience personalized product suggestions powered by CoFARS-Sparse machine learning model
          </p>
          <Link to="/products">
            <Button size="lg" variant="secondary">
              Shop Now
            </Button>
          </Link>
        </div>
        <div className="absolute right-0 top-0 bottom-0 w-1/3 opacity-10">
          <Sparkles className="w-full h-full" />
        </div>
      </section>

      {/* Features */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="p-6 rounded-lg border bg-card">
          <Sparkles className="h-10 w-10 text-primary mb-4" />
          <h3 className="text-xl font-semibold mb-2">AI-Powered Recommendations</h3>
          <p className="text-muted-foreground">
            Get personalized product suggestions based on your preferences and behavior
          </p>
        </div>
        <div className="p-6 rounded-lg border bg-card">
          <Clock className="h-10 w-10 text-primary mb-4" />
          <h3 className="text-xl font-semibold mb-2">Context-Aware</h3>
          <p className="text-muted-foreground">
            Recommendations adapt to time of day and your shopping patterns
          </p>
        </div>
        <div className="p-6 rounded-lg border bg-card">
          <TrendingUp className="h-10 w-10 text-primary mb-4" />
          <h3 className="text-xl font-semibold mb-2">Smart Segmentation</h3>
          <p className="text-muted-foreground">
            Tailored experience for power users, regular shoppers, and newcomers
          </p>
        </div>
      </section>

      {/* Personalized Recommendations */}
      {isAuthenticated && <RecommendationSection />}

      {/* Featured Products */}
      <section>
        <h2 className="text-3xl font-bold mb-6">Featured Products</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {featuredProducts?.map((product: any) => (
            <ProductCard key={product.id} product={product} />
          ))}
        </div>
      </section>
    </div>
  )
}
