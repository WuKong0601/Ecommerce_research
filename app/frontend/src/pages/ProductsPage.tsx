import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { ProductCard } from '@/components/product/ProductCard'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { apiClient } from '@/lib/api-client'
import { useAuthStore } from '@/store/auth'
import { Search, Loader2, Sparkles, ChevronLeft, ChevronRight } from 'lucide-react'

export function ProductsPage() {
  const [search, setSearch] = useState('')
  const [category, setCategory] = useState('')
  const [page, setPage] = useState(1)
  const { isAuthenticated } = useAuthStore()
  
  const ITEMS_PER_PAGE = 24

  const { data, isLoading } = useQuery({
    queryKey: ['products', search, category, page],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (search) params.append('search', search)
      if (category) params.append('category', category)
      params.append('page', page.toString())
      params.append('limit', ITEMS_PER_PAGE.toString())
      const { data } = await apiClient.get(`/products?${params}`)
      return data
    },
  })

  const { data: recommendations } = useQuery({
    queryKey: ['recommendations-for-you'],
    queryFn: async () => {
      const { data } = await apiClient.get('/recommendations/for-you?limit=20')
      return data
    },
    enabled: isAuthenticated && !search && !category,
  })

  const { data: categories } = useQuery({
    queryKey: ['categories'],
    queryFn: async () => {
      const { data } = await apiClient.get('/products/categories')
      return data
    },
  })

  // Merge recommendations with products if available
  const displayProducts = () => {
    if (search || category) {
      return data?.products || []
    }
    
    if (isAuthenticated && recommendations && recommendations.length > 0) {
      // Show recommendations first, then other products
      const recIds = new Set(recommendations.map((r: any) => r.id))
      const otherProducts = (data?.products || []).filter((p: any) => !recIds.has(p.id))
      return [...recommendations, ...otherProducts]
    }
    
    return data?.products || []
  }

  const products = displayProducts()
  const showRecommendedBadge = isAuthenticated && !search && !category && recommendations && recommendations.length > 0
  
  // Reset page when filters change
  const handleSearchChange = (value: string) => {
    setSearch(value)
    setPage(1)
  }
  
  const handleCategoryChange = (value: string) => {
    setCategory(value)
    setPage(1)
  }
  
  const totalPages = data?.totalPages || 1
  const currentPage = data?.page || 1
  const totalProducts = data?.total || 0

  return (
    <div>
      <div className="flex items-center justify-between mb-8">
        <h1 className="text-4xl font-bold">Products</h1>
        {showRecommendedBadge && (
          <div className="flex items-center gap-2 text-sm text-primary">
            <Sparkles className="h-4 w-4" />
            <span>Showing personalized recommendations first</span>
          </div>
        )}
      </div>

      {/* Filters */}
      <div className="mb-8 space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search products..."
            className="pl-10"
            value={search}
            onChange={(e) => handleSearchChange(e.target.value)}
          />
        </div>

        <div className="flex gap-2 flex-wrap">
          <Button
            variant={category === '' ? 'default' : 'outline'}
            onClick={() => handleCategoryChange('')}
          >
            All
          </Button>
          {categories?.map((cat: string) => (
            <Button
              key={cat}
              variant={category === cat ? 'default' : 'outline'}
              onClick={() => handleCategoryChange(cat)}
            >
              {cat}
            </Button>
          ))}
        </div>
      </div>

      {/* Products Grid */}
      {isLoading ? (
        <div className="flex justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
            {products.map((product: any, index: number) => (
              <div key={product.id} className="relative">
                <ProductCard product={product} />
                {showRecommendedBadge && index < (recommendations?.length || 0) && (
                  <div className="absolute -top-2 -left-2 bg-primary text-primary-foreground text-xs font-bold px-2 py-1 rounded-full shadow-lg flex items-center gap-1">
                    <Sparkles className="h-3 w-3" />
                    Recommended
                  </div>
                )}
              </div>
            ))}
          </div>

          {products.length === 0 && (
            <div className="text-center py-12">
              <p className="text-muted-foreground">No products found</p>
            </div>
          )}
          
          {/* Pagination */}
          {totalPages > 1 && (
            <div className="mt-8 flex items-center justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {((currentPage - 1) * ITEMS_PER_PAGE) + 1} to {Math.min(currentPage * ITEMS_PER_PAGE, totalProducts)} of {totalProducts} products
              </div>
              
              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={currentPage === 1}
                >
                  <ChevronLeft className="h-4 w-4" />
                  Previous
                </Button>
                
                <div className="flex items-center gap-1">
                  {Array.from({ length: Math.min(5, totalPages) }, (_, i) => {
                    let pageNum: number
                    if (totalPages <= 5) {
                      pageNum = i + 1
                    } else if (currentPage <= 3) {
                      pageNum = i + 1
                    } else if (currentPage >= totalPages - 2) {
                      pageNum = totalPages - 4 + i
                    } else {
                      pageNum = currentPage - 2 + i
                    }
                    
                    return (
                      <Button
                        key={pageNum}
                        variant={currentPage === pageNum ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setPage(pageNum)}
                        className="w-10"
                      >
                        {pageNum}
                      </Button>
                    )
                  })}
                </div>
                
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                  disabled={currentPage === totalPages}
                >
                  Next
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  )
}
