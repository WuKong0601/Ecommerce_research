import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { apiClient } from '@/lib/api-client'
import { useCartStore } from '@/store/cart'
import { useAuthStore } from '@/store/auth'
import { ShoppingCart, Star, Loader2, User, MessageSquare } from 'lucide-react'
import { formatVND } from '@/lib/utils'

export function ProductDetailPage() {
  const { id } = useParams()
  const queryClient = useQueryClient()
  const setCart = useCartStore((state) => state.setCart)
  const { isAuthenticated } = useAuthStore()

  const { data: product, isLoading } = useQuery({
    queryKey: ['product', id],
    queryFn: async () => {
      const { data } = await apiClient.get(`/products/${id}`)
      return data
    },
  })

  const { data: reviews, isLoading: reviewsLoading } = useQuery({
    queryKey: ['reviews', id],
    queryFn: async () => {
      const { data } = await apiClient.get(`/reviews/product/${id}`)
      return data
    },
  })

  const addToCartMutation = useMutation({
    mutationFn: async () => {
      const { data } = await apiClient.post('/cart/items', {
        productId: id,
        quantity: 1,
      })
      return data
    },
    onSuccess: async () => {
      const { data } = await apiClient.get('/cart')
      setCart(data.items, data.total)
      queryClient.invalidateQueries({ queryKey: ['cart'] })
    },
  })

  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <Loader2 className="h-8 w-8 animate-spin text-primary" />
      </div>
    )
  }

  if (!product) {
    return <div>Product not found</div>
  }

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <div className="aspect-square rounded-lg overflow-hidden bg-muted">
          {product.imageUrl ? (
            <img
              src={product.imageUrl}
              alt={product.name}
              className="w-full h-full object-cover"
              onError={(e) => {
                // Use inline fallback instead of external URL
                e.currentTarget.style.display = 'none'
                const parent = e.currentTarget.parentElement
                if (parent) {
                  parent.innerHTML = `<div class="w-full h-full flex items-center justify-center bg-gradient-to-br from-gray-100 to-gray-200 text-gray-400 text-lg p-8 text-center">${product.name.substring(0, 50)}...</div>`
                }
              }}
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-muted-foreground">
              No Image
            </div>
          )}
        </div>

        <div className="space-y-6">
          <div>
            <p className="text-sm text-muted-foreground mb-2">{product.category}</p>
            <h1 className="text-4xl font-bold mb-4">{product.name}</h1>
            <div className="flex items-center gap-2 mb-4">
              {Array.from({ length: product.ratingLevel }).map((_, i) => (
                <Star key={i} className="h-5 w-5 fill-yellow-400 text-yellow-400" />
              ))}
              <span className="text-sm text-muted-foreground ml-2">
                ({reviews?.length || 0} reviews)
              </span>
            </div>
            <p className="text-3xl font-bold">{formatVND(product.price)}</p>
          </div>

          <div>
            <h3 className="font-semibold mb-2">Description</h3>
            <p className="text-muted-foreground">{product.description}</p>
          </div>

          {isAuthenticated && (
            <div className="flex gap-4">
              <Button
                size="lg"
                className="flex-1"
                onClick={() => addToCartMutation.mutate()}
                disabled={addToCartMutation.isPending}
              >
                <ShoppingCart className="mr-2 h-5 w-5" />
                {addToCartMutation.isPending ? 'Adding...' : 'Add to Cart'}
              </Button>
            </div>
          )}

          <Card>
            <CardContent className="p-4">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Stock</p>
                  <p className="font-semibold">{product.stock} available</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Category</p>
                  <p className="font-semibold">{product.category}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Group</p>
                  <p className="font-semibold">{product.group}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Price Bucket</p>
                  <p className="font-semibold">Level {product.priceBucket}</p>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Reviews Section */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MessageSquare className="h-5 w-5" />
            Customer Reviews ({reviews?.length || 0})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {reviewsLoading ? (
            <div className="flex justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin text-primary" />
            </div>
          ) : reviews && reviews.length > 0 ? (
            <div className="space-y-4">
              {reviews.map((review: any) => (
                <div key={review.id} className="border-b pb-4 last:border-0">
                  <div className="flex items-start justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <User className="h-4 w-4 text-muted-foreground" />
                      <span className="font-medium">{review.user?.name || 'Anonymous'}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      {Array.from({ length: review.rating }).map((_, i) => (
                        <Star key={i} className="h-4 w-4 fill-yellow-400 text-yellow-400" />
                      ))}
                    </div>
                  </div>
                  {review.comment && (
                    <p className="text-sm text-muted-foreground">{review.comment}</p>
                  )}
                  <p className="text-xs text-muted-foreground mt-2">
                    {new Date(review.createdAt).toLocaleDateString()}
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-center text-muted-foreground py-8">
              No reviews yet. Be the first to review this product!
            </p>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
