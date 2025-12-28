import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Card, CardContent } from '@/components/ui/card'
import { apiClient } from '@/lib/api-client'
import { useCartStore } from '@/store/cart'
import { ShoppingCart, Star, Loader2 } from 'lucide-react'

export function ProductDetailPage() {
  const { id } = useParams()
  const queryClient = useQueryClient()
  const setCart = useCartStore((state) => state.setCart)

  const { data: product, isLoading } = useQuery({
    queryKey: ['product', id],
    queryFn: async () => {
      const { data } = await apiClient.get(`/products/${id}`)
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
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
      <div className="aspect-square rounded-lg overflow-hidden bg-muted">
        {product.imageUrl ? (
          <img
            src={product.imageUrl}
            alt={product.name}
            className="w-full h-full object-cover"
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
          </div>
          <p className="text-3xl font-bold">${product.price}</p>
        </div>

        <div>
          <h3 className="font-semibold mb-2">Description</h3>
          <p className="text-muted-foreground">{product.description}</p>
        </div>

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
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
