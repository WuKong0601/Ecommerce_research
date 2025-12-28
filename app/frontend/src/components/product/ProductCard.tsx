import { Link } from 'react-router-dom'
import { ShoppingCart, Star } from 'lucide-react'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { Button } from '@/components/ui/button'

interface ProductCardProps {
  product: {
    id: string
    name: string
    price: number
    imageUrl?: string
    category: string
    ratingLevel: number
  }
  onAddToCart?: () => void
}

export function ProductCard({ product, onAddToCart }: ProductCardProps) {
  return (
    <Card className="overflow-hidden hover:shadow-lg transition-shadow">
      <Link to={`/products/${product.id}`}>
        <div className="aspect-square overflow-hidden bg-muted">
          {product.imageUrl ? (
            <img
              src={product.imageUrl}
              alt={product.name}
              className="w-full h-full object-cover hover:scale-105 transition-transform"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center text-muted-foreground">
              No Image
            </div>
          )}
        </div>
      </Link>

      <CardHeader className="p-4">
        <Link to={`/products/${product.id}`}>
          <h3 className="font-semibold line-clamp-2 hover:text-primary">
            {product.name}
          </h3>
        </Link>
        <p className="text-sm text-muted-foreground">{product.category}</p>
      </CardHeader>

      <CardContent className="p-4 pt-0">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-2xl font-bold">${product.price}</p>
            <div className="flex items-center gap-1 mt-1">
              {Array.from({ length: product.ratingLevel }).map((_, i) => (
                <Star key={i} className="h-4 w-4 fill-yellow-400 text-yellow-400" />
              ))}
            </div>
          </div>
          <Button size="icon" onClick={onAddToCart}>
            <ShoppingCart className="h-4 w-4" />
          </Button>
        </div>
      </CardContent>
    </Card>
  )
}
