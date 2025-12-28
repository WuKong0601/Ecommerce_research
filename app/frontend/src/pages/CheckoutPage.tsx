import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation, useQuery } from '@tanstack/react-query'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { apiClient } from '@/lib/api-client'
import { useCartStore } from '@/store/cart'

export function CheckoutPage() {
  const navigate = useNavigate()
  const clearCart = useCartStore((state) => state.clearCart)
  const [formData, setFormData] = useState({
    shippingAddress: '',
    shippingPhone: '',
  })

  const { data: cart } = useQuery({
    queryKey: ['cart'],
    queryFn: async () => {
      const { data } = await apiClient.get('/cart')
      return data
    },
  })

  const createOrderMutation = useMutation({
    mutationFn: async () => {
      const { data } = await apiClient.post('/orders', formData)
      return data
    },
    onSuccess: () => {
      clearCart()
      navigate('/orders')
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    createOrderMutation.mutate()
  }

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-4xl font-bold mb-8">Checkout</h1>

      <form onSubmit={handleSubmit} className="space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Shipping Information</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Shipping Address</label>
              <Input
                value={formData.shippingAddress}
                onChange={(e) =>
                  setFormData({ ...formData, shippingAddress: e.target.value })
                }
                placeholder="123 Main St, City, Country"
                required
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Phone Number</label>
              <Input
                type="tel"
                value={formData.shippingPhone}
                onChange={(e) =>
                  setFormData({ ...formData, shippingPhone: e.target.value })
                }
                placeholder="0123456789"
                required
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Order Summary</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Items ({cart?.items?.length || 0})</span>
                <span>${cart?.total?.toFixed(2) || '0.00'}</span>
              </div>
              <div className="flex justify-between">
                <span>Shipping</span>
                <span>Free</span>
              </div>
              <div className="border-t pt-2 flex justify-between font-bold text-lg">
                <span>Total</span>
                <span>${cart?.total?.toFixed(2) || '0.00'}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <Button
          type="submit"
          className="w-full"
          size="lg"
          disabled={createOrderMutation.isPending}
        >
          {createOrderMutation.isPending ? 'Placing Order...' : 'Place Order'}
        </Button>
      </form>
    </div>
  )
}
