import { create } from 'zustand'

interface CartItem {
  id: string
  productId: string
  quantity: number
  product: any
}

interface CartState {
  items: CartItem[]
  total: number
  setCart: (items: CartItem[], total: number) => void
  clearCart: () => void
}

export const useCartStore = create<CartState>((set) => ({
  items: [],
  total: 0,
  setCart: (items, total) => set({ items, total }),
  clearCart: () => set({ items: [], total: 0 }),
}))
