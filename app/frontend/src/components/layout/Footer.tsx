export function Footer() {
  return (
    <footer className="border-t bg-muted/50">
      <div className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div>
            <h3 className="font-bold text-lg mb-4">CoFARS Shop</h3>
            <p className="text-sm text-muted-foreground">
              Smart e-commerce with AI-powered recommendations
            </p>
          </div>
          
          <div>
            <h4 className="font-semibold mb-4">Shop</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="/products" className="text-muted-foreground hover:text-foreground">All Products</a></li>
              <li><a href="/products?category=furniture" className="text-muted-foreground hover:text-foreground">Furniture</a></li>
              <li><a href="/products?category=decor" className="text-muted-foreground hover:text-foreground">Home Decor</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">Account</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="/profile" className="text-muted-foreground hover:text-foreground">My Profile</a></li>
              <li><a href="/orders" className="text-muted-foreground hover:text-foreground">My Orders</a></li>
              <li><a href="/cart" className="text-muted-foreground hover:text-foreground">Shopping Cart</a></li>
            </ul>
          </div>

          <div>
            <h4 className="font-semibold mb-4">About</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#" className="text-muted-foreground hover:text-foreground">About Us</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-foreground">Contact</a></li>
              <li><a href="#" className="text-muted-foreground hover:text-foreground">Privacy Policy</a></li>
            </ul>
          </div>
        </div>

        <div className="mt-8 pt-8 border-t text-center text-sm text-muted-foreground">
          <p>&copy; 2025 CoFARS E-commerce. Powered by CoFARS-Sparse ML Model.</p>
        </div>
      </div>
    </footer>
  )
}
