# 🚀 Quick Start Guide - CoFARS E-commerce

## ✅ What Has Been Created

A complete full-stack e-commerce application with:

### Backend (NestJS)
- ✅ Authentication with JWT
- ✅ User management with segmentation (Cold-start/Regular/Power)
- ✅ Product catalog with CRUD operations
- ✅ Shopping cart functionality
- ✅ Order processing
- ✅ Product reviews
- ✅ ML-powered recommendations using CoFARS-Sparse model
- ✅ Context-aware suggestions (time-based)
- ✅ PostgreSQL database with Prisma ORM
- ✅ Swagger API documentation

### Frontend (React + TypeScript)
- ✅ Modern UI with TailwindCSS + shadcn/ui
- ✅ Authentication pages (Login/Register)
- ✅ Product browsing and search
- ✅ Shopping cart and checkout
- ✅ Order history
- ✅ User profile with segment display
- ✅ AI-powered recommendation sections
- ✅ Responsive design

## 📋 Prerequisites

Before starting, ensure you have:
- Node.js 18+ installed
- PostgreSQL 14+ installed and running
- Python 3.8+ (for ML model inference)

## 🎯 Setup Instructions

### Step 1: Backend Setup

```bash
cd backend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Edit .env with your settings:
# DATABASE_URL="postgresql://postgres:password@localhost:5432/cofars_ecommerce"
# JWT_SECRET="your-secret-key-here"

# Generate Prisma client
npx prisma generate

# Run database migrations
npx prisma migrate dev --name init

# Start backend server
npm run start:dev
```

Backend will run on: **http://localhost:3001**
API Docs: **http://localhost:3001/api/docs**

### Step 2: Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env file
cp .env.example .env

# Start frontend dev server
npm run dev
```

Frontend will run on: **http://localhost:3000**

### Step 3: Access the Application

1. Open browser: **http://localhost:3000**
2. Click "Sign Up" to create an account
3. Browse products and get personalized recommendations!

## 🗄️ Database Setup

### Option 1: Create Database Manually

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE cofars_ecommerce;

# Exit
\q
```

### Option 2: Let Prisma Create It

Prisma will automatically create the database when you run migrations if it doesn't exist.

## 📊 Seed Sample Data (Optional)

To populate the database with sample products:

```bash
cd backend

# Create seed script
npx prisma db seed
```

## 🔧 Troubleshooting

### Backend Issues

**Port 3001 already in use:**
```bash
# Windows
netstat -ano | findstr :3001
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:3001 | xargs kill -9
```

**Database connection error:**
- Verify PostgreSQL is running
- Check DATABASE_URL in .env
- Ensure database exists

**Prisma errors:**
```bash
npx prisma generate
npx prisma migrate reset
```

### Frontend Issues

**Port 3000 already in use:**
```bash
# Windows
netstat -ano | findstr :3000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:3000 | xargs kill -9
```

**Module not found errors:**
```bash
rm -rf node_modules package-lock.json
npm install
```

## 🎨 Features Overview

### User Segmentation
- **Cold-start (1 interaction)**: Basic recommendations
- **Regular (2-4 interactions)**: Enhanced personalization
- **Power (5+ interactions)**: Full sequence modeling with GRU

### Context-Aware Recommendations
- Automatically detects time of day (morning/afternoon/evening/late_night)
- Considers weekend vs weekday patterns
- Uses CoFARS-Sparse model for predictions

### Shopping Flow
1. Browse products by category
2. View product details
3. Add to cart
4. Checkout with shipping info
5. Track orders
6. Leave reviews

## 📚 API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login
- `GET /api/auth/me` - Get current user

### Products
- `GET /api/products` - List products
- `GET /api/products/:id` - Get product details
- `GET /api/products/categories` - Get categories

### Cart
- `GET /api/cart` - Get cart
- `POST /api/cart/items` - Add to cart
- `DELETE /api/cart/items/:id` - Remove from cart

### Orders
- `POST /api/orders` - Create order
- `GET /api/orders` - Get user orders

### Recommendations
- `GET /api/recommendations/for-you` - Personalized recommendations
- `GET /api/recommendations/context-aware` - Context-based suggestions
- `GET /api/recommendations/similar/:id` - Similar products

## 🔐 Default Test Credentials

After seeding, you can use:
- Email: `test@example.com`
- Password: `password123`

Or create your own account via the registration page.

## 🚀 Production Deployment

### Backend
```bash
cd backend
npm run build
npm run start:prod
```

### Frontend
```bash
cd frontend
npm run build
# Serve dist/ folder with nginx or similar
```

## 📝 Environment Variables

### Backend (.env)
```env
DATABASE_URL="postgresql://user:pass@localhost:5432/cofars_ecommerce"
JWT_SECRET="your-super-secret-jwt-key"
JWT_EXPIRES_IN="7d"
PORT=3001
ML_MODEL_PATH="../../results/models/best_model.pt"
CORS_ORIGIN="http://localhost:3000"
```

### Frontend (.env)
```env
VITE_API_URL="http://localhost:3001/api"
```

## 🎯 Next Steps

1. **Customize Products**: Add your own products via API or admin panel
2. **Train ML Model**: Run the training script to generate recommendations
3. **Customize UI**: Modify components in `frontend/src/components`
4. **Add Features**: Extend with payment integration, admin dashboard, etc.

## 📖 Documentation

- Backend API: http://localhost:3001/api/docs (Swagger)
- Frontend Components: See `frontend/src/components`
- Database Schema: See `backend/prisma/schema.prisma`

## 🆘 Need Help?

1. Check the logs in `backend/logs/`
2. Review API documentation at `/api/docs`
3. Inspect database with `npx prisma studio`
4. Check browser console for frontend errors

## 🎉 Success!

If you see:
- ✅ Backend running on port 3001
- ✅ Frontend running on port 3000
- ✅ Can register and login
- ✅ Can browse products

**Congratulations! Your CoFARS E-commerce application is ready!** 🎊
