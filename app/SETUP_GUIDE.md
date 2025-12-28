# CoFARS E-commerce Setup Guide

## Quick Start

### Prerequisites
- Node.js 18+ installed
- PostgreSQL 14+ installed and running
- Python 3.8+ (for ML model)
- Git

### Step 1: Backend Setup

```bash
cd app/backend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Edit .env and update:
# - DATABASE_URL with your PostgreSQL credentials
# - JWT_SECRET with a secure random string

# Generate Prisma client
npx prisma generate

# Run database migrations
npx prisma migrate dev

# Seed sample data
npm run prisma:seed

# Start backend server
npm run start:dev
```

Backend will run on http://localhost:3001

### Step 2: Frontend Setup

```bash
cd app/frontend

# Install dependencies
npm install

# Copy environment file
cp .env.example .env

# Start frontend dev server
npm run dev
```

Frontend will run on http://localhost:3000

### Step 3: Access the Application

1. Open browser: http://localhost:3000
2. Register a new account or use demo credentials
3. Browse products and get personalized recommendations!

## API Documentation

Once backend is running, visit:
- Swagger UI: http://localhost:3001/api/docs

## Database Schema

The application uses PostgreSQL with the following main tables:
- `User` - User accounts with segmentation (Cold-start/Regular/Power)
- `Product` - Product catalog with attributes
- `Order` - Customer orders
- `CartItem` - Shopping cart items
- `Review` - Product reviews
- `UserInteraction` - Interaction tracking for ML model
- `ContextPrototype` - Context embeddings cache
- `RecommendationCache` - Cached recommendations

## ML Model Integration

The backend integrates with the trained CoFARS-Sparse model:
- Model path: `../../results/models/best_model.pt`
- Python inference service for real-time recommendations
- Context-aware suggestions based on time and user behavior

## Features

### User Features
- ✅ User registration & JWT authentication
- ✅ Browse products by category
- ✅ Context-aware recommendations (time-based)
- ✅ Shopping cart management
- ✅ Order placement & tracking
- ✅ Product reviews
- ✅ User profile & order history

### Admin Features
- ✅ Product management (CRUD)
- ✅ Order management
- ✅ User analytics
- ✅ Recommendation metrics

### ML Features
- ✅ Real-time context detection (time_slot + is_weekend)
- ✅ User segmentation (Power/Regular/Cold-start)
- ✅ Hybrid recommendation strategy
- ✅ Context similarity-based enrichment

## Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Create database manually if needed
createdb cofars_ecommerce
```

### Port Already in Use
```bash
# Backend (3001)
lsof -ti:3001 | xargs kill -9

# Frontend (3000)
lsof -ti:3000 | xargs kill -9
```

### Prisma Issues
```bash
# Reset database
npx prisma migrate reset

# Regenerate client
npx prisma generate
```

### ML Model Not Found
Ensure the trained model exists at:
`../../results/models/best_model.pt`

If not, train the model first:
```bash
cd ../..
python src/train.py
```

## Development

### Backend Development
```bash
cd backend
npm run start:dev  # Hot reload enabled
```

### Frontend Development
```bash
cd frontend
npm run dev  # Hot reload enabled
```

### Database Management
```bash
# Open Prisma Studio (GUI)
npx prisma studio

# Create new migration
npx prisma migrate dev --name migration_name

# View database
psql -d cofars_ecommerce
```

## Production Deployment

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
# Serve dist/ folder with nginx
```

## Environment Variables

### Backend (.env)
```
DATABASE_URL="postgresql://user:pass@localhost:5432/cofars_ecommerce"
JWT_SECRET="your-secret-key"
JWT_EXPIRES_IN="7d"
ML_MODEL_PATH="../../results/models/best_model.pt"
PORT=3001
```

### Frontend (.env)
```
VITE_API_URL="http://localhost:3001"
```

## Support

For issues or questions:
1. Check the logs in `backend/logs/`
2. Review API documentation at `/api/docs`
3. Check database with Prisma Studio

## License

MIT
