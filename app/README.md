# CoFARS-Sparse E-commerce Application

Full-stack e-commerce application with context-aware recommendations powered by CoFARS-Sparse model.

## Architecture

### Backend (NestJS)
- **Port**: 3001
- **Database**: PostgreSQL with Prisma ORM
- **Features**:
  - JWT Authentication
  - RESTful API
  - Real-time recommendations using trained CoFARS-Sparse model
  - Context-aware product suggestions
  - User segmentation (Power, Regular, Cold-start)

### Frontend (React + TypeScript)
- **Port**: 3000
- **UI Framework**: TailwindCSS + shadcn/ui
- **State Management**: React Query + Zustand
- **Features**:
  - Modern, responsive design
  - Real-time context detection
  - Personalized recommendations
  - Shopping cart & checkout
  - User profile & order history

## Project Structure

```
app/
├── backend/                 # NestJS backend
│   ├── src/
│   │   ├── auth/           # Authentication module
│   │   ├── users/          # User management
│   │   ├── products/       # Product catalog
│   │   ├── orders/         # Order processing
│   │   ├── recommendations/ # ML recommendation service
│   │   ├── contexts/       # Context detection
│   │   └── common/         # Shared utilities
│   ├── prisma/             # Database schema
│   └── ml-models/          # Trained PyTorch models
└── frontend/               # React frontend
    ├── src/
    │   ├── components/     # Reusable components
    │   ├── pages/          # Page components
    │   ├── hooks/          # Custom hooks
    │   ├── services/       # API services
    │   ├── store/          # State management
    │   └── lib/            # Utilities
    └── public/             # Static assets
```

## Setup Instructions

### Prerequisites
- Node.js 18+
- PostgreSQL 14+
- Python 3.8+ (for ML model inference)

### Backend Setup
```bash
cd backend
npm install
npx prisma generate
npx prisma migrate dev
npm run start:dev
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Features

### User Features
- ✅ User registration & authentication
- ✅ Browse products by category
- ✅ Context-aware recommendations
- ✅ Shopping cart management
- ✅ Order placement & tracking
- ✅ User profile & preferences
- ✅ Order history

### Admin Features
- ✅ Product management
- ✅ Order management
- ✅ User analytics
- ✅ Recommendation performance metrics

### ML Integration
- ✅ Real-time context detection (time_slot + is_weekend)
- ✅ User segmentation (Power/Regular/Cold-start)
- ✅ Hybrid recommendation strategy
- ✅ Context-based product suggestions
- ✅ Similarity-based context enrichment

## Technology Stack

### Backend
- NestJS
- Prisma ORM
- PostgreSQL
- JWT Authentication
- Python (for ML inference)
- PyTorch

### Frontend
- React 18
- TypeScript
- TailwindCSS
- shadcn/ui
- React Query
- Zustand
- Axios
- React Router

## API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `POST /auth/refresh` - Refresh token

### Products
- `GET /products` - List products
- `GET /products/:id` - Get product details
- `GET /products/category/:category` - Products by category

### Recommendations
- `GET /recommendations/for-you` - Personalized recommendations
- `GET /recommendations/context-aware` - Context-based suggestions
- `GET /recommendations/similar/:productId` - Similar products

### Orders
- `POST /orders` - Create order
- `GET /orders` - User orders
- `GET /orders/:id` - Order details

### Cart
- `GET /cart` - Get cart
- `POST /cart/items` - Add to cart
- `PUT /cart/items/:id` - Update cart item
- `DELETE /cart/items/:id` - Remove from cart

## Environment Variables

### Backend (.env)
```
DATABASE_URL="postgresql://user:password@localhost:5432/cofars_ecommerce"
JWT_SECRET="your-secret-key"
JWT_EXPIRES_IN="7d"
ML_MODEL_PATH="../results/models/best_model.pt"
PYTHON_PATH="python"
```

### Frontend (.env)
```
VITE_API_URL="http://localhost:3001"
```

## Development

### Run Backend
```bash
cd backend
npm run start:dev
```

### Run Frontend
```bash
cd frontend
npm run dev
```

### Database Migrations
```bash
cd backend
npx prisma migrate dev --name migration_name
```

## Deployment

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

## License

MIT
