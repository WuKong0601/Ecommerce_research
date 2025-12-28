#!/bin/bash

# Setup script for CoFARS E-commerce Backend

echo "🚀 Setting up CoFARS E-commerce Backend..."

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Generate Prisma client
echo "🔧 Generating Prisma client..."
npx prisma generate

# Create .env file if not exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your database credentials"
fi

# Create uploads directory
mkdir -p uploads

echo "✅ Backend setup complete!"
echo ""
echo "Next steps:"
echo "1. Update .env with your database credentials"
echo "2. Run: npx prisma migrate dev"
echo "3. Run: npm run prisma:seed (to seed sample data)"
echo "4. Run: npm run start:dev"
