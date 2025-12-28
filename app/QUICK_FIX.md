# 🔧 Quick Fix Guide

## Vấn đề hiện tại:

1. ✅ **Ảnh đã được cập nhật** - Chạy script thành công
2. ❌ **Recommendations không hiển thị** - API có thể đang trả về empty array

## Giải pháp nhanh:

### Bước 1: Restart Backend
Backend cần restart để apply code changes với logging:

```bash
cd app/backend
# Ctrl+C để stop backend hiện tại
npm run start:dev
```

### Bước 2: Refresh Frontend
```bash
# Refresh browser tại http://localhost:3000/recommendations
# Hoặc Ctrl+Shift+R (hard refresh)
```

### Bước 3: Check Backend Console
Sau khi refresh, xem backend console sẽ hiển thị logs:
```
Context-aware request: timeSlot=morning, isWeekend=false, contextId=0
User segment: COLD_START, userId: xxx
Using context-based recommendations for COLD_START user
getContextBasedRecommendations: contextId=0, limit=10
Found XX products with interactions in context 0
Returning XX recommendations
```

### Bước 4: Nếu vẫn không có data

Chạy script test để verify data có trong DB:
```bash
cd app/backend
python scripts\debug-recommendations.py
```

Expected output sẽ show:
- Total Interactions: 196,610
- Context 0: 32,752 interactions
- Top products cho mỗi context

## Debugging Steps:

### Check 1: Verify Backend Running
```
http://localhost:3001/api/docs
```
Should show Swagger API documentation

### Check 2: Test Login
```bash
POST http://localhost:3001/api/auth/login
Body: {
  "email": "test@cofars.com",
  "password": "test123"
}
```

Should return:
```json
{
  "user": {...},
  "access_token": "eyJhbGc..."
}
```

### Check 3: Test Recommendations API
```bash
GET http://localhost:3001/api/recommendations/context-aware?timeSlot=morning&isWeekend=false
Headers: Authorization: Bearer {token}
```

Should return array of products

## Common Issues:

### Issue 1: "No recommendations available"
**Cause**: API returning empty array
**Fix**: 
1. Check backend logs for errors
2. Verify contextId calculation
3. Check if products have isActive=true

### Issue 2: Images not showing
**Cause**: Script not run or internet connection
**Fix**:
```bash
cd app/backend
python scripts\update-product-images-by-category.py
```

### Issue 3: Unauthorized errors
**Cause**: JWT token expired or invalid
**Fix**:
1. Logout and login again
2. Clear browser localStorage
3. Check JWT_SECRET in backend .env

## Expected Behavior:

### For COLD_START User (test@cofars.com):
- Should see recommendations based on context popularity
- Morning Weekday → Kitchen products
- Evening Weekend → Entertainment products
- Recommendations change when selecting different contexts

### For POWER User (admin@cofars.com):
- Should see personalized recommendations
- Based on interaction history
- More accurate than cold-start

## Data Verification:

Run this to check if data exists:
```sql
-- Check interactions in context 0 (morning weekday)
SELECT COUNT(*) FROM "UserInteraction" WHERE "contextId" = 0;
-- Expected: ~32,752

-- Check active products
SELECT COUNT(*) FROM "Product" WHERE "isActive" = true;
-- Expected: 11,746
```

## Next Steps:

1. **Restart backend** với logging enabled
2. **Refresh frontend** và check recommendations page
3. **Watch backend console** để xem logs
4. **Report** what you see in the logs

Logs sẽ cho biết chính xác vấn đề ở đâu!
