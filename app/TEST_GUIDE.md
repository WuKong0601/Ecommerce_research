# 🧪 CoFARS E-commerce - Test Guide

## ✅ Tất cả các cải tiến đã hoàn thành!

---

## 🚀 Quick Start

### URLs:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:3001
- **API Docs**: http://localhost:3001/api/docs

---

## 🔐 Test Accounts

### 1. Cold-Start User (Test Recommendations cho user mới)
```
Email: test@cofars.com
Password: test123
Segment: COLD_START (0 interactions)
```

### 2. Admin Account (Full access)
```
Email: admin@cofars.com
Password: admin123
Segment: POWER
```

### 3. Real Users từ Dataset
```
Email: user{customer_id}@cofars.com
Password: password123

Ví dụ:
- user83@cofars.com
- user100@cofars.com
- user167@cofars.com
```

---

## 🎯 Test Scenarios

### ✅ Test 1: Product Images Match Categories
**Mục đích**: Kiểm tra ảnh sản phẩm phù hợp với category

**Steps**:
1. Truy cập http://localhost:3000/products
2. Browse qua các products
3. Click vào products thuộc categories khác nhau:
   - Dụng cụ nhà bếp → Xem ảnh kitchen/cookware
   - Đèn & thiết bị chiếu sáng → Xem ảnh lighting
   - Ngoài trời & sân vườn → Xem ảnh outdoor

**Expected Result**:
- ✅ Ảnh phù hợp với category của product
- ✅ Ảnh từ Unsplash, chất lượng cao
- ✅ Nếu ảnh lỗi → tự động fallback sang placeholder

---

### ✅ Test 2: Cold-Start User Recommendations
**Mục đích**: Verify recommendations hoạt động cho user mới (chưa có interaction history)

**Steps**:
1. Login: `test@cofars.com` / `test123`
2. Vào trang Products: http://localhost:3000/products
3. Quan sát header và products

**Expected Result**:
- ✅ Header hiển thị: "Showing personalized recommendations first"
- ✅ Top products có badge "Recommended" với icon Sparkles
- ✅ Recommendations dựa trên context hiện tại (time + day)
- ✅ Products được sort: Recommended → Other products

**Verify**:
```
- Badge "Recommended" chỉ hiện trên top products
- Khi search hoặc filter category → badge biến mất
- Recommendations thay đổi theo thời gian trong ngày
```

---

### ✅ Test 3: Context-Aware Recommendations
**Mục đích**: Test Context Selector và recommendations theo context

**Steps**:
1. Login với bất kỳ account nào
2. Vào: http://localhost:3000/recommendations
3. Thử các contexts khác nhau:
   - **Morning + Weekday**
   - **Afternoon + Weekend**
   - **Evening + Weekday**
   - **Late Night + Weekend**

**Expected Result**:
- ✅ Context Selector hiển thị đầy đủ options
- ✅ Recommendations thay đổi khi chọn context khác
- ✅ User segment badge hiển thị (COLD_START/REGULAR/POWER)
- ✅ "How It Works" section giải thích CoFARS-Sparse
- ✅ Context statistics hiển thị đúng

**Verify Different Contexts**:
```
Morning Weekday vs Evening Weekend:
- Recommendations khác nhau
- Products phù hợp với context
- Scores được tính dựa trên interaction patterns
```

---

### ✅ Test 4: Product Detail với Reviews
**Mục đích**: Verify reviews hiển thị đầy đủ

**Steps**:
1. Vào Products page
2. Click vào bất kỳ product nào
3. Scroll xuống Reviews section

**Expected Result**:
- ✅ Reviews section hiển thị với icon MessageSquare
- ✅ Mỗi review có:
  - User name
  - Rating stars (1-5)
  - Comment text
  - Created date
- ✅ Review count hiển thị: "Customer Reviews (X)"
- ✅ Empty state nếu chưa có reviews

---

### ✅ Test 5: Products Page Personalization
**Mục đích**: Verify products được sort theo recommendations

**Steps**:
1. Login với regular/power user (có interaction history)
2. Vào: http://localhost:3000/products
3. Không search, không filter

**Expected Result**:
- ✅ Message: "Showing personalized recommendations first"
- ✅ Top 10-20 products có badge "Recommended"
- ✅ Recommendations dựa trên:
  - User interaction history
  - Similar users' behavior
  - Current context
- ✅ Remaining products hiển thị sau recommendations

**Test Filtering**:
```
1. Search "lamp" → Badge biến mất, chỉ show filtered results
2. Select category → Badge biến mất, chỉ show category products
3. Clear filters → Badge xuất hiện lại
```

---

## 🧠 Understanding Recommendations

### Cold-Start Users (1 interaction)
```
Strategy: Context-Based
- Lấy products phổ biến trong context hiện tại
- Dựa trên patterns của users khác
- Không cần interaction history

Example:
Morning Weekday → Kitchen products (breakfast items)
Evening Weekend → Entertainment products
```

### Regular Users (2-4 interactions)
```
Strategy: Collaborative Filtering
- Tìm similar users
- Filter theo context
- Recommend based on similar users' interactions

Example:
User A liked products [1, 2, 3]
User B liked products [1, 2, 4, 5]
→ Recommend products [4, 5] to User A
```

### Power Users (≥5 interactions)
```
Strategy: Advanced Personalization
- Full interaction history
- Context-aware scoring
- Cached recommendations
- Highest accuracy

Example:
User has 10+ interactions
→ Deep pattern analysis
→ Highly personalized recommendations
```

---

## 🎨 UI Features to Verify

### 1. Context Selector Component
```
Location: /recommendations page
Features:
- 4 time slot buttons với icons
- Weekday/Weekend toggle
- Current context display
- Explanation text
```

### 2. Product Cards
```
Features:
- Category-matched images
- Image fallback on error
- Price, rating, category info
- Hover effects
```

### 3. Recommendation Badges
```
Location: Products page (when authenticated)
Features:
- "Recommended" badge với Sparkles icon
- Only on top recommended products
- Disappears when filtering
```

### 4. Reviews Section
```
Location: Product detail page
Features:
- User avatars
- Rating stars
- Comment text
- Timestamps
- Empty state message
```

---

## 📊 Data Verification

### Check Database Stats:
```sql
-- Total products
SELECT COUNT(*) FROM "Product"; 
-- Expected: 11,746

-- Total users
SELECT COUNT(*) FROM "User";
-- Expected: 40,523

-- Total reviews
SELECT COUNT(*) FROM "Review";
-- Expected: 48,131

-- Total interactions
SELECT COUNT(*) FROM "UserInteraction";
-- Expected: 196,608

-- User segments distribution
SELECT segment, COUNT(*) 
FROM "User" 
GROUP BY segment;
-- Expected:
-- COLD_START: ~1
-- REGULAR: ~35,251
-- POWER: ~5,271
```

---

## 🐛 Troubleshooting

### Issue: Recommendations không hiển thị
**Solution**:
1. Check user đã login chưa
2. Verify user có segment (COLD_START/REGULAR/POWER)
3. Check console logs cho errors
4. Verify API endpoint: GET /recommendations/for-you

### Issue: Images không load
**Solution**:
1. Check internet connection (Unsplash cần internet)
2. Verify fallback placeholder hiển thị
3. Check browser console cho CORS errors

### Issue: Context Selector không work
**Solution**:
1. Verify user đã login
2. Check API params: timeSlot & isWeekend
3. Verify endpoint: GET /recommendations/context-aware

---

## ✅ Success Criteria

Hệ thống hoạt động đúng khi:

1. ✅ **Images**: Ảnh phù hợp với category, có fallback
2. ✅ **Cold-Start**: User mới nhận được recommendations
3. ✅ **Context-Aware**: Recommendations thay đổi theo context
4. ✅ **Products Page**: Recommendations xếp lên đầu với badges
5. ✅ **Reviews**: Hiển thị đầy đủ cho mỗi product
6. ✅ **Sorting**: Recommended products → Other products
7. ✅ **Filtering**: Badges biến mất khi search/filter
8. ✅ **All Segments**: COLD_START, REGULAR, POWER đều có recommendations

---

## 🎊 Demo Flow

### Complete Demo Scenario:
```
1. Start: http://localhost:3000

2. Register new account:
   - Email: demo@test.com
   - Password: demo123
   - → Becomes COLD_START user

3. Browse Products:
   - See "Showing personalized recommendations first"
   - Top products have "Recommended" badges
   - Click product → See reviews

4. Go to Recommendations page:
   - Try different contexts
   - See recommendations change
   - Understand "How It Works"

5. Interact with products:
   - Add to cart
   - View details
   - Read reviews

6. Check personalization:
   - Products page shows personalized order
   - Recommendations improve over time
```

---

## 📞 Support

**Tất cả tính năng đã hoàn thành 100%!**

- ✅ Product images theo category
- ✅ Recommendations cho cold-start users
- ✅ Context-aware recommendations
- ✅ Products page với personalization
- ✅ Reviews display
- ✅ Full UI implementation

**Happy Testing! 🎉**
