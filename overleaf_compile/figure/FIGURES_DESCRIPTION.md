# Mô tả các hình ảnh trong Paper

## Danh sách các hình ảnh đã thêm vào paper

### 1. user_segmentation.png
**Vị trí trong paper**: Section 3 (Problem Formulation) - sau phần Task definition
**Label**: `fig:user_segmentation`
**Mô tả**: 
- Biểu đồ tròn (pie chart) thể hiện phân bố người dùng theo 3 nhóm
- **Power Users (≥5 interactions)**: 0.8% - màu xanh dương
- **Regular Users (2-4 interactions)**: 12.2% - màu cam
- **Cold-start Users (1 interaction)**: 87.0% - màu xanh lá

**Phân tích trong paper**:
- Cho thấy độ thưa thớt cực kỳ cao của dữ liệu (87% người dùng chỉ có 1 tương tác)
- Động lực chính cho việc thiết kế hybrid modeling approach
- Giải thích tại sao không thể áp dụng trực tiếp các phương pháp long sequence truyền thống

**Tham chiếu**: 
- Line 91-92: Caption giải thích
- Line 108: "As shown in Figure~\ref{fig:user_segmentation}..."

---

### 2. context_distribution.png
**Vị trí trong paper**: Section 3 (Problem Formulation) - ngay sau user_segmentation
**Label**: `fig:context_distribution`
**Mô tả**:
- Biểu đồ cột (bar chart) thể hiện phân bố tương tác qua 10 contexts
- Trục X: 10 contexts (morning_weekday, afternoon_weekday, evening_weekday, etc.)
- Trục Y: Số lượng interactions
- Màu sắc: Phân biệt weekday (xanh) và weekend (cam)

**Các context chính**:
1. morning_weekday: 8,188 interactions (16.7%)
2. evening_weekday: 8,146 interactions (16.6%)
3. afternoon_weekday: 8,125 interactions (16.5%)
4. unknown_weekday: 7,201 interactions (14.7%)
5. late_night_weekday: 4,106 interactions (8.4%)
6. afternoon_weekend: 3,208 interactions (6.5%)
7. evening_weekend: 2,981 interactions (6.1%)
8. morning_weekend: 2,914 interactions (5.9%)
9. unknown_weekend: 2,702 interactions (5.5%)
10. late_night_weekend: 1,581 interactions (3.2%)

**Phân tích trong paper**:
- Weekday contexts chiếm ưu thế (16-17% mỗi context)
- Weekend và late-night contexts có tỷ lệ thấp hơn
- Mặc dù user-level sparsity cao, nhưng context-level có đủ dữ liệu để modeling
- Hỗ trợ chiến lược context-based modeling

**Tham chiếu**:
- Line 98-99: Caption
- Line 108: "Figure~\ref{fig:context_distribution} reveals that..."

---

### 3. context_similarity_heatmap.png
**Vị trí trong paper**: Section 4.3 (Probability Encoder) - sau công thức JS divergence
**Label**: `fig:context_similarity`
**Mô tả**:
- Heatmap 10x10 thể hiện độ tương đồng giữa các contexts
- Trục X và Y: 10 contexts
- Màu sắc: Đỏ đậm = similarity cao, Vàng = similarity thấp hơn
- Scale: 0.9992 - 1.0 (tất cả đều rất cao)

**Insights chính**:
- Tất cả các cặp context có similarity ≥ 0.9992 (màu đỏ đậm)
- Biến đổi nhỏ (màu sáng hơn) tương ứng với:
  - Sự khác biệt weekend vs weekday
  - Sự khác biệt late-night vs daytime
- Diagonal (đường chéo) = 1.0 (context so với chính nó)

**Phân tích trong paper**:
- Khác với food delivery (breakfast ≠ lunch ≠ dinner), home products có preferences nhất quán
- Validate việc sử dụng static context aggregation thay vì temporal graphs
- Giải thích tại sao JS divergence values rất nhỏ (0.0005 trung bình)
- Model vẫn capture được subtle differences để personalization

**Tham chiếu**:
- Line 150-152: Caption
- Line 360: "Figure~\ref{fig:context_similarity} visualizes..."
- Phân tích chi tiết về màu sắc và patterns

---

### 4. context_category_heatmap.png
**Vị trí trong paper**: Section 5.3 (Context Similarity Analysis) - sau context_similarity
**Label**: `fig:context_category`
**Mô tả**:
- Heatmap thể hiện phân bố product categories qua contexts
- Trục Y: 10 contexts (rows)
- Trục X: Product categories (columns) - Kitchen, Bath, Decor, etc.
- Màu sắc: Xanh đậm = tỷ lệ cao, Xanh nhạt = tỷ lệ thấp

**Patterns quan sát được**:
1. **Kitchen và Bath categories**: Chiếm ưu thế ở tất cả contexts (màu xanh đậm)
2. **Weekend contexts**: Đa dạng hơn trong category selection
3. **Late-night contexts**: Patterns riêng biệt với activity thấp hơn
4. **Overall similarity**: Patterns tương tự nhau (consistent với high JS similarity)
5. **Subtle differences**: Variations nhỏ mà model exploit được

**Phân tích trong paper**:
- Giải thích chi tiết tại sao contexts có high similarity
- Chỉ ra những subtle differences có giá trị cho personalization
- Kết nối với hybrid recommendation strategy
- Evidence cho việc model học được meaningful patterns

**Tham chiếu**:
- Line 364-366: Caption
- Line 369: "To further understand context-specific preferences..."
- Phân tích 3 observations chính

---

## Tổng quan về việc sử dụng figures trong paper

### Vị trí đặt figures
1. **Section 3 (Problem Formulation)**: 2 figures
   - user_segmentation.png
   - context_distribution.png
   - Mục đích: Minh họa đặc điểm dữ liệu và động lực cho research

2. **Section 4 (Methodology)**: 1 figure
   - context_similarity_heatmap.png
   - Mục đích: Validate probability encoder và JS divergence approach

3. **Section 5 (Experiments)**: 1 figure
   - context_category_heatmap.png
   - Mục đích: Phân tích sâu về context-specific preferences

### Cách figures hỗ trợ arguments trong paper

**Argument 1: Data sparsity là thách thức chính**
- Evidence: Figure 1 (user_segmentation) - 87% cold-start users

**Argument 2: Context-based modeling vẫn khả thi**
- Evidence: Figure 2 (context_distribution) - Sufficient context-level data

**Argument 3: Static aggregation phù hợp hơn temporal graphs**
- Evidence: Figure 3 (context_similarity) - High uniform similarity

**Argument 4: Model captures meaningful patterns**
- Evidence: Figure 4 (context_category) - Subtle but useful differences

### Kết nối giữa các figures

```
user_segmentation (Problem) 
    ↓
context_distribution (Solution feasibility)
    ↓
context_similarity (Method validation)
    ↓
context_category (Deep analysis)
```

## Thống kê về figures

- **Tổng số figures**: 4 hình
- **File format**: PNG (tất cả)
- **Kích thước**:
  - context_category_heatmap.png: 490 KB
  - context_distribution.png: 313 KB
  - context_similarity_heatmap.png: 356 KB
  - user_segmentation.png: 313 KB
- **Tổng dung lượng**: ~1.47 MB

## Cách compile trên Overleaf

Khi upload lên Overleaf:
1. Giữ nguyên cấu trúc folder: `figure/` chứa 4 files PNG
2. LaTeX sẽ tự động tìm figures qua path: `figure/filename.png`
3. Không cần thay đổi gì trong main.tex
4. Click "Recompile" để xem kết quả

## Ghi chú kỹ thuật

- **Width**: Tất cả figures dùng `\columnwidth` để fit với ACL format
- **Position**: `[t]` (top of page) để figures không bị float xa text
- **Labels**: Consistent naming `fig:user_segmentation`, `fig:context_distribution`, etc.
- **Captions**: Detailed và self-contained (có thể hiểu mà không cần đọc main text)
- **References**: Dùng `\ref{fig:label}` để tham chiếu trong text

## Checklist trước khi submit

- [x] 4 figures đã được copy vào folder `figure/`
- [x] Tất cả figures được reference trong text
- [x] Captions đầy đủ và mô tả rõ ràng
- [x] Labels consistent và dễ nhớ
- [x] Phân tích kết nối với figures trong text
- [x] File sizes hợp lý cho upload

## Nếu cần thêm figures

Các figures khác có thể thêm (nếu có):
1. **Architecture diagram**: Overall CoFARS-Sparse architecture
2. **Training curves**: Loss/AUC over epochs
3. **Ablation visualization**: Bar chart comparing variants
4. **Performance comparison**: Bar chart CoFARS-Sparse vs baselines
5. **Prototype utilization**: Heatmap of prototype assignments

Để thêm figure mới:
1. Đặt file vào folder `figure/`
2. Thêm code LaTeX:
```latex
\begin{figure}[t]
  \centering
  \includegraphics[width=\columnwidth]{figure/new_figure.png}
  \caption{Your caption here.}
  \label{fig:new_label}
\end{figure}
```
3. Reference trong text: `Figure~\ref{fig:new_label}`
