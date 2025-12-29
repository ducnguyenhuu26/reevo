import numpy as np
import matplotlib.pyplot as plt

def visualize_heuristics(size=50):
    # 1. Tạo dữ liệu ngẫu nhiên (giống như ReEvo làm)
    nodes = np.random.rand(size, 2)
    dist_matrix = np.sqrt(np.sum((nodes[:, None, :] - nodes[None, :, :])**2, axis=-1))
    
    # 2. Hàm Heuristic Cơ bản (Vòng lặp 0)
    h_basic = 1 / (dist_matrix + 1e-6)
    np.fill_diagonal(h_basic, 0)
    
    # 3. Hàm Heuristic của bạn (Kết quả sau Iteration 7)
    # Tôi copy lại logic từ log của bạn
    base_score = 1 / (dist_matrix + 1e-6)
    penalization = np.exp(-dist_matrix / np.mean(dist_matrix))
    h_ai = (base_score ** 2) * penalization
    threshold = np.percentile(h_ai, 20)
    h_ai[h_ai < threshold] = 0
    np.fill_diagonal(h_ai, 0)

    # Vẽ hình
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    for i, ax, h, title in zip([0, 1], [ax1, ax2], [h_basic, h_ai], ["Heuristic Cơ Bản (1/d)", "Heuristic AI (Llama-3.3)"]):
        ax.scatter(nodes[:, 0], nodes[:, 1], c='red', zorder=5)
        
        # Vẽ các cạnh "đậm" nhất (promising edges)
        flat_h = h.flatten()
        top_indices = np.argsort(flat_h)[-size*2:] # Lấy top các cạnh hứa hẹn nhất
        
        for idx in top_indices:
            start_node, end_node = divmod(idx, size)
            if start_node != end_node:
                alpha = h[start_node, end_node] / np.max(h)
                ax.plot([nodes[start_node, 0], nodes[end_node, 0]], 
                        [nodes[start_node, 1], nodes[end_node, 1]], 
                        'b-', alpha=alpha * 0.8)
        ax.set_title(title)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig('tsp_comparison.png')
    print("Đã lưu hình ảnh so sánh vào file 'tsp_comparison.png'")
    plt.show()

if __name__ == "__main__":
    visualize_heuristics(50)