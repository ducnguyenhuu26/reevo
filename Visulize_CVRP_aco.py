import matplotlib.pyplot as plt
import numpy as np

def plot_cvrp(nodes, routes, title="CVRP Solution by Llama-3.3"):
    plt.figure(figsize=(10, 8))
    
    # Vẽ Depot (Kho hàng) ở vị trí đầu tiên (thường là node 0)
    plt.scatter(nodes[0, 0], nodes[0, 1], c='red', marker='s', s=100, label='Depot')
    
    # Vẽ các khách hàng
    plt.scatter(nodes[1:, 0], nodes[1:, 1], c='blue', alpha=0.6, label='Customers')

    # Mỗi route là một danh sách các node, ví dụ: [0, 5, 3, 2, 0]
    colors = plt.cm.rainbow(np.linspace(0, 1, len(routes)))
    
    for i, (route, color) in enumerate(zip(routes, colors)):
        route_coords = nodes[route]
        plt.plot(route_coords[:, 0], route_coords[:, 1], color=color, linewidth=2, label=f'Vehicle {i+1}')
        # Vẽ mũi tên chỉ hướng
        for j in range(len(route)-1):
            plt.arrow(route[j], route[j+1], ... ) # Đơn giản hóa bằng plt.plot ở trên

    plt.title(title)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.grid(True)
    plt.show()

# Giả lập dữ liệu để bạn hình dung
# nodes = np.load('path_to_your_coordinates.npy') 
# routes = [[0, 1, 4, 0], [0, 2, 3, 5, 0]]