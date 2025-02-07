import re
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from pyvis import network as net
import plotly.graph_objects as go
from matplotlib.cm import get_cmap
from scipy.stats import gaussian_kde
from tdamapper.plot import MapperLayoutInteractive
from utils.utils_v3 import rotate_x, rotate_y, rotate_z

class MapperPlotter:
    def __init__(self, mapper_info, rbind_data, cmap='jet', seed=10, width=400, height=400, iterations=30, dim=3, range_lst=None):
        self.mapper_info = mapper_info
        self.rbind_data = rbind_data
        self.cmap = cmap
        self.iterations = iterations
        self.seed = seed
        self.width = width
        self.height = height
        self.mapper_plot = None
        self.full_info = None
        self.filtered_info = None
        self.outlier_info = None
        self.dim = dim
        self.range_lst = range_lst

    def create_mapper_plot(self, choose, encoded_label, avg=False):
        if avg:
            self.rbind_data['color_for_plot'] = self.rbind_data[choose].astype(float)
        else:
            self.rbind_data['color_for_plot'] = pd.factorize(self.rbind_data[choose])[0]
        self.mapper_plot = MapperLayoutInteractive(
            self.mapper_info,
            colors=self.rbind_data['color_for_plot'].to_numpy(),
            cmap=self.cmap,
            agg=encoded_label,
            dim=self.dim,
            iterations=self.iterations,
            seed=self.seed,
            width=self.width,
            height=self.height
        )
        print("Mapper plot created.")

        return self.mapper_plot

    def extract_data(self, rx=False, ry=False, rz=False):
        x = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['x']
        y = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['y']
        if self.dim==3:
            z = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['z']
            threeDimData = pd.DataFrame({'x': x, 'y': y, 'z': z})
        else:
            threeDimData = pd.DataFrame({'x': x, 'y': y})
        
        data_tuple = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['text']
        data = []
        for item in data_tuple:
            try:
                match = float(re.search(r'color: ([\d.-]+)', item).group(1))
            except Exception as e:
                match = float(re.search(r'color: ([\d.]+)', item).group(1))

            color = float(match)
            node = int(re.search(r'node: (\d+)', item).group(1))
            size = int(re.search(r'size: (\d+)', item).group(1))
            data.append({'color': color, 'node': node, 'size': size})
        component_info = pd.DataFrame(data)
        
        self.full_info = pd.concat([component_info, threeDimData], axis=1)
        mp_content_origin = vars(self.mapper_plot._MapperLayoutInteractive__graph)['_node']
        mp_content = pd.DataFrame.from_dict(mp_content_origin, orient='index').reset_index()
        mp_content.rename(columns={'index': 'node'}, inplace=True)
        
        self.full_info = pd.merge(self.full_info, mp_content, on=['node', 'size'], how='inner')
        self.full_info["ratio"] = self.full_info["color"] / self.full_info["size"]
        print("Data extracted.")
        
        if rx:
            self.full_info[['x', 'y', 'z']] = rotate_x(self.full_info[['x', 'y', 'z']].to_numpy(), rx)
        if ry:
            self.full_info[['x', 'y', 'z']] = rotate_y(self.full_info[['x', 'y', 'z']].to_numpy(), ry)
        if rz:
            self.full_info[['x', 'y', 'z']] = rotate_z(self.full_info[['x', 'y', 'z']].to_numpy(), rz)

        find_connected_points(self.full_info)
        self.outlier_info = self.full_info[self.full_info['outlier'] == True]
        self.filtered_info = self.full_info[self.full_info['outlier'] == False]
        
        return self.filtered_info, self.outlier_info

    def map_colors(self, choose, size=0, threshold=5):

        # 過濾大小的資料點
        df = self.filtered_info[(self.filtered_info['size'] > size)]

        if self.range_lst is not None:
            df = df[(df['x'] > self.range_lst[0]) & (df['y'] < self.range_lst[2]) & 
                    (df['x'] < self.range_lst[1]) & (df['y'] > self.range_lst[3])]
        
        # 為了計算kde需要個別處理，以免kde的值表現不好
        self.full_info = self.full_info[(self.full_info['x'] > self.range_lst[0]) & (self.full_info['y'] < self.range_lst[2]) &
                                      (self.full_info['x'] < self.range_lst[1]) & (self.full_info['y'] > self.range_lst[3])]

        # 計算每個標籤的出現次數
        category_counts = self.rbind_data[choose].value_counts()
        # 篩選出現次數大於 threshold 的標籤
        filtered_categories = category_counts[category_counts > threshold].index
        # 取得唯一值並過濾不需要的類別
        unique_values = self.rbind_data.reset_index()[[choose, 'color_for_plot']].drop_duplicates()
        unique_values = unique_values[unique_values[choose].isin(filtered_categories)]

        # 更新 unique_categories 和 color_mapping_fixed
        unique_categories = filtered_categories.tolist()
        color_palette = get_cmap("tab20", len(unique_categories))
        color_mapping_fixed = {category: color_palette(i) for i, category in enumerate(unique_categories)}

        # 合併資料
        df = df.merge(unique_values, left_on='color', right_on='color_for_plot', how='left')
        
        # 處理 category 類型
        if df[choose].dtype.name == 'category':
            df['color_for_plot_fixed'] = df[choose].astype(str).map(color_mapping_fixed)
        else:
            if isinstance(df[choose], pd.Series):
                df['color_for_plot_fixed'] = df[choose].map(color_mapping_fixed)
            else:
                df['color_for_plot_fixed'] = df[choose].astype(str).map(color_mapping_fixed)

        # 為threshold過濾掉的類別設定默認顏色
        default_color = (0.5, 0.5, 0.5, 1) 
        df['color_for_plot_fixed'] = df['color_for_plot_fixed'].apply(
            lambda x: x if pd.notna(x) else default_color
        )

        self.filtered_info = df
        self.color_palette = color_mapping_fixed
        self.unique_categories = unique_categories  # 保存篩選後的 categories
        print("Colors mapped using predefined mapping.")

    def plot(self, choose, avg=None, save_path=None, set_label=False, size=100, anchor=1):
        
        # self.filtered_info = self.filtered_info[self.filtered_info['size'] > size]
        
        # 過濾掉無效的顏色資料
        valid_data = self.filtered_info.dropna(subset=['color_for_plot_fixed'])
        clipped_size = np.clip(valid_data['size'], None, size)

        plt.figure(figsize=(15, 12))
        
        if avg:
            color = self.filtered_info['color']
        else:
            # 確保 'color_for_plot_fixed' 是有效的顏色格式
            color = [tuple(c) if isinstance(c, (list, tuple)) else c for c in valid_data['color_for_plot_fixed']]

        scatter = plt.scatter(
            self.filtered_info['x'], self.filtered_info['y'],
            c=color,
            edgecolors='black',
            linewidths=0.5,
            s=clipped_size,
            marker='o',
            alpha=0.9
        )

        node_positions = {row['node']: (row['x'], row['y']) for _, row in self.filtered_info.iterrows()}
        graph = vars(self.mapper_plot._MapperLayoutInteractive__graph)
        edges = graph['edges']
        for edge in edges:
            if edge[0] in node_positions and edge[1] in node_positions:
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
                plt.plot(x_coords, y_coords, color='grey', alpha=0.5, linewidth=0.5, zorder=0)

        if set_label:
            if avg:
                colorbar = plt.colorbar(scatter, ax=plt.gca(), orientation='vertical', pad=0.02)
            else:
                handles = [
                    plt.Line2D(
                        [0], [0],
                        marker='o',
                        color=self.color_palette[name],
                        markersize=10,
                        label=name
                    ) for name in self.unique_categories
                ]
                ax_position = plt.gca().get_position()
                plt.legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(anchor, 1))

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Mapper plot')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def plot_dens(self, choose, avg=None, save_path=None, set_label=False, size=100, minimum_lst=None):
        # 作圖用filter，密度則使用原始進行計算(使用size篩選前的資料)
        clipped_size = np.clip(self.filtered_info['size'], None, size)
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
        color = self.filtered_info['color'] if avg else [tuple(c) if isinstance(c, (list, tuple)) else c for c in self.filtered_info['color_for_plot_fixed']]
        # 為了讓兩個圖的 x 軸刻度標籤一致，固定刻度範圍
        ticks = np.arange(self.range_lst[0], self.range_lst[1] + 0.025, 0.025)
        x_min, x_max = self.filtered_info["x"].min(), self.filtered_info["x"].max()

        # 拓樸圖
        scatter = ax[0].scatter(
            self.filtered_info['x'], self.filtered_info['y'],
            c=color,
            edgecolors='black',
            linewidths=0.5,
            s=clipped_size,
            marker='o',
            alpha=0.9
        )

        node_positions = {row['node']: (row['x'], row['y']) for _, row in self.filtered_info.iterrows()}
        graph = vars(self.mapper_plot._MapperLayoutInteractive__graph)
        edges = graph['edges']
        for edge in edges:
            if edge[0] in node_positions and edge[1] in node_positions:
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
                ax[0].plot(x_coords, y_coords, color='grey', alpha=0.5, linewidth=0.5, zorder=0)

        if set_label:
            if avg:
                colorbar = plt.colorbar(scatter, ax=ax[0], orientation='vertical', pad=0.02)
            else:
                handles = [
                    plt.Line2D(
                        [0], [0],
                        marker='o',
                        color=self.color_palette[name],
                        markersize=10,
                        label=name
                    ) for name in self.unique_categories
                ]
                ax[0].legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(1, 1))
        
        ax[0].set_xlabel('')
        ax[0].set_ylabel('Y')
        ax[0].set_title('Mapper plot')
        ax[0].grid(True)
        ax[0].set_xlim(x_min, x_max)
        ax[0].set_xticks(ticks)
        ax[0].tick_params(axis='x', labelbottom=False)  # 隱藏第一張圖的 x 軸刻度標籤 

        # 使用 sns.kdeplot 繪製密度圖並提取數據
        kdeplot = sns.kdeplot(
            x=self.full_info["x"],
            weights=self.full_info["ratio"],
            fill=False,
            # cmap="viridis",
            ax=ax[1],
            bw_adjust=.3,
        )
            
        # 從 kdeplot 中提取數據並找出最低點
        line = kdeplot.lines[0]
        x_vals, kde_vals = line.get_data()
        
        # 透過minimum_lst篩選 filtered_info['x']，如果篩選資料太少，會無法計算kde
        if minimum_lst:
            mask = (x_vals >= minimum_lst[0]) & (x_vals <= minimum_lst[1])
            x_vals = x_vals[mask]
            kde_vals = kde_vals[mask]

        min_idx = np.argmin(kde_vals)
        min_x = x_vals[min_idx]
        min_y = kde_vals[min_idx]

        # 在拓樸圖和密度圖上標出最低點的垂直線
        ax[0].axvline(x=min_x, color='#598e9c', linestyle='--', label=f"x={min_x:.4f}")
        ax[1].axvline(x=min_x, color='#598e9c', linestyle='--', label=f"x={min_x:.4f}")

        # 在密度圖上顯示最低點
        ax[1].scatter([min_x], [min_y], color="red", label=f"Min Density ({min_x:.4f}, {min_y:.4f})")
        ax[1].legend()

        # 設置密度圖的標籤和格線
        ax[1].set_xlabel("X")
        ax[1].set_ylabel("Density")
        ax[1].grid(True)
        ax[1].set_xlim(x_min, x_max)
        ax[1].set_xticks(ticks)

        # 自動調整佈局
        plt.tight_layout()

        # 如果提供保存路徑，則保存圖片，否則顯示圖片
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def update_colors(self, choose, threshold=5):
        """
        更新顏色對應，不重新跑 create_mapper_plot
        """
        # 計算類別出現次數
        category_counts = self.rbind_data[choose].value_counts()

        # 篩選出現次數大於 threshold 的標籤
        filtered_categories = category_counts[category_counts > threshold].index

        # 取得唯一值並過濾不需要的類別
        unique_values = self.rbind_data[[choose, 'color_for_plot']].drop_duplicates()
        unique_values = unique_values[unique_values[choose].isin(filtered_categories)]

        # 設定新的顏色對應
        unique_categories = filtered_categories.tolist()
        color_palette = get_cmap("tab20", len(unique_categories))
        color_mapping_fixed = {category: color_palette(i) for i, category in enumerate(unique_categories)}

        # 更新 `self.filtered_info`
        self.filtered_info = self.filtered_info.merge(unique_values, left_on='color', right_on='color_for_plot', how='left')

        # 確保 `color_for_plot_fixed` 正確映射
        if self.filtered_info[choose].dtype.name == 'category':
            self.filtered_info['color_for_plot_fixed'] = self.filtered_info[choose].astype(str).map(color_mapping_fixed)
        else:
            if isinstance(self.filtered_info[choose], pd.Series):
                self.filtered_info['color_for_plot_fixed'] = self.filtered_info[choose].map(color_mapping_fixed)
            else:
                self.filtered_info['color_for_plot_fixed'] = self.filtered_info[choose].astype(str).map(color_mapping_fixed)

        # 為 threshold 過濾掉的類別設定默認顏色
        default_color = (0.5, 0.5, 0.5, 1)
        self.filtered_info['color_for_plot_fixed'] = self.filtered_info['color_for_plot_fixed'].apply(
            lambda x: x if pd.notna(x) else default_color
        )

        # 更新 color_palette
        self.color_palette = color_mapping_fixed
        self.unique_categories = unique_categories  # 保存篩選後的 categories
        print("Colors updated without rerunning create_mapper_plot.")

    def plot_3d(self, choose, avg=None, save_path=None, set_label=False, size=100):
        # 過濾掉無效的顏色資料
        # self.full_info = self.full_info.dropna(subset=['color_for_plot_fixed'])

        clipped_size = np.clip(self.full_info['size'], None, size)

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        if avg:
            color = self.full_info['color']
        else:
            # 確保 'color_for_plot_fixed' 是有效的顏色格式
            color = [tuple(c) if isinstance(c, (list, tuple)) else c for c in self.full_info['color_for_plot_fixed']]

        scatter = ax.scatter(
            self.full_info['x'], self.full_info['y'], self.full_info['z'],
            c=color,
            edgecolors='black',
            linewidths=0.5,
            s=clipped_size,
            marker='o',
            alpha=0.7
        )

        node_positions = {row['node']: (row['x'], row['y'], row['z']) for _, row in self.full_info.iterrows()}
        graph = vars(self.mapper_plot._MapperLayoutInteractive__graph)
        edges = graph['edges']
        for edge in edges:
            if edge[0] in node_positions and edge[1] in node_positions:
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0]]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1]]
                z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2]]
                ax.plot(x_coords, y_coords, z_coords, color='grey', alpha=0.5, linewidth=0.5, zorder=0)

        if set_label:
            if avg:
                colorbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02)
            else:
                handles = [
                    plt.Line2D(
                        [0], [0],
                        marker='o',
                        color=self.color_palette[name],
                        markersize=10,
                        label=name
                    ) for name in self.unique_categories
                ]
                ax.legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(1, 1))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Mapper 3D plot')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D plot saved to {save_path}")
        else:
            plt.show()

    def plot_3d_pvis(self, path_name='./MapperGraphs/o4i7interactive', range_lst=None, size=0):

        G = nx.Graph()

        df = self.full_info[(self.full_info['size'] > size)]

        if range_lst is not None:
            df = df[(df['x'] > range_lst[0]) & (df['y'] < range_lst[2]) & 
                    (df['x'] < range_lst[1]) & (df['y'] > range_lst[3])]

        for i, row in df.iterrows():
            G.add_node(row['node'], size=row['size'], color=row['color'])
        for i, row1 in df.iterrows():
            for j, row2 in df.iterrows():
                if i < j and set(row1['ids']).intersection(set(row2['ids'])):
                    G.add_edge(row1['node'], row2['node'])

        g = net.Network(height='1500px', width='100%', heading='')

        g.add_nodes(
            [node for node in G.nodes()],
            value=[G.nodes[node]['size'] for node in G.nodes()],
            title=[f"Node {node}" for node in G.nodes()],
            color=[f"rgb({255 - int(abs(G.nodes[node]['color']) * 50)}, 150, 150)" for node in G.nodes()]
        )
        g.add_edges([(source, target) for source, target in G.edges()])
        g.set_options("""
        var options = {
            "physics": {
                "stabilization": {
                "enabled": true,
                "iterations": 200
                },
                "barnesHut": {
                "gravitationalConstant": -2000,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.04,
                "damping": 0.5
                },
                "minVelocity": 0.5
                },
            "background": {
                "color": "black"
            }
            }
        """)

        g.write_html(f'{path_name}.html')

def find_connected_points(df):
    """
    找出所有互相連接的點，基於 size 最大的點作為起點。
    """
    # 1. 找到 size 最大的點的索引
    max_size_idx = df['size'].idxmax()
    visited = set()  # 存儲已訪問的索引
    to_visit = {max_size_idx}  # 初始化待訪問的節點集合

    # 2. 使用廣度優先搜索 (BFS) 查找所有連接的點
    while to_visit:
        current_idx = to_visit.pop()
        if current_idx in visited:
            continue
        visited.add(current_idx)
        # 查找與當前點有交集的所有點
        current_ids = set(df.loc[current_idx, "ids"])
        connected_indices = df[df['ids'].apply(lambda x: bool(current_ids & set(x)))].index
        # 將未訪問的索引加入待訪問集合
        to_visit.update(set(connected_indices) - visited)

    # 3. 標記所有點是否為 outlier
    df["outlier"] = ~df.index.isin(visited)

def plot_mca(mca, data):

    col_coordinate = mca.column_coordinates(data)

    col_coordinate['dummy_index'] = range(len(col_coordinate))  # 為每個欄位分配索引

    # Normalize the 'dummy_index' for mapping to colormap
    norm = plt.Normalize(vmin=col_coordinate['dummy_index'].min(), vmax=col_coordinate['dummy_index'].max())
    cmap = cm.viridis  # 選擇漸層色盤，例如 viridis, plasma, inferno 等

    plt.figure(figsize=(20, 12))
    plt.gca().set_facecolor('white')

    # 繪製散點圖，顏色根據 dummy_index 映射
    scatter = plt.scatter(
        col_coordinate[0],  # X coordinate
        col_coordinate[1],  # Y coordinate
        c=col_coordinate['dummy_index'],  # 使用索引作為顏色映射
        cmap=cmap,
        alpha=0.7
    )

    # Adding labels with colors matching the scatter plot
    offset = 0.15
    for i, label in enumerate(col_coordinate.index):
        plt.text(
            col_coordinate.iloc[i, 0],  # X coordinate
            col_coordinate.iloc[i, 1] + offset,  # Y coordinate
            str(label),  # 假設 label 包含中文
            fontsize=10,
            ha='center', 
            va='center',
            color=cmap(norm(col_coordinate.iloc[i]['dummy_index'])),  # 文字顏色與點顏色一致
            # rotation=30
        )

    # Add colorbar
    # cbar = plt.colorbar(scatter)
    # cbar.set_label("Field Index (Gradient)", fontsize=12)

    # Axes lines and grid
    plt.axhline(0, color='grey', linestyle='--', linewidth=0.5)
    plt.axvline(0, color='grey', linestyle='--', linewidth=0.5)
    # plt.title("MCA", fontsize=16)
    plt.xlabel("Dimension 1", fontsize=12)
    plt.ylabel("Dimension 2", fontsize=12)
    plt.grid(True)

    plt.show()
    
def plot_labels(label_0, label_1, label_out, title="Label Visualization"):
    plt.figure(figsize=(10, 8))

    # 繪製 label_0 的點
    plt.scatter(label_0['x'], label_0['y'], c='blue', label='Label 0', alpha=0.6)

    # 繪製 label_1 的點
    plt.scatter(label_1['x'], label_1['y'], c='green', label='Label 1', alpha=0.6)

    # 繪製 label_0_outliers 的點
    plt.scatter(label_out['x'], label_out['y'], c='red', label='Outliers', alpha=0.8)

    # 標記圖表
    plt.title(title, fontsize=16)
    plt.xlabel('X', fontsize=14)
    plt.ylabel('Y', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()