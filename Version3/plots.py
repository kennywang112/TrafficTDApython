import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.font_manager import FontProperties
from tdamapper.plot import MapperLayoutInteractive

class MapperPlotter:
    def __init__(self, mapper_info, rbind_data, cmap='jet', seed=10, width=400, height=400, iterations=30):
        self.mapper_info = mapper_info
        self.rbind_data = rbind_data
        self.cmap = cmap
        self.iterations = iterations
        self.seed = seed
        self.width = width
        self.height = height
        self.mapper_plot = None
        self.full_info = None

    def create_mapper_plot(self, choose, encoded_label, avg=False):
        if avg:
            self.rbind_data['color_for_plot'] = self.rbind_data[choose]
        else:
            self.rbind_data['color_for_plot'] = pd.factorize(self.rbind_data[choose])[0]
        self.mapper_plot = MapperLayoutInteractive(
            self.mapper_info,
            colors=self.rbind_data['color_for_plot'].to_numpy(),
            cmap=self.cmap,
            agg=encoded_label,
            dim=3,
            iterations=self.iterations,
            seed=self.seed,
            width=self.width,
            height=self.height
        )
        print("Mapper plot created.")

        return self.mapper_plot

    def extract_data(self):
        x = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['x']
        y = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['y']
        z = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['z']
        threeDimData = pd.DataFrame({'x': x, 'y': y, 'z': z})
        
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
        print("Data extracted.")

        return self.full_info

    def map_colors(self, choose, size=0, threshold=5, range_lst = None):
        # range_lst [x_min, x_max, y_min, y_max]
        # 過濾大小的資料點
        df = self.full_info[(self.full_info['size'] > size)]

        if range_lst is not None:
            # 車的分析適用
            df = df[(df['x'] > range_lst[0]) & (df['y'] < range_lst[2]) & (df['x'] < range_lst[1]) & (df['y'] > range_lst[3])]
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

        self.full_info = df
        self.color_palette = color_mapping_fixed
        self.unique_categories = unique_categories  # 保存篩選後的 categories
        print("Colors mapped using predefined mapping.")

    def plot(self, choose, avg=None, save_path=None, set_label=False, size=100):
        # 過濾掉無效的顏色資料
        # self.full_info = self.full_info.dropna(subset=['color_for_plot_fixed'])

        clipped_size = np.clip(self.full_info['size'], None, size)

        plt.figure(figsize=(15, 12))
        
        if avg:
            color = self.full_info['color']
        else:
            # 確保 'color_for_plot_fixed' 是有效的顏色格式
            color = [tuple(c) if isinstance(c, (list, tuple)) else c for c in self.full_info['color_for_plot_fixed']]

        scatter = plt.scatter(
            self.full_info['x'], self.full_info['y'],
            c=color,
            edgecolors='black',
            linewidths=0.5,
            s=clipped_size,
            marker='o',
            alpha=0.9
        )

        node_positions = {row['node']: (row['x'], row['y']) for _, row in self.full_info.iterrows()}
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
                plt.legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(1, 1))

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Mapper plot')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

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
    
    def plot_3d_interactive(self, choose, avg=None, save_path=None, set_label=False, size=100):
        # 過濾掉無效的顏色資料
        self.full_info = self.full_info.dropna(subset=['color_for_plot_fixed'])

        clipped_size = np.clip(self.full_info['size'], None, size)

        if avg:
            color = self.full_info['color']
        else:
            # 確保 'color_for_plot_fixed' 是有效的顏色格式
            color = [tuple(c) if isinstance(c, (list, tuple)) else c for c in self.full_info['color_for_plot_fixed']]

        # 建立 3D 散點圖
        scatter = go.Scatter3d(
            x=self.full_info['x'],
            y=self.full_info['y'],
            z=self.full_info['z'],
            mode='markers',
            marker=dict(
                size=clipped_size / 10,  # 將節點大小縮放以適應 Plotly
                color=color,
                opacity=0.8,
                line=dict(width=0.5, color='black')
            ),
            text=self.full_info['node'],  # 顯示節點 ID
            hoverinfo='text'
        )

        # 添加邊的資料
        node_positions = {row['node']: (row['x'], row['y'], row['z']) for _, row in self.full_info.iterrows()}
        graph = vars(self.mapper_plot._MapperLayoutInteractive__graph)
        edges = graph['edges']

        edge_traces = []
        for edge in edges:
            if edge[0] in node_positions and edge[1] in node_positions:
                x_coords = [node_positions[edge[0]][0], node_positions[edge[1]][0], None]
                y_coords = [node_positions[edge[0]][1], node_positions[edge[1]][1], None]
                z_coords = [node_positions[edge[0]][2], node_positions[edge[1]][2], None]

                edge_traces.append(
                    go.Scatter3d(
                        x=x_coords,
                        y=y_coords,
                        z=z_coords,
                        mode='lines',
                        line=dict(color='grey', width=2),
                        hoverinfo='none'
                    )
                )

        # 建立繪圖佈局
        layout = go.Layout(
            title='Mapper Interactive 3D Plot',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
        )

        # 合併節點和邊
        fig = go.Figure(data=[scatter] + edge_traces, layout=layout)

        if save_path:
            fig.write_html(save_path)
            print(f"Interactive 3D plot saved to {save_path}")
        else:
            fig.show()