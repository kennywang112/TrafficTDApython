import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from Version3.tdamapper.plot import MapperLayoutInteractive

class MapperPlotter:
    def __init__(self, mapper_info, rbind_data, detailed_results_df, cmap='jet', seed=10, width=400, height=400):
        self.mapper_info = mapper_info
        self.rbind_data = rbind_data
        self.detailed_results_df = detailed_results_df
        self.cmap = cmap
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
            dim=2,
            iterations=30,
            seed=self.seed,
            width=self.width,
            height=self.height
        )
        print("Mapper plot created.")

    def extract_data(self):
        x = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['x']
        y = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['y']
        threeDimData = pd.DataFrame({'x': x, 'y': y})
        
        data_tuple = vars(self.mapper_plot._MapperLayoutInteractive__fig)['_data_objs'][1]['text']
        data = []
        for item in data_tuple:
            color = float(re.search(r'color: ([\d.]+)', item).group(1))
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

    def map_colors(self, choose, color_mapping_fixed, size=0):
        df = self.full_info[(self.full_info['size'] > size)]
        
        # 修改这部分代码
        unique_values = self.rbind_data.reset_index()[[choose, 'color_for_plot']].drop_duplicates()
        df = df.merge(unique_values, left_on='color', right_on='color_for_plot', how='left')
        
        # 處理category類型
        if df[choose].dtype.name == 'category':
            # 將category轉為字串符
            df['color_for_plot_fixed'] = df[choose].astype(str).map(color_mapping_fixed)
        else:
            # 其他數據類型
            if isinstance(df[choose], pd.Series):
                df['color_for_plot_fixed'] = df[choose].map(color_mapping_fixed)
            else:
                df['color_for_plot_fixed'] = df[choose].astype(str).map(color_mapping_fixed)
        
        self.full_info = df
        self.color_palette = color_mapping_fixed
        print("Colors mapped using predefined mapping.")

    def plot(self, choose, unique_categories, color_palette, 
             avg=None, save_path=None, set_label=False, size=100):
        myfont = FontProperties(fname=r"/System/Library/Fonts/PingFang.ttc")
        clipped_size = np.clip(self.full_info['size'], None, size)
        # clipped_size = self.full_info['size']

        plt.figure(figsize=(15, 12))

        if avg:
            color=self.full_info['color']
        else:
            color=self.full_info['color_for_plot_fixed'].tolist()

        scatter = plt.scatter(
            self.full_info['x'], self.full_info['y'],
            c=color,
            edgecolors='black',
            linewidths=0.5,
            s=clipped_size,
            marker='o',
            alpha=0.7
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
                colorbar.set_label(f'{choose}', fontproperties=myfont)
            else:
                handles = [
                    plt.Line2D(
                        [0], [0],
                        marker='o',
                        color=color_palette(i),
                        markersize=10,
                        label=name
                    ) for i, name in enumerate(unique_categories)
                ]
                plt.legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(1.2, 1))

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Mapper plot')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()