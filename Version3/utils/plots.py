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
from matplotlib.colors import Normalize
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
        self.all_info = None
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

        text_series = pd.Series(data_tuple)
        extracted = text_series.str.extract(r'color:\s*([\d.-]+).*?node:\s*(\d+).*?size:\s*(\d+)')
        fail_idx = extracted.index[extracted.isna().any(axis=1)]
        for i in fail_idx:
            item = data_tuple[i]
            try:
                match = float(re.search(r'color: ([\d.-]+)', item).group(1))
            except Exception:
                match = float(re.search(r'color: ([\d.]+)', item).group(1))
            extracted.loc[i, 0] = match
            extracted.loc[i, 1] = re.search(r'node: (\d+)', item).group(1)
            extracted.loc[i, 2] = re.search(r'size: (\d+)', item).group(1)

        component_info = pd.DataFrame({
            'color': extracted[0].astype(float),
            'node': extracted[1].astype(int),
            'size': extracted[2].astype(int),
        })
        
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
    
        self.all_info = self.full_info.copy()
        self.outlier_info = self.all_info[self.all_info['outlier'] == True]
        self.filtered_info = self.all_info[self.all_info['outlier'] == False]

        return self.filtered_info, self.outlier_info

    def map_colors(self, choose, size=0, threshold=5, include_outliers=False):
        # include_outliers=False（預設，與原本行為一致）：只保留連通的節點來畫圖，離群節點濾掉。
        # include_outliers=True：連離群節點也一起畫進圖裡。
        base_info = self.all_info if include_outliers else self.all_info[self.all_info['outlier'] == False]

        # 過濾大小的資料點
        df = base_info[(base_info['size'] > size)]

        if self.range_lst is not None:
            df = df[(df['x'] > self.range_lst[0]) & (df['y'] < self.range_lst[2]) &
                    (df['x'] < self.range_lst[1]) & (df['y'] > self.range_lst[3])]

        # 注意：以前這裡會另外把 self.full_info 也依 range 篩選一次，是為了餵給舊版的 KDE 計算。
        # plot_dens 現在改用 self.filtered_info（跟拓樸圖同一份，已經篩過 outlier/size/range）算密度，
        # 不再需要這份資料，所以拿掉這個覆寫，讓 self.full_info 維持 extract_data 給的完整節點資料，
        # 避免 plot_3d / plot_3d_pvis 等其他方法拿到的 full_info 被意外裁切。

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

    def plot(self, choose, avg=None, save_path=None, set_label=False, size=100, anchor=1,
             color_vmax=None, color_percentile=95, group_rate_col=None):

        # self.filtered_info = self.filtered_info[self.filtered_info['size'] > size]

        # 過濾掉無效的顏色資料（group_rate_col 模式改看整個 filtered_info，避免 dropna 篩掉的筆數
        # 跟 self.filtered_info['x']/['y'] 對不齊）
        valid_data = self.filtered_info if group_rate_col is not None else self.filtered_info.dropna(subset=['color_for_plot_fixed'])
        clipped_size = np.clip(valid_data['size'], None, size)

        plt.figure(figsize=(15, 12))

        norm = None
        if group_rate_col is not None:
            # group_rate_col：不看單一節點自己的比例(ratio)，而是「這個節點所屬的 choose 類別，
            # 在整個母體上的死亡率」——同一類別的所有節點會拿到同一個數值，數值本身有意義，
            # 走跟 avg=True 一樣的連續色階 + colorbar，而不是 avg=False 的離散分類圖例。
            # 需要先呼叫過 map_colors(choose, ...)，讓 self.filtered_info 裡有 choose 這欄可以查表。
            color = self._group_rate_color(choose, group_rate_col)
            vmax = color_vmax if color_vmax is not None else np.nanpercentile(color.dropna(), color_percentile)
            vmax = max(float(vmax), 1e-6)
            norm = Normalize(vmin=0, vmax=vmax)
        elif avg:
            # avg=True 代表要看「比例」而不是節點內的原始加總數字（例如死亡數），
            # 所以顏色要用 ratio = color(該節點的加總值) / size(該節點樣本數)，
            # 不然節點越大、樣本數越多，color 就會單純因為加總的筆數多而變大，顏色會被 population 撐高，
            # 跟 avg 想表達的「平均/比例」意思不符。
            color = self.filtered_info['ratio']

            # 死亡率是稀有事件，大多數節點的比例都很小；但只要有一兩個樣本數極少的節點剛好 1/1=100%，
            # 顏色刻度上限就會被這種極端值撐到 1.0，其餘正常節點全部擠在最深色那一端，看不出彼此差異。
            # 改用「大多數節點所在的百分位數」當顏色刻度上限（預設 95th percentile，可用 color_vmax
            # 直接指定），超過上限的節點一樣畫成最亮的顏色，colorbar 用 extend='max' 標示有截斷。
            vmax = color_vmax if color_vmax is not None else np.nanpercentile(color, color_percentile)
            vmax = max(float(vmax), 1e-6)
            norm = Normalize(vmin=0, vmax=vmax)
        else:
            # 確保 'color_for_plot_fixed' 是有效的顏色格式
            color = [tuple(c) if isinstance(c, (list, tuple)) else c for c in valid_data['color_for_plot_fixed']]

        scatter = plt.scatter(
            self.filtered_info['x'], self.filtered_info['y'],
            c=color,
            norm=norm,
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
            if group_rate_col is not None:
                extend = 'max' if color.max() > norm.vmax else 'neither'
                colorbar = plt.colorbar(scatter, ax=plt.gca(), orientation='vertical', pad=0.02, extend=extend)
                colorbar.set_label(f'{choose} group {group_rate_col} rate')
            elif avg:
                extend = 'max' if color.max() > norm.vmax else 'neither'
                colorbar = plt.colorbar(scatter, ax=plt.gca(), orientation='vertical', pad=0.02, extend=extend)
                colorbar.set_label(f'{choose} ratio')
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
                plt.legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(anchor, 1), framealpha=0.1)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Mapper plot')
        plt.grid(True)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def _group_rate_color(self, choose, group_rate_col):

        rate_map = self.rbind_data.groupby(choose)[group_rate_col].mean()
        return self.filtered_info[choose].map(rate_map)

    def _aggregate_ids_sum(self, choose):

        values = self.rbind_data[choose].astype(float)

        def _agg(ids):
            vals = values.reindex(list(ids)).dropna()
            return float(np.nansum(vals.to_numpy())) if len(vals) else np.nan

        return self.filtered_info['ids'].apply(_agg)

    def get_density_curve(self, min_dens_size=10, n_grid=500, band_width=0.025, dens_choose=None):

        x_min, x_max = self.filtered_info["x"].min(), self.filtered_info["x"].max()

        info = self.filtered_info.copy()
        if dens_choose is not None:
            info['color'] = self._aggregate_ids_sum(dens_choose)

        band_start = self.range_lst[0] if self.range_lst is not None else x_min
        info['band'] = np.floor((info['x'] - band_start) / band_width).astype(int)

        band_totals = info.groupby('band')['size'].sum()
        valid_bands = band_totals[band_totals > min_dens_size].index
        density_info = info[info['band'].isin(valid_bands)]

        x_vals_all = density_info["x"].to_numpy()
        death_weights = density_info["color"].to_numpy()
        pop_weights = density_info["size"].to_numpy()

        grid = np.linspace(x_min, x_max, n_grid)
        base_std = x_vals_all.std(ddof=1) if len(x_vals_all) > 1 else 1.0
        scotts_factor = len(x_vals_all) ** (-1.0 / 5)
        h = max(base_std * scotts_factor * 0.3, 1e-6)

        kernel = np.exp(-0.5 * ((grid[:, None] - x_vals_all[None, :]) / h) ** 2)
        death_smooth = kernel @ death_weights
        pop_smooth = kernel @ pop_weights
        rate_curve = death_smooth / np.clip(pop_smooth, 1e-12, None)

        return grid, rate_curve, density_info

    def plot_dens(self, choose, avg=None, save_path=None, set_label=False, size=100, min_dens_size=10,
                  color_vmax=None, color_percentile=95, band_width=0.025, dens_choose=None, group_rate_col=None):
        # 作圖用filter，密度則使用原始進行計算(使用size篩選前的資料)
        clipped_size = np.clip(self.filtered_info['size'], None, size)
        fig, ax = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [4, 1], 'hspace': 0})
        # avg=True 用 ratio（= color/size，該節點的比例）而不是 color 的原始加總數字，理由同 plot()。
        norm = None
        if group_rate_col is not None:
            # group_rate_col：拓樸圖上色改用「這個節點所屬的 choose 類別，在整個母體上的死亡率」，
            # 不是單一節點自己的 ratio，也不是離散分類——同一類別的所有節點會拿到同一個數值，
            # 數值本身有意義，走連續色階 + colorbar。詳細說明見 _group_rate_color()。
            # 需要先呼叫過 map_colors(choose, ...)，讓 self.filtered_info 裡有 choose 這欄可以查表，
            # 對任何 categorical 欄位都適用（例如 choose 換成 '道路型態子類別名稱' 一樣可以用）。
            color = self._group_rate_color(choose, group_rate_col)
            vmax = color_vmax if color_vmax is not None else np.nanpercentile(color.dropna(), color_percentile)
            vmax = max(float(vmax), 1e-6)
            norm = Normalize(vmin=0, vmax=vmax)
        elif avg:
            color = self.filtered_info['ratio']
            # 顏色刻度上限用百分位數而不是固定 0~1，理由同 plot()：避免極少數小樣本節點的
            # 100% 死亡率把刻度撐爆，導致其餘節點全部擠在深色端看不出差異。
            vmax = color_vmax if color_vmax is not None else np.nanpercentile(color, color_percentile)
            vmax = max(float(vmax), 1e-6)
            norm = Normalize(vmin=0, vmax=vmax)
        else:
            color = [tuple(c) if isinstance(c, (list, tuple)) else c for c in self.filtered_info['color_for_plot_fixed']]
        # 為了讓兩個圖的 x 軸刻度標籤一致，固定刻度範圍
        ticks = np.arange(self.range_lst[0], self.range_lst[1] + 0.025, 0.025)
        x_min, x_max = self.filtered_info["x"].min(), self.filtered_info["x"].max()

        # 拓樸圖
        scatter = ax[0].scatter(
            self.filtered_info['x'], self.filtered_info['y'],
            c=color,
            norm=norm,
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
            if group_rate_col is not None:
                extend = 'max' if color.max() > norm.vmax else 'neither'
                colorbar = plt.colorbar(scatter, ax=ax[0], orientation='vertical', pad=0.02, extend=extend)
                colorbar.set_label(f'{choose} group {group_rate_col} rate')
            elif avg:
                extend = 'max' if color.max() > norm.vmax else 'neither'
                colorbar = plt.colorbar(scatter, ax=ax[0], orientation='vertical', pad=0.02, extend=extend)
                colorbar.set_label(f'{choose} ratio')
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
                ax[0].legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(1, 1), framealpha=0.1)
        
        ax[0].set_xlabel('')
        ax[0].set_ylabel('Y')
        ax[0].set_title('Mapper plot')
        ax[0].grid(True)
        ax[0].set_xlim(x_min, x_max)
        ax[0].set_xticks(ticks)
        ax[0].tick_params(axis='x', labelbottom=False)

        grid, rate_curve, density_info = self.get_density_curve(
            min_dens_size=min_dens_size, band_width=band_width, dens_choose=dens_choose
        )

        ax[1].plot(grid, rate_curve, color='#598e9c')

        # 設置密度圖的標籤和格線
        ax[1].set_xlabel("X")
        ax[1].set_ylabel(f"Normalized {dens_choose if dens_choose is not None else choose} Rate")
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

    def recolor(self, choose, avg=False):

        if self.all_info is None:
            raise RuntimeError('請先呼叫 extract_data() 建立節點資料，才能用 recolor() 換顏色')

        self.rbind_data['color_for_plot'] = (
            self.rbind_data[choose].astype(float) if avg
            else pd.factorize(self.rbind_data[choose])[0]
        )
        color_lookup = self.rbind_data['color_for_plot']

        def _aggregate(ids):
            vals = color_lookup.reindex(list(ids)).dropna()
            if len(vals) == 0:
                return np.nan
            if avg:
                return float(np.nansum(vals.to_numpy()))  # 對應 sum_of_data
            return vals.value_counts().idxmax()             # 對應 most_common_encoded_label

        self.all_info = self.all_info.copy()
        self.all_info['color'] = self.all_info['ids'].apply(_aggregate)
        self.all_info['ratio'] = self.all_info['color'] / self.all_info['size']

        print(f"Recolored by '{choose}' without rerunning create_mapper_plot / MapperLayoutInteractive.")

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

    def plot_3d(self, choose, avg=None, save_path=None, set_label=False, size=100,
                color_vmax=None, color_percentile=95):
        # 過濾掉無效的顏色資料
        # self.full_info = self.full_info.dropna(subset=['color_for_plot_fixed'])

        clipped_size = np.clip(self.full_info['size'], None, size)

        fig = plt.figure(figsize=(15, 12))
        ax = fig.add_subplot(111, projection='3d')

        norm = None
        if avg:
            # 同 plot()/plot_dens()：avg=True 要看比例，用 ratio 而不是原始加總的 color。
            color = self.full_info['ratio']
            # 顏色刻度上限用百分位數而不是固定 0~1，理由同 plot()。
            vmax = color_vmax if color_vmax is not None else np.nanpercentile(color, color_percentile)
            vmax = max(float(vmax), 1e-6)
            norm = Normalize(vmin=0, vmax=vmax)
        else:
            # 確保 'color_for_plot_fixed' 是有效的顏色格式
            color = [tuple(c) if isinstance(c, (list, tuple)) else c for c in self.full_info['color_for_plot_fixed']]

        scatter = ax.scatter(
            self.full_info['x'], self.full_info['y'], self.full_info['z'],
            c=color,
            norm=norm,
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
                extend = 'max' if color.max() > norm.vmax else 'neither'
                colorbar = plt.colorbar(scatter, ax=ax, orientation='vertical', pad=0.02, extend=extend)
                colorbar.set_label(f'{choose} ratio')
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
                ax.legend(handles=handles, title=f"{choose}", loc='upper right', bbox_to_anchor=(1, 1), framealpha=0.1)

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

        # 原本是兩兩節點互相比對 set(ids) 交集（O(n^2)且重複建 set），改用「id -> 節點」的反向索引，
        # 只有共享同一個 id 的節點才需要互相連邊，結果與原本完全相同，只是快很多。
        id_to_rows = {}
        for i, row in df.iterrows():
            for _id in row['ids']:
                id_to_rows.setdefault(_id, []).append(i)
        edge_set = set()
        for rows_sharing_id in id_to_rows.values():
            for a in range(len(rows_sharing_id)):
                for b in range(a + 1, len(rows_sharing_id)):
                    i, j = rows_sharing_id[a], rows_sharing_id[b]
                    if i > j:
                        i, j = j, i
                    edge_set.add((df.loc[i, 'node'], df.loc[j, 'node']))
        for n1, n2 in edge_set:
            G.add_edge(n1, n2)

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

    邏輯與原本完全相同：兩個節點只要共享至少一個 id 就算連通，最後標記與「size 最大的節點」
    不連通的點為 outlier。原本的寫法是逐點 BFS，每訪問一個節點就要對全部節點重新做一次
    `set(x)` 轉換再比對交集，等於 O(n^2) 還外加重複建 set 的開銷。這裡改用 Union-Find：
    對每個 id 建一次「第一次出現在哪個節點」的索引，共享同一個 id 的節點直接 union 起來，
    一次線性掃描就能得到與原本 BFS 完全相同的連通分量結果，只是快很多。
    """
    n = len(df)
    if n == 0:
        df["outlier"] = pd.Series([], dtype=bool)
        return

    idx_list = df.index.to_numpy()
    pos_of_idx = {idx: pos for pos, idx in enumerate(idx_list)}

    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    first_owner = {}
    for pos, ids in enumerate(df['ids']):
        for _id in ids:
            if _id in first_owner:
                union(pos, first_owner[_id])
            else:
                first_owner[_id] = pos

    root = find(pos_of_idx[df['size'].idxmax()])
    connected_mask = np.array([find(pos) == root for pos in range(n)])

    # 3. 標記所有點是否為 outlier
    df["outlier"] = ~connected_mask

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