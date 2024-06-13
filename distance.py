import numpy as np

def latlon_to_xyz(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.vstack((x, y, z)).T

# 計算球面距離
def spherical_dist(pos1, pos2, radius=6371):
    cos_angle = np.dot(pos1, pos2.T)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    return radius * angle

def calculate_distances(df, scenic_df, colname, distance_threshold=500):
    # 將經緯度轉換成球體的三維坐標
    df_xyz = latlon_to_xyz(df['緯度'], df['經度'])
    scenic_xyz = latlon_to_xyz(scenic_df['緯度'], scenic_df['經度'])

    # 計算所有配對之間的球面距離，並轉換為米
    distances = spherical_dist(df_xyz, scenic_xyz) * 1000

    # 檢查哪些距離小於設定的threshold
    distances_less_than_threshold = (distances < distance_threshold)

    # 計算每一行小於閾值的點數量
    df[colname] = distances_less_than_threshold.sum(axis=1)

    return df