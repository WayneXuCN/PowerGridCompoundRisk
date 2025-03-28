class ClimateParams:
    flood_max_lat = 55      # 洪水风险最大纬度
    flood_scale = 10        # 洪水风险衰减系数
    heat_min_lat = 40       # 热浪风险最小纬度
    heat_scale = 15         # 热浪风险增长系数
    drought_offset = 10     # 干旱经度偏移量
    drought_scale = 20      # 干旱风险缩放系数

class DynamicParams:
    k = 0.1                 # 失效速率
    r = 0.05                # 恢复速率
    sigma = 50              # 地理距离衰减参数 (km)
    w1 = 0.7                # 一阶邻居权重
    w2 = 0.3                # 二阶邻居权重

class GNNParams:
    input_dim = 5
    hidden_dim = 16
    output_dim = 1
    lr = 0.01
    epochs = 100