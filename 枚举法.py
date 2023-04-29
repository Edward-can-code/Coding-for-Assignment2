import math
import itertools

# 定义城市的坐标列表
cities = [[116, 40],
          [117, 39],
          [115, 39],
          # ... 省略其他城市以防止程序卡机，可以自行酌情添加(‐^▽^‐)...
          [102, 37],
          [106, 38],
          [88, 44]]

# 定义计算两个城市之间距离的函数
def distance(city1, city2):
    x1, y1 = city1
    x2, y2 = city2
    # 使用欧几里得距离计算两个城市之间的距离
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

# 定义解决TSP问题的函数
def tsp(cities):
    # 初始化最小距离为正无穷
    min_distance = float('inf')
    # 初始化最小距离对应的路径为None
    min_path = None
    # 使用itertools.permutations生成所有城市的排列组合
    for path in itertools.permutations(cities):
        # 初始化当前路径的总距离为0
        total_distance = 0
        # 遍历当前路径上相邻两个城市，计算它们之间的距离并加到总距离上
        for i in range(len(path) - 1):
            total_distance += distance(path[i], path[i+1])
        # 如果当前路径的总距离比之前的最小距离小，则更新最小距离和最小距离对应的路径
        if total_distance < min_distance:
            min_distance = total_distance
            min_path = path
    # 返回最小距离对应的路径和最小距离
    return min_path, min_distance

# 调用tsp函数并输出结果
min_path, min_distance = tsp(cities)
print("The optimal path is: ", min_path)
print("The total distance is: ", min_distance)


