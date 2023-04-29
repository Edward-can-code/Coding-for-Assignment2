# 导入所需模块
import random
import numpy as np
from scipy.spatial import distance
from scipy.spatial import distance_matrix

# 定义城市坐标
cities = np.array([[116, 40],
                   [117, 39],
                   [115, 39],
                   [113, 38],
                   [112, 41],
                   [123, 42],
                   [125, 44],
                   [127, 46],
                   [121, 31],
                   [119, 32],
                   [120, 30],
                   [117, 32],
                   [119, 26],
                   [116, 29],
                   [117, 37],
                   [114, 35],
                   [114, 31],
                   [113, 28],
                   [113, 23],
                   [108, 23],
                   [110, 20],
                   [107, 30],
                   [104, 31],
                   [107, 27],
                   [103, 25],
                   [91, 30],
                   [109, 34],
                   [104, 36],
                   [102, 37],
                   [106, 38],
                   [88, 44]])

# 定义蚁群算法的类
class AntColony:
    def __init__(self, distances, n_ants, n_iterations, decay, alpha=1, beta=1):
        """
        初始化蚁群算法的参数
        
        :param distances: ndarray, 距离矩阵，表示各个点之间的距离
        :param n_ants: int, 蚂蚁数量
        :param n_iterations: int, 迭代次数
        :param decay: float, 信息素挥发率
        :param alpha: float, 信息素重要程度因子
        :param beta: float, 启发式函数重要程度因子
        """
        self.distances = distances
        self.pheromones = np.ones(distances.shape)  # 初始化信息素矩阵
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
    def run(self):
        """
        运行蚁群算法
        
        :return: tuple, 最优路径的长度和路径本身
        """
        best_distance = np.inf  # 初始化最优距离
        best_path = None  # 初始化最优路径
        
        for i in range(self.n_iterations):
            paths = self.generate_paths()  # 生成一批路径
            distances = [self.get_distance(path) for path in paths]  # 计算每条路径的距离
            if min(distances) < best_distance:  # 如果找到更优的路径，则更新最优路径和最优距离
                best_distance = min(distances)
                best_path =paths[np.argmin(distances)]
            self.update_pheromones(paths, distances)  # 更新信息素矩阵
            
        return best_distance, best_path
        
    def generate_paths(self):
        """
        生成一批路径
        
        :return: list, 包含多条路径的列表
        """
        paths = []
        for i in range(self.n_ants):
            path = self.generate_path()  # 生成一条路径
            paths.append(path)
        return paths
        
    def generate_path(self):
        """
        生成一条路径
        
        :return: list, 包含路径上各个节点的列表
        """
        start = random.randint(0, len(self.distances) - 1)    # 随机选择一个起始城市
        path = [start]    # 将起始城市添加到路径中
        unvisited = set(range(len(self.distances)))    # 创建未访问城市的集合
        unvisited.remove(start)    # 从未访问城市中移除起始城市
        
        while unvisited:  # 当还有未访问节点时
            next_node = self.choose_next(path[-1], unvisited)    # 选择下一个要访问的城市
            path.append(next_node)    # 将下一个城市添加到路径中
            unvisited.remove(next_node)    # 从未访问城市中移除已访问的城市
            
        # 将起始城市添加到路径的末尾，形成一个环路
        path.append(start)
       
        return path
 
    
    def choose_next(self, node, unvisited):
        """
        选择下一个节点
        
        :param node: int, 当前节点
        :param unvisited: set, 未访问节点集合
        :return: int, 下一个节点
        """
        unvisited = list(unvisited)
        pheromones = [self.pheromones[node][other] for other in unvisited]  # 当前节点到其他未访问节点的信息素浓度
        distances = [self.distances[node][other] for other in unvisited]  # 当前节点到其他未访问节点的距离
        scores = np.array(pheromones) ** self.alpha * np.array(distances) ** -self.beta  # 计算各个节点的得分
        probabilities = scores / sum(scores)  # 计算各个节点的概率
        return unvisited[self.choose_index(probabilities)]  # 根据概率选择下一个节点
    
    def choose_index(self, probabilities):
        """
        根据概率选择节点的索引
        
        :param probabilities: list, 各个节点的概率
        :return: int, 被选中的节点的索引
        """
        r = random.uniform(0, sum(probabilities))  # 生成一个随机数
        for i, probability in enumerate(probabilities):
            r -= probability
            if r <= 0:
                return i
        
    def get_distance(self, path):
        """
        计算路径的距离
        
        :param path: list, 包含路径上各个节点的列表
        :return: float, 路径的距离
        """
        distance = 0
        for i in range(len(path) - 1):
            distance += self.distances[path[i]][path[i + 1]]
        distance += self.distances[path[-1]][path[0]]
        return distance
    
    def update_pheromones(self, paths, distances):
        """
        更新信息素矩阵
        
        :param paths: list, 包含多条路径的列表
        :param distances: list, 各条路径的距离
        """
        for i in range(len(self.distances)):
            for j in range(len(self.distances)):
                if i == j:
                    continue
                self.pheromones[i][j] *= (1 - self.decay)  # 信息素挥发
                for path, distance in zip(paths, distances):
                    if j in path and i in path:
                        self.pheromones[i][j] += 1 / distance  # 信息素增加
# 计算城市之间的距离矩阵
distances = distance_matrix(cities, cities)

# 创建 AntColony 并运行算法
colony = AntColony(distances, n_ants=30, n_iterations=200, decay=0.1)
best_distance, best_path = colony.run()

# 打印结果
print(f'Best distance: {best_distance:.2f}')
print(f'Best path: {best_path}')
