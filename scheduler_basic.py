"""
scheduler_basic.py - マシンに基づく簡易スケジューリングとガントチャート可視化

このスクリプトは、AMR（搬送）を考慮せず、マシンのみでジョブスケジュールを組み、
ガントチャートで表示し、実行結果をテキストファイルに出力します。

経路を考慮したガントチャートも作成可能です。
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import itertools
import random
import copy
import os
import shutil
from datetime import datetime
from package.data_manager import load_job_data, load_machine_data, convert_machine_list_into_dict, load_route_data, convert_route_list_into_dict, load_edge_data
from package.classes import Job, Machine, Process, Route
from typing import List, Dict, Optional, Tuple


class GeneticAlgorithm:
    """
    遺伝的アルゴリズム（GA）によるジョブスケジュール最適化クラス
    """
    
    def __init__(self, job_list: List[Job], machine_dict: Dict[str, List[Machine]], 
                 population_size: int = 50, crossover_rate: float = 0.8, 
                 mutation_rate: float = 0.1, max_generations: int = 100,
                 route_dict: Optional[Dict[str, Route]] = None,
                 edge_graph: Optional[Dict[str, Dict[str, float]]] = None):
        """
        GAのパラメータを初期化
        
        Args:
            job_list: ジョブのリスト
            machine_dict: 工程ごとにグループ化されたマシンの辞書
            population_size: 個体数
            crossover_rate: 交叉確率
            mutation_rate: 突然変異確率
            max_generations: 最大世代数
            route_dict: 経路データの辞書（搬送時間を考慮する場合、後方互換性のため）
            edge_graph: エッジグラフ（搬送時間を計算する場合）
        """
        self.job_list = job_list
        self.machine_dict = machine_dict
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        self.route_dict = route_dict  # 経路データ（後方互換性のため）
        self.edge_graph = edge_graph  # エッジグラフ（最短経路計算用）
        
        # ジョブIDのリストを作成
        self.job_ids = [job.name for job in job_list]
        
        # 最良解を保存
        self.best_individual = None
        self.best_fitness = float('inf')
        self.best_schedule = None
        
        # 進化の履歴を保存
        self.fitness_history = []
    
    def create_individual(self) -> List[str]:
        """
        ランダムな個体（ジョブ順序）を生成
        
        Returns:
            ジョブIDの順序リスト
        """
        individual = self.job_ids.copy()
        random.shuffle(individual)
        return individual
    
    def evaluate_fitness(self, individual: List[str]) -> float:
        """
        個体の適応度（makespan）を評価
        
        Args:
            individual: ジョブ順序のリスト
            
        Returns:
            適応度（makespan）- 小さいほど良い
        """
        # ジョブ順序に基づいてジョブリストを再構築
        ordered_jobs = []
        for job_id in individual:
            for job in self.job_list:
                if job.name == job_id:
                    ordered_jobs.append(job)
                    break
        
        # スケジュールを評価
        machine_schedules = self._evaluate_schedule_with_order(ordered_jobs)
        
        # makespanを計算（最大完了時刻）
        makespan = 0.0
        for schedules in machine_schedules.values():
            for schedule in schedules:
                makespan = max(makespan, schedule['end_time'])
        
        return makespan
    
    def _calculate_shortest_path(self, from_node: str, to_node: str) -> Tuple[List[str], float]:
        """
        ダイクストラ法で最短経路を計算
        
        Args:
            from_node: 出発ノード
            to_node: 到着ノード
            
        Returns:
            (経路のノードリスト, 総コスト)
        """
        if self.edge_graph is None or from_node not in self.edge_graph:
            return ([from_node, to_node], 0.0)
        
        if from_node == to_node:
            return ([from_node], 0.0)
        
        # ダイクストラ法
        import heapq
        
        distances = {node: float('inf') for node in self.edge_graph}
        distances[from_node] = 0.0
        previous = {node: None for node in self.edge_graph}
        queue = [(0.0, from_node)]
        visited = set()
        
        while queue:
            current_dist, current_node = heapq.heappop(queue)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            if current_node == to_node:
                break
            
            if current_node not in self.edge_graph:
                continue
            
            for neighbor, cost in self.edge_graph[current_node].items():
                new_dist = current_dist + cost
                if new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current_node
                    heapq.heappush(queue, (new_dist, neighbor))
        
        # 経路を復元
        if distances[to_node] == float('inf'):
            return ([from_node, to_node], 0.0)
        
        path = []
        current = to_node
        while current is not None:
            path.append(current)
            current = previous[current]
        path.reverse()
        
        return (path, distances[to_node])
    
    def _get_transport_time(self, from_node: str, to_node: str) -> float:
        """
        2つのノード間の搬送時間を取得（エッジデータから最短経路を計算）
        
        Args:
            from_node: 出発ノード
            to_node: 到着ノード
            
        Returns:
            搬送時間
        """
        if self.edge_graph is None:
            # エッジデータがない場合は、既存のroute_dictを使用
            if self.route_dict is not None:
                route_key = f"{from_node}->{to_node}"
                if route_key in self.route_dict:
                    return self.route_dict[route_key].cost
            return 0.0
        
        _, cost = self._calculate_shortest_path(from_node, to_node)
        return cost
    
    def _evaluate_schedule_with_order(self, ordered_jobs: List[Job]) -> Dict[str, List]:
        """
        指定されたジョブ順序でスケジュールを評価（搬送時間を考慮）
        
        Args:
            ordered_jobs: ジョブの順序リスト
            
        Returns:
            マシンごとのスケジュール情報（搬送時間も含む）
        """
        machine_schedules: Dict[str, List] = {}
        machine_available_time: Dict[str, float] = {}
        transport_schedules: List[Dict] = []  # 搬送スケジュール
        
        # 全てのマシンに対して初期化
        for process_name, machines in self.machine_dict.items():
            for machine in machines:
                if machine.name not in machine_schedules:
                    machine_schedules[machine.name] = []
                if machine.name not in machine_available_time:
                    machine_available_time[machine.name] = 0.0
        
        # 指定された順序でジョブを処理
        for job in ordered_jobs:
            for i, process in enumerate(job.process_list):
                process_name = process.label
                process_time = process.time
                
                # この工程を処理できるマシンを見つける
                available_machines = self.machine_dict.get(process_name, [])
                
                if len(available_machines) == 0:
                    continue
                
                # 前の工程の終了マシンと終了時刻を取得
                prev_machine = None
                prev_end_time = 0.0
                from_node = None
                
                if i > 0:
                    prev_process = job.process_list[i - 1]
                    # 前の工程が処理されたマシンを見つける
                    for mach_name, schedule in machine_schedules.items():
                        for item in schedule:
                            if item['job_name'] == job.name and item['process_label'] == prev_process.label:
                                prev_end_time = max(prev_end_time, item['end_time'])
                            # 前の工程を処理したマシンを見つける
                            if prev_machine is None:
                                for m in self.machine_dict.get(prev_process.label, []):
                                    if m.name == mach_name:
                                        prev_machine = m
                                        from_node = m.exit_port.node
                                        break
                else:
                    # 最初の工程の場合、出発地点をノード1とする（AMRがジョブを運ぶ開始地点）
                    from_node = "1"
                
                # 最も早く利用可能なマシンを見つける
                best_machine = None
                earliest_start_time = float('inf')
                
                for machine in available_machines:
                    # 搬送時間を考慮
                    transport_time = 0.0
                    if from_node is not None and self.edge_graph is not None:
                        # 出発ノードから現在のマシンの入口ノードへの搬送時間
                        to_node = machine.entrance_port.node
                        transport_time = self._get_transport_time(from_node, to_node)
                    
                    # このマシンが利用可能になる時刻（前工程終了時刻 + 搬送時間）
                    # 最初の工程の場合、prev_end_timeは0なので、搬送時間だけが加算される
                    machine_start_time = max(machine_available_time[machine.name], prev_end_time + transport_time)
                    
                    if machine_start_time < earliest_start_time:
                        earliest_start_time = machine_start_time
                        best_machine = machine
                
                if best_machine is None:
                    continue
                
                # 搬送時間を記録
                if from_node is not None and self.edge_graph is not None:
                    to_node = best_machine.entrance_port.node
                    transport_time = self._get_transport_time(from_node, to_node)
                    
                    if transport_time > 0:
                        transport_start_time = prev_end_time
                        transport_schedules.append({
                            'job_name': job.name,
                            'from_machine': prev_machine.name if prev_machine else 'Start',
                            'to_machine': best_machine.name,
                            'from_node': from_node,
                            'to_node': to_node,
                            'start_time': transport_start_time,
                            'end_time': transport_start_time + transport_time,
                            'duration': transport_time,
                            'type': 'transport'
                        })
                
                # マシンが利用可能になる時刻を更新
                start_time = earliest_start_time
                end_time = start_time + process_time
                machine_available_time[best_machine.name] = end_time
                
                # スケジュール情報を保存
                schedule_item = {
                    'job_name': job.name,
                    'process_label': process.label,
                    'process_index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': process_time,
                    'type': 'process'
                }
                machine_schedules[best_machine.name].append(schedule_item)
        
        # 搬送スケジュールをマシンスケジュールに追加（"Transport"という仮想マシンとして）
        if len(transport_schedules) > 0:
            machine_schedules['Transport'] = transport_schedules
        
        return machine_schedules
    
    def tournament_selection(self, population: List[List[str]], fitness_values: List[float], 
                           tournament_size: int = 3) -> List[str]:
        """
        トーナメント選択
        
        Args:
            population: 個体群
            fitness_values: 適応度値のリスト
            tournament_size: トーナメントサイズ
            
        Returns:
            選択された個体
        """
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        
        # 最小の適応度（最良）の個体を選択
        winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return population[winner_index].copy()
    
    def roulette_selection(self, population: List[List[str]], fitness_values: List[float]) -> List[str]:
        """
        ルーレット選択
        
        Args:
            population: 個体群
            fitness_values: 適応度値のリスト
            
        Returns:
            選択された個体
        """
        # 適応度を最大化問題に変換（小さい値ほど大きな重み）
        max_fitness = max(fitness_values)
        weights = [max_fitness - fitness + 1 for fitness in fitness_values]
        
        # 重みに基づいて選択
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for i, weight in enumerate(weights):
            cumulative_weight += weight
            if r <= cumulative_weight:
                return population[i].copy()
        
        return population[-1].copy()
    
    def one_point_crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        一点交叉
        
        Args:
            parent1: 親個体1
            parent2: 親個体2
            
        Returns:
            子個体のペア
        """
        if len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        
        # 子個体1: parent1の前半 + parent2の後半（重複を避ける）
        child1 = parent1[:crossover_point].copy()
        remaining1 = [job for job in parent2 if job not in child1]
        child1.extend(remaining1)
        
        # 子個体2: parent2の前半 + parent1の後半（重複を避ける）
        child2 = parent2[:crossover_point].copy()
        remaining2 = [job for job in parent1 if job not in child2]
        child2.extend(remaining2)
        
        return child1, child2
    
    def order_crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """
        順序交叉（OX）
        
        Args:
            parent1: 親個体1
            parent2: 親個体2
            
        Returns:
            子個体のペア
        """
        if len(parent1) <= 2:
            return parent1.copy(), parent2.copy()
        
        # 交叉点をランダムに選択
        start = random.randint(0, len(parent1) - 2)
        end = random.randint(start + 1, len(parent1) - 1)
        
        # 子個体1の作成
        child1 = [None] * len(parent1)
        child1[start:end+1] = parent1[start:end+1]
        
        # parent2から残りの要素を順序を保って追加
        remaining = [job for job in parent2 if job not in child1[start:end+1]]
        idx = 0
        for i in range(len(child1)):
            if child1[i] is None:
                child1[i] = remaining[idx]
                idx += 1
        
        # 子個体2の作成
        child2 = [None] * len(parent2)
        child2[start:end+1] = parent2[start:end+1]
        
        # parent1から残りの要素を順序を保って追加
        remaining = [job for job in parent1 if job not in child2[start:end+1]]
        idx = 0
        for i in range(len(child2)):
            if child2[i] is None:
                child2[i] = remaining[idx]
                idx += 1
        
        return child1, child2
    
    def swap_mutation(self, individual: List[str]) -> List[str]:
        """
        スワップ突然変異（2つのジョブの順序を入れ替える）
        
        Args:
            individual: 個体
            
        Returns:
            突然変異後の個体
        """
        if len(individual) <= 1:
            return individual.copy()
        
        mutated = individual.copy()
        i, j = random.sample(range(len(mutated)), 2)
        mutated[i], mutated[j] = mutated[j], mutated[i]
        
        return mutated
    
    def evolve(self) -> Tuple[List[str], float, Dict[str, List]]:
        """
        遺伝的アルゴリズムの進化を実行
        
        Returns:
            最良個体、最良適応度、最良スケジュール
        """
        print(f"遺伝的アルゴリズム開始: 個体数={self.population_size}, 世代数={self.max_generations}")
        print(f"交叉確率={self.crossover_rate}, 突然変異確率={self.mutation_rate}")
        
        # 初期個体群を生成
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.max_generations):
            # 適応度を評価
            fitness_values = [self.evaluate_fitness(ind) for ind in population]
            
            # 最良解を更新
            min_fitness = min(fitness_values)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                best_idx = fitness_values.index(min_fitness)
                self.best_individual = population[best_idx].copy()
                self.best_schedule = self._evaluate_schedule_with_order(
                    [job for job_id in self.best_individual 
                     for job in self.job_list if job.name == job_id]
                )
            
            self.fitness_history.append(min_fitness)
            
            # 進捗を表示
            if generation % 10 == 0 or generation == self.max_generations - 1:
                print(f"世代 {generation+1}/{self.max_generations}: 最良適応度 = {min_fitness:.2f}")
            
            # 新しい世代を生成
            new_population = []
            
            # エリート保存（最良個体を保持）
            elite_idx = fitness_values.index(min_fitness)
            new_population.append(population[elite_idx].copy())
            
            # 残りの個体を生成
            while len(new_population) < self.population_size:
                # 選択
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    if random.random() < 0.5:
                        child1, child2 = self.one_point_crossover(parent1, parent2)
                    else:
                        child1, child2 = self.order_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # 突然変異
                if random.random() < self.mutation_rate:
                    child1 = self.swap_mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.swap_mutation(child2)
                
                new_population.extend([child1, child2])
            
            # 個体数が超過した場合は調整
            if len(new_population) > self.population_size:
                new_population = new_population[:self.population_size]
            
            population = new_population
        
        print(f"遺伝的アルゴリズム完了: 最良適応度 = {self.best_fitness:.2f}")
        return self.best_individual, self.best_fitness, self.best_schedule


def schedule_jobs(job_list: List[Job], machine_dict: Dict[str, List[Machine]]) -> Dict[str, List]:
    """
    ジョブをマシンに割り当ててスケジュールを作成する
    
    Args:
        job_list: ジョブのリスト
        machine_dict: 工程ごとにグループ化されたマシンの辞書
        
    Returns:
        マシンごとのスケジュール情報（開始時刻、終了時刻、ジョブID）
    """
    # 各マシンのスケジュール情報を保存する辞書
    machine_schedules: Dict[str, List[Dict]] = {}
    
    # 各マシンの利用可能時刻を追跡
    machine_available_time: Dict[str, float] = {}
    
    # 全てのマシンに対して初期化
    for process_name, machines in machine_dict.items():
        for machine in machines:
            machine_schedules[machine.name] = []
            machine_available_time[machine.name] = 0.0
    
    # 全てのジョブについてスケジュールを作成
    for job in job_list:
        # ジョブの各工程を順番に処理
        for i, process in enumerate(job.process_list):
            process_name = process.label
            process_time = process.time
            
            # この工程を処理できるマシンを見つける
            available_machines = machine_dict.get(process_name, [])
            
            if len(available_machines) == 0:
                print(f"警告: 工程 '{process_name}' を処理できるマシンが見つかりません")
                continue
            
            # 最も早く利用可能なマシンを見つける（FIFO的な割り当て）
            best_machine = None
            earliest_start_time = float('inf')
            
            for machine in available_machines:
                # 前の工程が終わった時刻を取得
                prev_end_time = 0.0
                if i > 0:
                    # 前の工程がどこかで処理された情報を探す
                    prev_process = job.process_list[i - 1]
                    # 前の工程が処理されたことを考慮
                    for mach_name, schedule in machine_schedules.items():
                        for item in schedule:
                            if item['job_name'] == job.name and item['process_label'] == prev_process.label:
                                prev_end_time = max(prev_end_time, item['end_time'])
                
                # このマシンが利用可能になる時刻
                machine_start_time = max(machine_available_time[machine.name], prev_end_time)
                
                if machine_start_time < earliest_start_time:
                    earliest_start_time = machine_start_time
                    best_machine = machine
            
            if best_machine is None:
                continue
            
            # マシンが利用可能になる時刻を更新
            start_time = earliest_start_time
            end_time = start_time + process_time
            machine_available_time[best_machine.name] = end_time
            
            # スケジュール情報を保存
            schedule_item = {
                'job_name': job.name,
                'process_label': process.label,
                'process_index': i,
                'start_time': start_time,
                'end_time': end_time,
                'duration': process_time
            }
            machine_schedules[best_machine.name].append(schedule_item)
    
    return machine_schedules


def generate_all_schedule_patterns(job_list: List[Job], machine_dict: Dict[str, List[Machine]]) -> int:
    """
    すべての可能なマシン割り当てパターン数を計算する
    
    Args:
        job_list: ジョブのリスト
        machine_dict: 工程ごとにグループ化されたマシンの辞書
        
    Returns:
        すべてのマシン割り当てパターン数
    """
    total_patterns = 1
    
    # 各ジョブについて、その工程の組み合わせ数を計算
    for job in job_list:
        job_patterns = 1
        
        for process in job.process_list:
            process_name = process.label
            available_machines = machine_dict.get(process_name, [])
            
            if len(available_machines) == 0:
                print(f"警告: 工程 '{process_name}' を処理できるマシンが見つかりません")
                continue
            
            job_patterns *= len(available_machines)
        
        total_patterns *= job_patterns
    
    return total_patterns


def generate_random_schedule(job_list: List[Job], machine_dict: Dict[str, List[Machine]]) -> tuple:
    """
    ランダムなスケジュールパターンを1つ生成する
    
    Args:
        job_list: ジョブのリスト
        machine_dict: 工程ごとにグループ化されたマシンの辞書
        
    Returns:
        ランダムなマシン割り当てパターン（タプル）
    """
    pattern = []
    
    for job in job_list:
        job_assignment = []
        
        for process in job.process_list:
            process_name = process.label
            available_machines = machine_dict.get(process_name, [])
            
            if len(available_machines) == 0:
                continue
            
            # この工程からランダムに1つのマシンを選択
            selected_machine = random.choice(available_machines)
            job_assignment.append(selected_machine)
        
        pattern.append(tuple(job_assignment))
    
    return tuple(pattern)


def evaluate_schedule(job_list: List[Job], machine_assignment: tuple, machine_dict: Dict[str, List[Machine]]) -> Dict[str, List]:
    """
    特定のマシン割り当てパターンに対してスケジュールをシミュレートする
    
    Args:
        job_list: ジョブのリスト
        machine_assignment: マシン割り当てパターン（ジョブ×工程のタプル）
        machine_dict: 工程ごとにグループ化されたマシンの辞書
        
    Returns:
        マシンごとのスケジュール情報
    """
    machine_schedules: Dict[str, List] = {}
    machine_available_time: Dict[str, float] = {}
    
    # 全てのマシンに対して初期化
    for process_name, machines in machine_dict.items():
        for machine in machines:
            if machine.name not in machine_schedules:
                machine_schedules[machine.name] = []
            if machine.name not in machine_available_time:
                machine_available_time[machine.name] = 0.0
    
    # 各ジョブの各工程を処理
    for job_idx, job in enumerate(job_list):
        # machine_assignmentの構造:
        # ((ジョブ1の全工程マシン), (ジョブ2の全工程マシン), ...)
        # 各ジョブの割り当てはタプル: (process1_machine, process2_machine, ...)
        
        job_assignment = machine_assignment[job_idx]
        
        for i, process in enumerate(job.process_list):
            # この工程に割り当てられたマシンを取得
            if i >= len(job_assignment):
                continue
            
            assigned_machine = job_assignment[i]
            
            # マシンが有効かチェック
            if assigned_machine is None or not hasattr(assigned_machine, 'name'):
                continue
            
            process_time = process.time
            
            # 前の工程が終わった時刻を取得
            prev_end_time = 0.0
            if i > 0:
                prev_process = job.process_list[i - 1]
                # 前の工程の終了時刻を探す
                for mach_name, schedule in machine_schedules.items():
                    for item in schedule:
                        if item['job_name'] == job.name and item['process_label'] == prev_process.label:
                            prev_end_time = max(prev_end_time, item['end_time'])
            
            # マシンが利用可能になる時刻
            start_time = max(machine_available_time[assigned_machine.name], prev_end_time)
            end_time = start_time + process_time
            machine_available_time[assigned_machine.name] = end_time
            
            # スケジュール情報を保存
            schedule_item = {
                'job_name': job.name,
                'process_label': process.label,
                'process_index': i,
                'start_time': start_time,
                'end_time': end_time,
                'duration': process_time
            }
            machine_schedules[assigned_machine.name].append(schedule_item)
    
    return machine_schedules


def generate_schedules(job_list: List[Job], machine_dict: Dict[str, List[Machine]], 
                       num_samples: Optional[int] = None) -> List[Dict[str, List]]:
    """
    スケジュールパターンを生成する（ランダムサンプリングのみ）
    
    Args:
        job_list: ジョブのリスト
        machine_dict: 工程ごとにグループ化されたマシンの辞書
        num_samples: サンプル数（Noneの場合は5、整数の場合はその数だけランダムサンプリング）
        
    Returns:
        スケジュール候補のリスト
    """
    # 全パターン数を計算
    total_patterns = generate_all_schedule_patterns(job_list, machine_dict)
    
    if total_patterns == 0:
        print("警告: 生成可能なスケジュールパターンがありません")
        return []
    
    print(f"全スケジュールパターン数: {total_patterns}")
    
    # サンプル数を決定
    if num_samples is None:
        num_samples = 5
        print(f"デフォルトで{num_samples}通りのランダムサンプリングを実行")
    else:
        print(f"{num_samples}通りのランダムサンプリングを実行")
    
    # ランダムにサンプルを生成
    schedules = []
    for idx in range(num_samples):
        # ランダムなパターンを生成
        pattern = generate_random_schedule(job_list, machine_dict)
        
        # このパターンでスケジュールを評価
        machine_schedules = evaluate_schedule(job_list, pattern, machine_dict)
        schedules.append(machine_schedules)
        print(f"スケジュール {idx+1}/{num_samples} を生成しました")
    
    return schedules


def create_gantt_chart(machine_schedules: Dict[str, List], machine_list: List[Machine], 
                       filename: str = "ganttchart.jpeg", include_transport: bool = False):
    """
    ガントチャートを作成して表示する（搬送時間を考慮した版）
    
    Args:
        machine_schedules: マシンごとのスケジュール情報
        machine_list: マシンのリスト（表示順序を決定するため）
        filename: 出力ファイル名
        include_transport: 搬送時間を含めるかどうか
    """
    # 日本語フォントを設定
    plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 図と軸を作成
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 各ジョブに色を割り当て（同じジョブは同じ色）
    job_colors = {}
    colors = plt.colormaps['tab20']
    
    # マシンリストの順序を維持して表示
    machine_order = [machine.name for machine in machine_list]
    
    # 搬送時間を含める場合、Transport行を追加
    if include_transport and 'Transport' in machine_schedules:
        machine_order.append('Transport')
    
    y_positions = {machine_name: idx for idx, machine_name in enumerate(machine_order)}
    
    # タイムラインの最大値を見つける
    max_time = 0
    
    # 各マシンのスケジュールを描画
    for machine_name in machine_order:
        if machine_name not in machine_schedules:
            continue
        
        y_pos = y_positions[machine_name]
        schedules = machine_schedules[machine_name]
        
        for schedule in schedules:
            start = schedule['start_time']
            end = schedule['end_time']
            job_name = schedule['job_name']
            
            # ジョブごとに色を割り当て
            if job_name not in job_colors:
                job_colors[job_name] = colors(len(job_colors))
            
            # バーを作成
            duration = end - start
            
            # 搬送時間と処理時間で異なるスタイルを使用
            if schedule.get('type') == 'transport':
                # 搬送時間は点線のバーで表示
                bar_label = f"{job_name}\n({schedule.get('from_machine', '')}→{schedule.get('to_machine', '')})"
                edge_color = 'red'
                line_style = '--'
                alpha = 0.6
            else:
                # 処理時間は通常のバーで表示
                process_label = schedule.get('process_label', '')
                bar_label = f"{job_name}-{process_label}"
                edge_color = 'black'
                line_style = '-'
                alpha = 1.0
            
            # ガントチャートバーを描画
            if schedule.get('type') == 'transport':
                # 搬送時間は点線のバー
                ax.barh(y_pos, duration, left=start, height=0.5, 
                       color=job_colors[job_name], edgecolor=edge_color, 
                       linewidth=1.5, linestyle=line_style, alpha=alpha, hatch='///')
            else:
                # 処理時間は通常のバー
                ax.barh(y_pos, duration, left=start, height=0.7, 
                       color=job_colors[job_name], edgecolor=edge_color, linewidth=0.5)
            
            # バーの中央にラベルを追加
            if duration > 1:  # 短すぎるバーにはラベルを付けない
                ax.text(start + duration / 2, y_pos, bar_label,
                       ha='center', va='center', fontsize=7, fontweight='bold')
            
            max_time = max(max_time, end)
    
    # 軸の設定
    ax.set_yticks(list(range(len(machine_order))))
    ax.set_yticklabels(machine_order)
    ax.set_xlabel('時間', fontsize=12, fontweight='bold')
    ax.set_ylabel('マシン', fontsize=12, fontweight='bold')
    
    if include_transport:
        ax.set_title('ジョブスケジュール - ガントチャート（搬送時間考慮）', fontsize=14, fontweight='bold')
    else:
        ax.set_title('ジョブスケジュール - ガントチャート', fontsize=14, fontweight='bold')
    
    ax.set_xlim(0, max_time * 1.05)
    ax.grid(True, axis='x', alpha=0.3)
    
    # 凡例を作成
    legend_elements = []
    for job_name, color in job_colors.items():
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=job_name))
    
    # 搬送時間の説明を追加
    if include_transport:
        legend_elements.append(mpatches.Patch(facecolor='gray', edgecolor='red', 
                                             linestyle='--', hatch='///', alpha=0.6, 
                                             label='搬送時間'))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), title='ジョブ')
    
    # レイアウトを調整
    plt.tight_layout()
    
    # 図をJPEGファイルとして保存
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"ガントチャートを {filename} に保存しました。")
    
    # 図を閉じる（メモリ節約のため表示しない）
    plt.close()


def create_text_output(machine_schedules: Dict[str, List], machine_list: List[Machine], output_file: str = "schedule_result.txt"):
    """
    スケジュール結果をテキストファイルに出力する__
    
    Args:
        machine_schedules: マシンごとのスケジュール情報
        machine_list: マシンのリスト
        output_file: 出力ファイル名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("ジョブスケジュール結果\n")
        f.write("=" * 60 + "\n\n")
        
        # マシンごとのスケジュールを出力
        machine_order = [machine.name for machine in machine_list]
        
        for machine_name in machine_order:
            if machine_name not in machine_schedules or len(machine_schedules[machine_name]) == 0:
                continue
            
            f.write(f"【{machine_name}】\n")
            f.write("-" * 40 + "\n")
            
            schedules = machine_schedules[machine_name]
            for schedule in schedules:
                if schedule.get('type') == 'transport':
                    f.write(f"  {schedule['job_name']} 搬送 ({schedule.get('from_machine', '')}→{schedule.get('to_machine', '')}): "
                           f"{schedule['start_time']:.1f} ~ {schedule['end_time']:.1f} "
                           f"(所要時間: {schedule['duration']:.1f})\n")
                else:
                    f.write(f"  {schedule['job_name']}-{schedule.get('process_label', '')}: "
                       f"{schedule['start_time']:.1f} ~ {schedule['end_time']:.1f} "
                       f"(所要時間: {schedule['duration']:.1f})\n")
            f.write("\n")
        
        # 統計情報を出力
        f.write("=" * 60 + "\n")
        f.write("統計情報\n")
        f.write("=" * 60 + "\n")
        
        total_jobs = 0
        total_processes = 0
        max_end_time = 0
        
        for schedules in machine_schedules.values():
            total_processes += len(schedules)
            for schedule in schedules:
                max_end_time = max(max_end_time, schedule['end_time'])
        
        # ジョブ数を計算
        job_names = set()
        for schedules in machine_schedules.values():
            for schedule in schedules:
                job_names.add(schedule['job_name'])
        total_jobs = len(job_names)
        
        f.write(f"総ジョブ数: {total_jobs}\n")
        f.write(f"総工程数: {total_processes}\n")
        f.write(f"最大完了時刻: {max_end_time:.1f}\n")
        f.write(f"使用マシン数: {len([m for m in machine_schedules.values() if len(m) > 0])}\n")
        
        # ジョブごとの完了時刻
        f.write("\n" + "-" * 40 + "\n")
        f.write("ジョブごとの完了時刻\n")
        f.write("-" * 40 + "\n")
        
        job_completion_times = {}
        for schedules in machine_schedules.values():
            for schedule in schedules:
                job_name = schedule['job_name']
                if job_name not in job_completion_times:
                    job_completion_times[job_name] = []
                job_completion_times[job_name].append(schedule['end_time'])
        
        for job_name in sorted(job_completion_times.keys()):
            max_job_time = max(job_completion_times[job_name])
            f.write(f"  {job_name}: {max_job_time:.1f}\n")
    
    print(f"結果を {output_file} に出力しました。")


def print_schedule_summary(machine_schedules: Dict[str, List]):
    """
    スケジュールの概要をテキストで出力する
    
    Args:
        machine_schedules: マシンごとのスケジュール情報
    """
    print("\n========== スケジュール概要 ==========")
    
    for machine_name in sorted(machine_schedules.keys()):
        schedules = machine_schedules[machine_name]
        if len(schedules) == 0:
            continue
        
        print(f"\n【{machine_name}】")
        for schedule in schedules:
            if schedule.get('type') == 'transport':
                print(f"  {schedule['job_name']} 搬送 ({schedule.get('from_machine', '')}→{schedule.get('to_machine', '')}): "
                      f"{schedule['start_time']:.1f} ~ {schedule['end_time']:.1f} "
                      f"(所要時間: {schedule['duration']:.1f})")
            else:
                print(f"  {schedule['job_name']}-{schedule.get('process_label', '')}: "
                  f"{schedule['start_time']:.1f} ~ {schedule['end_time']:.1f} "
                  f"(所要時間: {schedule['duration']:.1f})")


class TeeOutput:
    """ターミナル出力をファイルにも同時に出力するクラス"""
    def __init__(self, file_path):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.stdout.write(text)
        
    def flush(self):
        self.file.flush()
        self.stdout.flush()
        
    def close(self):
        self.file.close()


def main():
    """
    メイン関数：データを読み込み、スケジュールを作成し、ガントチャートを表示する
    """
    # ログファイル名を固定
    log_file = "execution_log.txt"
    
    # ターミナル出力をファイルにも保存
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print("=" * 60)
        print("ジョブスケジューリングシステム実行開始")
        print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print("\nデータ読み込み中...")
        
        # データファイルを読み込む
        job_list = load_job_data('data/case01_job.csv')
        machine_list = load_machine_data('data/case01_machine.csv')
        
        # エッジデータを読み込む（最短経路計算用）
        edge_graph = load_edge_data('data/case01_edge.csv')
        
        # 経路データを読み込む（後方互換性のため）
        route_list = load_route_data('data/case01_route.csv')
        route_dict = convert_route_list_into_dict(route_list)
        
        print(f"読み込まれたジョブ数: {len(job_list)}")
        print(f"読み込まれたマシン数: {len(machine_list)}")
        print(f"読み込まれたエッジ数: {sum(len(neighbors) for neighbors in edge_graph.values()) // 2}")
        print(f"読み込まれた経路数: {len(route_list)}")
        
        # マシンを工程ごとにグループ化
        machine_dict = convert_machine_list_into_dict(machine_list)
        
        print("\nマシンの工程別グループ:")
        for process_name, machines in machine_dict.items():
            print(f"  {process_name}: {[m.name for m in machines]}")
        
        # 遺伝的アルゴリズムでスケジュール最適化（搬送時間を考慮しない）
        print("\n遺伝的アルゴリズムによるスケジュール最適化中（搬送時間考慮なし）...")
        
        # GAパラメータ（簡単に変更可能）
        POPULATION_SIZE = 50      # 個体数
        CROSSOVER_RATE = 0.8      # 交叉確率
        MUTATION_RATE = 0.1       # 突然変異確率
        MAX_GENERATIONS = 100     # 最大世代数
        
        # GAを実行（搬送時間を考慮しない）
        ga = GeneticAlgorithm(
            job_list=job_list,
            machine_dict=machine_dict,
            population_size=POPULATION_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            max_generations=MAX_GENERATIONS,
            route_dict=None  # 搬送時間を考慮しない
        )
        
        best_individual, best_fitness, best_schedule = ga.evolve()
        
        print(f"\n最適化完了（搬送時間考慮なし）!")
        print(f"最良ジョブ順序: {best_individual}")
        print(f"最良makespan: {best_fitness:.2f}")
        
        # 最適スケジュールの詳細を表示
        print("\n--- 最適スケジュール（搬送時間考慮なし） ---")
        print_schedule_summary(best_schedule)
        
        # テキストファイルに詳細結果を出力
        output_file = "schedule_result_optimized.txt"
        print(f"詳細結果を {output_file} に出力中...")
        create_text_output(best_schedule, machine_list, output_file)
        
        # ガントチャートを作成（搬送時間を含まない）
        gantt_file = "ganttchart_optimized.jpeg"
        print(f"ガントチャート（搬送時間考慮なし）を {gantt_file} に出力中...")
        create_gantt_chart(best_schedule, machine_list, gantt_file, include_transport=False)
        
        # 遺伝的アルゴリズムでスケジュール最適化（搬送時間を考慮）
        print("\n" + "=" * 60)
        print("遺伝的アルゴリズムによるスケジュール最適化中（搬送時間考慮）...")
        
        # GAを実行（エッジグラフを渡して最短経路を計算）
        ga_transport = GeneticAlgorithm(
            job_list=job_list,
            machine_dict=machine_dict,
            population_size=POPULATION_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            max_generations=MAX_GENERATIONS,
            route_dict=route_dict,  # 後方互換性のため
            edge_graph=edge_graph  # エッジグラフで最短経路を計算
        )
        
        best_individual_transport, best_fitness_transport, best_schedule_transport = ga_transport.evolve()
        
        print(f"\n最適化完了（搬送時間考慮）!")
        print(f"最良ジョブ順序: {best_individual_transport}")
        print(f"最良makespan: {best_fitness_transport:.2f}")
        
        # 最適スケジュールの詳細を表示
        print("\n--- 最適スケジュール（搬送時間考慮） ---")
        print_schedule_summary(best_schedule_transport)
        
        # テキストファイルに詳細結果を出力
        output_file_transport = "schedule_result_optimized_with_transport.txt"
        print(f"詳細結果を {output_file_transport} に出力中...")
        create_text_output(best_schedule_transport, machine_list, output_file_transport)
        
        # ガントチャートを作成（搬送時間を含む）
        gantt_file_transport = "ganttchart_optimized_with_transport.jpeg"
        print(f"ガントチャート（搬送時間考慮）を {gantt_file_transport} に出力中...")
        create_gantt_chart(best_schedule_transport, machine_list, gantt_file_transport, include_transport=True)
        
        # 比較結果を表示
        print("\n" + "=" * 60)
        print("比較結果")
        print("=" * 60)
        print(f"搬送時間考慮なし: makespan = {best_fitness:.2f}")
        print(f"搬送時間考慮あり: makespan = {best_fitness_transport:.2f}")
        if best_fitness > 0:
            difference = best_fitness_transport - best_fitness
            percentage = (difference / best_fitness) * 100
            print(f"差: {difference:.2f} ({percentage:+.1f}%)")
        
        # 進化の履歴を表示（搬送時間考慮なし）
        print(f"\n進化履歴（搬送時間考慮なし、最初の10世代と最後の10世代）:")
        for i, fitness in enumerate(ga.fitness_history):
            if i < 10 or i >= len(ga.fitness_history) - 10:
                print(f"  世代 {i+1}: {fitness:.2f}")
            elif i == 10 and len(ga.fitness_history) > 20:
                print("  ...")
        
        # 進化の履歴を表示（搬送時間考慮あり）
        print(f"\n進化履歴（搬送時間考慮あり、最初の10世代と最後の10世代）:")
        for i, fitness in enumerate(ga_transport.fitness_history):
            if i < 10 or i >= len(ga_transport.fitness_history) - 10:
                print(f"  世代 {i+1}: {fitness:.2f}")
            elif i == 10 and len(ga_transport.fitness_history) > 20:
                print("  ...")
        
        # 比較のため、ランダムスケジュールも生成（搬送時間考慮なし）
        print(f"\n比較のため、ランダムスケジュールも生成中（搬送時間考慮なし）...")
        random_schedules = generate_schedules(job_list, machine_dict, num_samples=3)
        
        if len(random_schedules) > 0:
            print(f"\nランダムスケジュールとの比較:")
            for idx, schedule in enumerate(random_schedules):
                # makespanを計算
                makespan = 0.0
                for schedules in schedule.values():
                    for sched in schedules:
                        makespan = max(makespan, sched['end_time'])
                
                print(f"  ランダムスケジュール {idx+1}: makespan = {makespan:.2f}")
                
                # ランダムスケジュールも出力
                output_file = f"schedule_result_random_{idx+1}.txt"
                create_text_output(schedule, machine_list, output_file)
                
                gantt_file = f"ganttchart_random_{idx+1}.jpeg"
                create_gantt_chart(schedule, machine_list, gantt_file)
            
            # ランダムスケジュールの平均makespanを計算
            random_makespans = []
            for schedule in random_schedules:
                makespan = 0.0
                for schedules in schedule.values():
                    for sched in schedules:
                        makespan = max(makespan, sched['end_time'])
                random_makespans.append(makespan)
            
            avg_random_makespan = sum(random_makespans) / len(random_makespans)
            improvement = ((avg_random_makespan - best_fitness) / avg_random_makespan) * 100
            
            print(f"\nGA最適化結果: makespan = {best_fitness:.2f}")
            print(f"ランダム平均: makespan = {avg_random_makespan:.2f}")
            print(f"改善効果: ランダム平均より約 {improvement:.1f}% 改善")
        
        print("\n" + "=" * 60)
        print("実行完了")
        print(f"ログファイル: {log_file}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
    finally:
        # 出力を元に戻す
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()

