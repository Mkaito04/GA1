"""
scheduler_with_transport.py - 搬送あり・衝突無視のジョブスケジューリング

このスクリプトは、AMR（搬送）を考慮し、衝突は無視してジョブスケジュールを組み、
ガントチャートで表示し、実行結果をテキストファイルに出力します。

前提条件:
- すべてのジョブは最初にノード1（START）にある
- AMR1, AMR2, AMR3の初期位置はすべてノード1
- AMRは最大3つのジョブを同時に運べる（容量考慮）
- 回送（空荷移動）時間を考慮する
- makespan = 最後のジョブのENDノード到着時刻
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import random
import copy
from datetime import datetime
from package.data_manager import (
    load_job_data, load_machine_data, load_amr_data, load_route_data,
    convert_machine_list_into_dict, convert_route_list_into_dict
)
from package.classes import Job, Machine, Process, Amr, Route, Port
from typing import List, Dict, Optional, Tuple


def get_amr_for_transport(prev_location: str, next_location: str) -> str:
    """
    搬送に使用するAMRを決定する固定ルール
    
    Args:
        prev_location: 搬送元（'START'または工程名）
        next_location: 搬送先（工程名または'END'）
    
    Returns:
        AMR名
    """
    if prev_location == 'START' and next_location == 'A':
        return 'AMR1'
    elif prev_location == 'A' and next_location == 'B':
        return 'AMR2'
    elif prev_location == 'B' and next_location == 'C':
        return 'AMR3'
    elif prev_location == 'C' and next_location == 'D':
        return 'AMR3'
    elif prev_location == 'D' and next_location == 'END':
        return 'AMR3'
    return None


class AMRState:
    """AMRの状態を管理するクラス"""
    
    def __init__(self, name: str, initial_node: str = '1', max_capacity: int = 3):
        self.name = name
        self.current_node = initial_node
        self.available_time = 0.0
        self.max_capacity = max_capacity
        self.timeline = []  # [(start_time, end_time, from_node, to_node, job_names, action_type)]
    
    def add_timeline(self, start_time: float, end_time: float, from_node: str, 
                    to_node: str, job_names: List[str], action_type: str):
        """タイムラインにエントリを追加"""
        self.timeline.append({
            'start_time': start_time,
            'end_time': end_time,
            'from_node': from_node,
            'to_node': to_node,
            'job_names': job_names,
            'action_type': action_type  # 'transport' or 'return'
        })
        self.current_node = to_node
        self.available_time = end_time


class GeneticAlgorithmWithTransport:
    """
    搬送を考慮した遺伝的アルゴリズム（GA）によるジョブスケジュール最適化クラス
    """
    
    def __init__(self, job_list: List[Job], machine_dict: Dict[str, List[Machine]],
                 machine_list: List[Machine], route_dict: Dict[str, Route],
                 population_size: int = 50, crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1, max_generations: int = 100):
        self.job_list = job_list
        self.machine_dict = machine_dict
        self.machine_list = machine_list
        self.route_dict = route_dict
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.max_generations = max_generations
        
        # ジョブIDのリストを作成
        self.job_ids = [job.name for job in job_list]
        
        # マシン名からマシンオブジェクトへのマッピング
        self.machine_by_name = {m.name: m for m in machine_list}
        
        # 最良解を保存
        self.best_individual = None
        self.best_fitness = float('inf')
        self.best_schedule = None
        self.best_amr_timelines = None
        
        # 進化の履歴
        self.fitness_history = []
    
    def get_route_cost(self, from_node: str, to_node: str) -> float:
        """2つのノード間の経路コスト（時間）を取得"""
        if from_node == to_node:
            return 0.0
        route_key = f"{from_node}->{to_node}"
        if route_key in self.route_dict:
            return self.route_dict[route_key].cost
        return 0.0  # 経路が存在しない場合は0
    
    def create_individual(self) -> Dict:
        """ランダムな個体を生成"""
        job_order = self.job_ids.copy()
        random.shuffle(job_order)
        
        machine_assignment = {}
        for job in self.job_list:
            for process in job.process_list:
                process_name = process.label
                if process_name == 'END':
                    continue
                available_machines = self.machine_dict.get(process_name, [])
                if len(available_machines) > 0:
                    selected_machine = random.choice(available_machines)
                    machine_assignment[(job.name, process_name)] = selected_machine.name
        
        return {
            'job_order': job_order,
            'machine_assignment': machine_assignment
        }
    
    def evaluate_fitness(self, individual: Dict) -> Tuple[float, Dict, Dict]:
        """
        個体の適応度（makespan）を評価
        搬送時間を考慮し、makespanは最後のジョブのENDノード到着時刻
        """
        job_order = individual['job_order']
        machine_assignment = individual['machine_assignment']
        
        # ジョブ順序に基づいてジョブリストを再構築
        ordered_jobs = []
        for job_id in job_order:
            for job in self.job_list:
                if job.name == job_id:
                    ordered_jobs.append(job)
                    break
        
        # スケジュールを評価
        machine_schedules, amr_states, end_arrival_times = \
            self._evaluate_schedule_with_transport(ordered_jobs, machine_assignment)
        
        # AMRタイムラインを抽出
        amr_timelines = {name: state.timeline for name, state in amr_states.items()}
        
        # makespanは最後のジョブのENDノード到着時刻
        if end_arrival_times:
            makespan = max(end_arrival_times.values())
        else:
            makespan = float('inf')
        
        return makespan, machine_schedules, amr_timelines
    
    def _evaluate_schedule_with_transport(self, ordered_jobs: List[Job],
                                          machine_assignment: Dict[Tuple[str, str], str]
                                          ) -> Tuple[Dict, Dict, Dict]:
        """搬送時間を考慮したスケジュール評価"""
        
        # マシンスケジュールの初期化
        machine_schedules: Dict[str, List] = {}
        machine_available_time: Dict[str, float] = {}
        for machine in self.machine_list:
            machine_schedules[machine.name] = []
            machine_available_time[machine.name] = 0.0
        
        # AMR状態の初期化
        amr_states = {
            'AMR1': AMRState('AMR1', '1', 3),
            'AMR2': AMRState('AMR2', '1', 3),
            'AMR3': AMRState('AMR3', '1', 3)
        }
        
        # ジョブの状態を追跡
        job_current_node: Dict[str, str] = {}  # ジョブの現在位置
        job_available_time: Dict[str, float] = {}  # ジョブが搬送可能になる時刻
        job_process_end_time: Dict[str, Dict[str, float]] = {}  # ジョブごとの各工程完了時刻
        end_arrival_times: Dict[str, float] = {}  # ENDノード到着時刻
        
        # 全ジョブは最初にノード1にある
        for job in ordered_jobs:
            job_current_node[job.name] = '1'
            job_available_time[job.name] = 0.0
            job_process_end_time[job.name] = {}
        
        # Phase 1: START → 工程A（AMR1が担当）
        self._transport_start_to_A(ordered_jobs, machine_assignment, amr_states,
                                   job_current_node, job_available_time)
        
        # Phase 2: 各工程の処理と工程間搬送
        for job in ordered_jobs:
            prev_process_label = 'START'
            
            for i, process in enumerate(job.process_list):
                process_name = process.label
                process_time = process.time
                
                if process_name == 'END':
                    # D→ENDの搬送
                    self._transport_to_end(job, prev_process_label, machine_assignment,
                                          amr_states, job_current_node, job_available_time,
                                          job_process_end_time, end_arrival_times)
                    continue
                
                # マシン割り当てを取得
                assignment_key = (job.name, process_name)
                if assignment_key not in machine_assignment:
                    continue
                
                assigned_machine_name = machine_assignment[assignment_key]
                assigned_machine = self.machine_by_name.get(assigned_machine_name)
                
                if assigned_machine is None:
                    continue
                
                # 工程間搬送（A→B, B→C, C→D）
                if prev_process_label != 'START' and prev_process_label != process_name:
                    self._transport_between_processes(
                        job, prev_process_label, process_name, machine_assignment,
                        amr_states, job_current_node, job_available_time, job_process_end_time
                    )
                
                # ジョブが搬送完了してマシンの入口に到着する時刻
                transport_arrival_time = job_available_time[job.name]
                
                # マシンが利用可能になる時刻と比較
                start_time = max(machine_available_time[assigned_machine_name], transport_arrival_time)
                end_time = start_time + process_time
                machine_available_time[assigned_machine_name] = end_time
                
                # スケジュール情報を保存
                schedule_item = {
                    'job_name': job.name,
                    'process_label': process.label,
                    'process_index': i,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': process_time
                }
                machine_schedules[assigned_machine_name].append(schedule_item)
                
                # ジョブの状態を更新
                job_current_node[job.name] = assigned_machine.exit_port.node
                job_available_time[job.name] = end_time
                job_process_end_time[job.name][process_name] = end_time
                
                prev_process_label = process_name
        
        return machine_schedules, amr_states, end_arrival_times
    
    def _transport_start_to_A(self, ordered_jobs: List[Job], machine_assignment: Dict,
                              amr_states: Dict, job_current_node: Dict,
                              job_available_time: Dict):
        """START（ノード1）から工程Aへの搬送（AMR1）"""
        amr = amr_states['AMR1']
        
        # ジョブを3つずつグループ化
        jobs_to_transport = []
        for job in ordered_jobs:
            assignment_key = (job.name, 'A')
            if assignment_key in machine_assignment:
                jobs_to_transport.append(job)
        
        # 3つずつまとめて搬送
        batch_size = amr.max_capacity
        for i in range(0, len(jobs_to_transport), batch_size):
            batch = jobs_to_transport[i:i+batch_size]
            
            if not batch:
                continue
            
            # 各ジョブの搬送先マシンを取得
            destinations = []
            for job in batch:
                machine_name = machine_assignment[(job.name, 'A')]
                machine = self.machine_by_name[machine_name]
                destinations.append(machine.entrance_port.node)
            
            # 搬送元（現在位置）から搬送先への搬送
            from_node = amr.current_node
            
            # 最初の搬送先への移動時間（最も遠い搬送先を基準）
            max_transport_time = 0.0
            farthest_node = destinations[0]
            for dest in destinations:
                transport_time = self.get_route_cost(from_node, dest)
                if transport_time > max_transport_time:
                    max_transport_time = transport_time
                    farthest_node = dest
            
            # 搬送を実行
            start_time = amr.available_time
            end_time = start_time + max_transport_time
            
            amr.add_timeline(start_time, end_time, from_node, farthest_node,
                           [job.name for job in batch], 'transport')
            
            # 各ジョブの状態を更新
            for job in batch:
                machine_name = machine_assignment[(job.name, 'A')]
                machine = self.machine_by_name[machine_name]
                job_current_node[job.name] = machine.entrance_port.node
                job_available_time[job.name] = end_time
            
            # 回送（ノード1に戻る）- 次のバッチがある場合のみ
            if i + batch_size < len(jobs_to_transport):
                return_time = self.get_route_cost(farthest_node, '1')
                return_start = end_time
                return_end = return_start + return_time
                amr.add_timeline(return_start, return_end, farthest_node, '1', [], 'return')
    
    def _transport_between_processes(self, job: Job, prev_process: str, next_process: str,
                                     machine_assignment: Dict, amr_states: Dict,
                                     job_current_node: Dict, job_available_time: Dict,
                                     job_process_end_time: Dict):
        """工程間の搬送（A→B, B→C, C→D）"""
        amr_name = get_amr_for_transport(prev_process, next_process)
        if amr_name is None:
            return
        
        amr = amr_states[amr_name]
        
        # 搬送元：前工程マシンの出口ポート
        prev_machine_name = machine_assignment.get((job.name, prev_process))
        if prev_machine_name is None:
            return
        prev_machine = self.machine_by_name.get(prev_machine_name)
        if prev_machine is None:
            return
        from_node = prev_machine.exit_port.node
        
        # 搬送先：次工程マシンの入口ポート
        next_machine_name = machine_assignment.get((job.name, next_process))
        if next_machine_name is None:
            return
        next_machine = self.machine_by_name.get(next_machine_name)
        if next_machine is None:
            return
        to_node = next_machine.entrance_port.node
        
        # 前工程の完了時刻
        prev_end_time = job_process_end_time[job.name].get(prev_process, 0.0)
        
        # AMRが利用可能になる時刻
        amr_ready_time = amr.available_time
        
        # AMRの現在位置から搬送元への回送時間
        return_time = 0.0
        if amr.current_node != from_node:
            return_time = self.get_route_cost(amr.current_node, from_node)
            # 回送をタイムラインに追加
            return_start = max(amr_ready_time, prev_end_time - return_time)
            if return_start < amr_ready_time:
                return_start = amr_ready_time
            return_end = return_start + return_time
            amr.add_timeline(return_start, return_end, amr.current_node, from_node, [], 'return')
            amr_ready_time = return_end
        
        # 搬送開始時刻（前工程完了かつAMRが搬送元にいる）
        transport_start = max(amr_ready_time, prev_end_time)
        
        # 搬送時間
        transport_time = self.get_route_cost(from_node, to_node)
        transport_end = transport_start + transport_time
        
        # 搬送をタイムラインに追加
        amr.add_timeline(transport_start, transport_end, from_node, to_node, [job.name], 'transport')
        
        # ジョブの状態を更新
        job_current_node[job.name] = to_node
        job_available_time[job.name] = transport_end
    
    def _transport_to_end(self, job: Job, prev_process: str, machine_assignment: Dict,
                         amr_states: Dict, job_current_node: Dict, job_available_time: Dict,
                         job_process_end_time: Dict, end_arrival_times: Dict):
        """D工程からENDノードへの搬送（AMR3）"""
        amr_name = get_amr_for_transport('D', 'END')
        if amr_name is None:
            return
        
        amr = amr_states[amr_name]
        
        # 搬送元：D工程マシンの出口ポート
        prev_machine_name = machine_assignment.get((job.name, 'D'))
        if prev_machine_name is None:
            return
        prev_machine = self.machine_by_name.get(prev_machine_name)
        if prev_machine is None:
            return
        from_node = prev_machine.exit_port.node
        
        # 搬送先：ENDノード（ノード7）
        to_node = '7'
        
        # D工程の完了時刻
        prev_end_time = job_process_end_time[job.name].get('D', 0.0)
        
        # AMRが利用可能になる時刻
        amr_ready_time = amr.available_time
        
        # AMRの現在位置から搬送元への回送時間
        if amr.current_node != from_node:
            return_time = self.get_route_cost(amr.current_node, from_node)
            return_start = max(amr_ready_time, prev_end_time - return_time)
            if return_start < amr_ready_time:
                return_start = amr_ready_time
            return_end = return_start + return_time
            amr.add_timeline(return_start, return_end, amr.current_node, from_node, [], 'return')
            amr_ready_time = return_end
        
        # 搬送開始時刻
        transport_start = max(amr_ready_time, prev_end_time)
        
        # 搬送時間
        transport_time = self.get_route_cost(from_node, to_node)
        transport_end = transport_start + transport_time
        
        # 搬送をタイムラインに追加
        amr.add_timeline(transport_start, transport_end, from_node, to_node, [job.name], 'transport')
        
        # ENDノード到着時刻を記録
        end_arrival_times[job.name] = transport_end
        
        # ジョブの状態を更新
        job_current_node[job.name] = to_node
        job_available_time[job.name] = transport_end
    
    def tournament_selection(self, population: List[Dict], fitness_values: List[float],
                           tournament_size: int = 3) -> Dict:
        """トーナメント選択"""
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_values[i] for i in tournament_indices]
        winner_index = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return copy.deepcopy(population[winner_index])
    
    def crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """交叉操作"""
        job_order1 = parent1['job_order']
        job_order2 = parent2['job_order']
        
        if len(job_order1) <= 2:
            child1_job_order = job_order1.copy()
            child2_job_order = job_order2.copy()
        else:
            # 順序交叉（OX）
            start = random.randint(0, len(job_order1) - 2)
            end = random.randint(start + 1, len(job_order1) - 1)
            
            child1_job_order = [None] * len(job_order1)
            child1_job_order[start:end+1] = job_order1[start:end+1]
            remaining = [job for job in job_order2 if job not in child1_job_order[start:end+1]]
            idx = 0
            for i in range(len(child1_job_order)):
                if child1_job_order[i] is None:
                    child1_job_order[i] = remaining[idx]
                    idx += 1
            
            child2_job_order = [None] * len(job_order2)
            child2_job_order[start:end+1] = job_order2[start:end+1]
            remaining = [job for job in job_order1 if job not in child2_job_order[start:end+1]]
            idx = 0
            for i in range(len(child2_job_order)):
                if child2_job_order[i] is None:
                    child2_job_order[i] = remaining[idx]
                    idx += 1
        
        # マシン割り当ての一様交叉
        machine_assignment1 = parent1['machine_assignment']
        machine_assignment2 = parent2['machine_assignment']
        child1_machine_assignment = {}
        child2_machine_assignment = {}
        all_keys = set(machine_assignment1.keys()) | set(machine_assignment2.keys())
        
        for key in all_keys:
            if random.random() < 0.5:
                if key in machine_assignment1:
                    child1_machine_assignment[key] = machine_assignment1[key]
                if key in machine_assignment2:
                    child2_machine_assignment[key] = machine_assignment2[key]
            else:
                if key in machine_assignment2:
                    child1_machine_assignment[key] = machine_assignment2[key]
                if key in machine_assignment1:
                    child2_machine_assignment[key] = machine_assignment1[key]
        
        child1 = {'job_order': child1_job_order, 'machine_assignment': child1_machine_assignment}
        child2 = {'job_order': child2_job_order, 'machine_assignment': child2_machine_assignment}
        
        return child1, child2
    
    def mutation(self, individual: Dict) -> Dict:
        """突然変異"""
        mutated = copy.deepcopy(individual)
        
        # ジョブ順序のスワップ
        job_order = mutated['job_order']
        if len(job_order) > 1:
            i, j = random.sample(range(len(job_order)), 2)
            job_order[i], job_order[j] = job_order[j], job_order[i]
        
        # マシン割り当ての変更
        machine_assignment = mutated['machine_assignment']
        mutation_keys = list(machine_assignment.keys())
        if len(mutation_keys) > 0:
            num_mutations = max(1, len(mutation_keys) // 10)
            keys_to_mutate = random.sample(mutation_keys, min(num_mutations, len(mutation_keys)))
            
            for key in keys_to_mutate:
                job_name, process_name = key
                available_machines = self.machine_dict.get(process_name, [])
                if len(available_machines) > 0:
                    selected_machine = random.choice(available_machines)
                    machine_assignment[key] = selected_machine.name
        
        return mutated
    
    def evolve(self) -> Tuple[Dict, float, Dict, Dict]:
        """遺伝的アルゴリズムの進化を実行"""
        print(f"遺伝的アルゴリズム開始: 個体数={self.population_size}, 世代数={self.max_generations}")
        print(f"交叉確率={self.crossover_rate}, 突然変異確率={self.mutation_rate}")
        print("最適化対象: ジョブ順序 + マシン割り当て（搬送時間考慮）")
        
        # 初期個体群を生成
        population = [self.create_individual() for _ in range(self.population_size)]
        
        for generation in range(self.max_generations):
            # 適応度を評価
            fitness_results = [self.evaluate_fitness(ind) for ind in population]
            fitness_values = [r[0] for r in fitness_results]
            
            # 最良解を更新
            min_fitness = min(fitness_values)
            if min_fitness < self.best_fitness:
                self.best_fitness = min_fitness
                best_idx = fitness_values.index(min_fitness)
                self.best_individual = copy.deepcopy(population[best_idx])
                self.best_schedule = fitness_results[best_idx][1]
                self.best_amr_timelines = fitness_results[best_idx][2]
            
            self.fitness_history.append(min_fitness)
            
            if generation % 10 == 0 or generation == self.max_generations - 1:
                print(f"世代 {generation+1}/{self.max_generations}: 最良適応度 = {min_fitness:.2f}")
            
            # 新しい世代を生成
            new_population = []
            
            # エリート保存
            elite_idx = fitness_values.index(min_fitness)
            new_population.append(copy.deepcopy(population[elite_idx]))
            
            # 残りの個体を生成
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection(population, fitness_values)
                parent2 = self.tournament_selection(population, fitness_values)
                
                if random.random() < self.crossover_rate:
                    child1, child2 = self.crossover(parent1, parent2)
                else:
                    child1 = copy.deepcopy(parent1)
                    child2 = copy.deepcopy(parent2)
                
                if random.random() < self.mutation_rate:
                    child1 = self.mutation(child1)
                if random.random() < self.mutation_rate:
                    child2 = self.mutation(child2)
                
                new_population.extend([child1, child2])
            
            if len(new_population) > self.population_size:
                new_population = new_population[:self.population_size]
            
            population = new_population
        
        print(f"遺伝的アルゴリズム完了: 最良適応度 = {self.best_fitness:.2f}")
        return self.best_individual, self.best_fitness, self.best_schedule, self.best_amr_timelines


def create_gantt_chart_with_transport(machine_schedules: Dict[str, List], 
                                      amr_timelines: Dict[str, List],
                                      machine_list: List[Machine],
                                      filename: str = "ganttchart_collision.jpeg"):
    """搬送を含むガントチャートを作成"""
    plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False
    
    # マシンとAMRを合わせた表示
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # 各ジョブに色を割り当て
    job_colors = {}
    colors = plt.colormaps['tab20']
    
    # マシンリストの順序を維持
    machine_order = [machine.name for machine in machine_list]
    y_positions = {machine_name: idx for idx, machine_name in enumerate(machine_order)}
    
    max_time = 0
    
    # マシンスケジュールを描画
    for machine_name in machine_order:
        if machine_name not in machine_schedules:
            continue
        
        y_pos = y_positions[machine_name]
        schedules = machine_schedules[machine_name]
        
        for schedule in schedules:
            start = schedule['start_time']
            end = schedule['end_time']
            job_name = schedule['job_name']
            process_label = schedule['process_label']
            
            if job_name not in job_colors:
                job_colors[job_name] = colors(len(job_colors) % 20)
            
            duration = end - start
            bar_label = f"{job_name}-{process_label}"
            
            ax1.barh(y_pos, duration, left=start, height=0.7,
                    color=job_colors[job_name], edgecolor='black', linewidth=0.5)
            
            if duration > 5:
                ax1.text(start + duration / 2, y_pos, bar_label,
                        ha='center', va='center', fontsize=7, fontweight='bold')
            
            max_time = max(max_time, end)
    
    ax1.set_yticks(list(range(len(machine_order))))
    ax1.set_yticklabels(machine_order)
    ax1.set_xlabel('時間', fontsize=12, fontweight='bold')
    ax1.set_ylabel('マシン', fontsize=12, fontweight='bold')
    ax1.set_title('マシンスケジュール', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, max_time * 1.05)
    ax1.grid(True, axis='x', alpha=0.3)
    
    # AMRスケジュールを描画
    amr_order = ['AMR1', 'AMR2', 'AMR3']
    amr_y_positions = {amr_name: idx for idx, amr_name in enumerate(amr_order)}
    
    for amr_name in amr_order:
        if amr_name not in amr_timelines:
            continue
        
        y_pos = amr_y_positions[amr_name]
        timeline = amr_timelines[amr_name]
        
        for entry in timeline:
            start = entry['start_time']
            end = entry['end_time']
            job_names = entry['job_names']
            action_type = entry['action_type']
            from_node = entry['from_node']
            to_node = entry['to_node']
            
            duration = end - start
            
            if action_type == 'transport':
                # 搬送：ジョブの色を使用
                if job_names:
                    color = job_colors.get(job_names[0], 'lightblue')
                else:
                    color = 'lightblue'
                label = ','.join(job_names) if job_names else ''
            else:
                # 回送：グレー
                color = 'lightgray'
                label = '回送'
            
            ax2.barh(y_pos, duration, left=start, height=0.7,
                    color=color, edgecolor='black', linewidth=0.5)
            
            if duration > 5:
                ax2.text(start + duration / 2, y_pos, f"{from_node}→{to_node}",
                        ha='center', va='center', fontsize=6)
            
            max_time = max(max_time, end)
    
    ax2.set_yticks(list(range(len(amr_order))))
    ax2.set_yticklabels(amr_order)
    ax2.set_xlabel('時間', fontsize=12, fontweight='bold')
    ax2.set_ylabel('AMR', fontsize=12, fontweight='bold')
    ax2.set_title('AMRスケジュール', fontsize=14, fontweight='bold')
    ax2.set_xlim(0, max_time * 1.05)
    ax2.grid(True, axis='x', alpha=0.3)
    
    # 凡例
    legend_elements = []
    for job_name, color in job_colors.items():
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=job_name))
    legend_elements.append(mpatches.Patch(facecolor='lightgray', edgecolor='black', label='回送'))
    
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99), title='ジョブ')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"ガントチャートを {filename} に保存しました。")
    plt.close()


def print_schedule_summary_with_transport(machine_schedules: Dict[str, List], 
                                          amr_timelines: Dict[str, List]):
    """スケジュールの概要を表示"""
    print("\n========== マシンスケジュール概要 ==========")
    
    for machine_name in sorted(machine_schedules.keys()):
        schedules = machine_schedules[machine_name]
        if len(schedules) == 0:
            continue
        
        print(f"\n【{machine_name}】")
        for schedule in schedules:
            print(f"  {schedule['job_name']}-{schedule['process_label']}: "
                  f"{schedule['start_time']:.1f} ~ {schedule['end_time']:.1f} "
                  f"(所要時間: {schedule['duration']:.1f})")
    
    print("\n========== AMRスケジュール概要 ==========")
    
    for amr_name in ['AMR1', 'AMR2', 'AMR3']:
        if amr_name not in amr_timelines:
            continue
        
        timeline = amr_timelines[amr_name]
        if len(timeline) == 0:
            continue
        
        print(f"\n【{amr_name}】")
        for entry in timeline:
            job_str = ','.join(entry['job_names']) if entry['job_names'] else '(空荷)'
            action = '搬送' if entry['action_type'] == 'transport' else '回送'
            print(f"  {action}: {job_str} {entry['from_node']} → {entry['to_node']}: "
                  f"{entry['start_time']:.1f} ~ {entry['end_time']:.1f} "
                  f"(所要時間: {entry['end_time'] - entry['start_time']:.1f})")


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
    """メイン関数"""
    log_file = "execution_log_transport.txt"
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print("=" * 60)
        print("搬送あり・衝突無視ジョブスケジューリングシステム実行開始")
        print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        print("\nデータ読み込み中...")
        
        # データファイルを読み込む
        job_list = load_job_data('data/case01_job.csv')
        machine_list = load_machine_data('data/case01_machine.csv')
        amr_list = load_amr_data('data/case01_amr.csv')
        route_list = load_route_data('data/case01_route.csv')
        
        print(f"読み込まれたジョブ数: {len(job_list)}")
        print(f"読み込まれたマシン数: {len(machine_list)}")
        print(f"読み込まれたAMR数: {len(amr_list)}")
        print(f"読み込まれたルート数: {len(route_list)}")
        
        # 辞書に変換
        machine_dict = convert_machine_list_into_dict(machine_list)
        route_dict = convert_route_list_into_dict(route_list)
        
        print("\nマシンの工程別グループ:")
        for process_name, machines in machine_dict.items():
            print(f"  {process_name}: {[m.name for m in machines]}")
        
        print("\nAMR情報:")
        for amr in amr_list:
            print(f"  {amr.name}: 容量={amr.max_capacity}, 担当工程={amr.target_process}")
        
        # 遺伝的アルゴリズムでスケジュール最適化
        print("\n遺伝的アルゴリズムによるスケジュール最適化中...")
        
        # GAパラメータ
        POPULATION_SIZE = 50
        CROSSOVER_RATE = 0.8
        MUTATION_RATE = 0.1
        MAX_GENERATIONS = 100
        
        ga = GeneticAlgorithmWithTransport(
            job_list=job_list,
            machine_dict=machine_dict,
            machine_list=machine_list,
            route_dict=route_dict,
            population_size=POPULATION_SIZE,
            crossover_rate=CROSSOVER_RATE,
            mutation_rate=MUTATION_RATE,
            max_generations=MAX_GENERATIONS
        )
        
        best_individual, best_fitness, best_schedule, best_amr_timelines = ga.evolve()
        
        print(f"\n最適化完了!")
        print(f"最良ジョブ順序: {best_individual['job_order']}")
        print(f"最良makespan（ENDノード到着時刻）: {best_fitness:.2f}")
        
        # スケジュールの詳細を表示
        print_schedule_summary_with_transport(best_schedule, best_amr_timelines)
        
        # ガントチャートを作成
        gantt_file = "ganttchart_collision.jpeg"
        print(f"\nガントチャートを {gantt_file} に出力中...")
        create_gantt_chart_with_transport(best_schedule, best_amr_timelines, machine_list, gantt_file)
        
        # 進化の履歴を表示
        print(f"\n進化履歴（最初の10世代と最後の10世代）:")
        for i, fitness in enumerate(ga.fitness_history):
            if i < 10 or i >= len(ga.fitness_history) - 10:
                print(f"  世代 {i+1}: {fitness:.2f}")
            elif i == 10 and len(ga.fitness_history) > 20:
                print("  ...")
        
        # 統計情報
        max_machine_time = 0.0
        for schedules in best_schedule.values():
            for schedule in schedules:
                max_machine_time = max(max_machine_time, schedule['end_time'])
        
        print(f"\n========== 統計情報 ==========")
        print(f"最終マシン処理完了時刻: {max_machine_time:.2f}")
        print(f"最終ENDノード到着時刻（makespan）: {best_fitness:.2f}")
        print(f"搬送による追加時間: {best_fitness - max_machine_time:.2f}")
        
        print("\n" + "=" * 60)
        print("実行完了")
        print(f"ログファイル: {log_file}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nエラーが発生しました: {str(e)}")
        import traceback
        print(traceback.format_exc())
        
    finally:
        sys.stdout = tee.stdout
        tee.close()


if __name__ == "__main__":
    main()

