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
import time
import os
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
    
    def get_route_sequence(self, from_node: str, to_node: str) -> List[str]:
        """2つのノード間の経路シーケンス（中間ノードを含む）を取得"""
        if from_node == to_node:
            return [from_node]
        route_key = f"{from_node}->{to_node}"
        if route_key in self.route_dict:
            route = self.route_dict[route_key]
            return route.sequence.copy() if hasattr(route, 'sequence') and route.sequence else [from_node, to_node]
        return [from_node, to_node]  # 経路が存在しない場合は直接経路を返す
    
    def _calculate_all_process_end_times(self, ordered_jobs: List[Job], 
                                         machine_assignment: Dict[Tuple[str, str], str],
                                         amr_states: Dict,
                                         job_available_time: Dict) -> Dict[str, Dict[str, float]]:
        """
        全ジョブの各工程完了時刻を事前計算（搬送時間を考慮しない簡易版）
        
        Args:
            ordered_jobs: ジョブ順序リスト
            machine_assignment: マシン割り当て
            amr_states: AMR状態（START→Aの搬送後に使用）
            job_available_time: ジョブが搬送可能になる時刻（START→Aの搬送後）
        
        Returns:
            ジョブごとの各工程完了時刻の辞書
        """
        job_process_end_time: Dict[str, Dict[str, float]] = {}
        machine_available_time: Dict[str, float] = {}
        
        # マシンの利用可能時刻を初期化
        for machine in self.machine_list:
            machine_available_time[machine.name] = 0.0
        
        # 全ジョブの各工程完了時刻を初期化
        for job in ordered_jobs:
            job_process_end_time[job.name] = {}
        
        # 各工程の処理完了時刻を計算
        for job in ordered_jobs:
            prev_process_label = 'START'
            
            for i, process in enumerate(job.process_list):
                process_name = process.label
                process_time = process.time
                
                if process_name == 'END':
                    continue
                
                # マシン割り当てを取得
                assignment_key = (job.name, process_name)
                if assignment_key not in machine_assignment:
                    continue
                
                assigned_machine_name = machine_assignment[assignment_key]
                
                # 前工程の完了時刻を取得
                prev_end_time = 0.0
                if prev_process_label == 'START':
                    # A工程の場合：START→Aの搬送完了時刻を使用
                    arrival_time = job_available_time.get(job.name, 0.0)
                    start_time = max(machine_available_time[assigned_machine_name], arrival_time)
                else:
                    prev_end_time = job_process_end_time[job.name].get(prev_process_label, 0.0)
                    start_time = max(machine_available_time[assigned_machine_name], prev_end_time)
                
                end_time = start_time + process_time
                machine_available_time[assigned_machine_name] = end_time
                job_process_end_time[job.name][process_name] = end_time
                
                prev_process_label = process_name
        
        return job_process_end_time
    
    def _optimize_visit_order(self, nodes: List[str], start_node: str) -> List[str]:
        """
        複数のノードを訪問する最適な順序を決定（最近傍法を使用）
        
        Args:
            nodes: 訪問するノードのリスト
            start_node: 開始ノード
        
        Returns:
            最適な訪問順序
        """
        if len(nodes) == 0:
            return [start_node]
        
        if len(nodes) == 1:
            return [start_node, nodes[0]]
        
        # 最近傍法を使用
        order = [start_node]
        remaining = nodes.copy()
        current = start_node
        
        while remaining:
            nearest = min(remaining, key=lambda n: self.get_route_cost(current, n))
            order.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return order
    
    def _plan_route_for_batch(self, batch: List[Dict], current_node: str,
                              amr_ready_time: float, max_prev_end_time: float,
                              transport_plans: Dict = None, prev_process: str = None,
                              next_process: str = None) -> Dict:
        """
        バッチ内のジョブを運ぶための経路計画（AMR経路計画を使用）
        
        Args:
            batch: バッチ内のジョブ情報のリスト
                [{
                    'job': Job,
                    'prev_end_time': float,
                    'from_node': str,
                    'to_node': str
                }, ...]
            current_node: AMRの現在位置
            amr_ready_time: AMRの利用可能時刻
            max_prev_end_time: バッチ内の全ジョブの前工程完了時刻の最大値
            transport_plans: AMR経路計画
            prev_process: 前工程名
            next_process: 次工程名
        
        Returns:
            経路計画の辞書
        """
        # 搬送元ノードのリストを取得（重複を除去）
        from_nodes = list(set([item['from_node'] for item in batch]))
        
        # 搬送先ノードのリストを取得（重複を除去）
        to_nodes = list(set([item['to_node'] for item in batch]))
        
        # 訪問順序を決定（常に最近傍法を使用）
        if len(from_nodes) == 1:
            if len(to_nodes) == 1:
                visit_order = from_nodes + to_nodes
            else:
                visit_order = from_nodes + to_nodes
        else:
            visit_order = self._optimize_visit_order(from_nodes, current_node)
            if len(to_nodes) > 1:
                visit_order.extend(self._optimize_visit_order(to_nodes, visit_order[-1]))
            else:
                visit_order.extend(to_nodes)
        
        # 訪問順序に中間ノードを展開（経路上の中間ノードでジョブを拾えるようにする）
        expanded_visit_order = []
        for i in range(len(visit_order)):
            if i == 0:
                expanded_visit_order.append(visit_order[i])
            else:
                # 前のノードから現在のノードへの経路の中間ノードを取得
                prev_node = visit_order[i-1]
                current_node_in_route = visit_order[i]
                route_sequence = self.get_route_sequence(prev_node, current_node_in_route)
                
                # 中間ノードを追加（最初のノードは前回追加済みなので、2番目以降を追加）
                if len(route_sequence) > 1:
                    # 前のノードを除いて、中間ノードと現在のノードを追加
                    for j in range(1, len(route_sequence)):
                        node = route_sequence[j]
                        # 重複を避ける（直前のノードと同じ場合はスキップ）
                        if not expanded_visit_order or expanded_visit_order[-1] != node:
                            expanded_visit_order.append(node)
                else:
                    # 経路が存在しない場合、現在のノードを追加
                    if not expanded_visit_order or expanded_visit_order[-1] != current_node_in_route:
                        expanded_visit_order.append(current_node_in_route)
        
        # 総移動時間を計算（展開後の経路で）
        total_time = 0.0
        for i in range(len(expanded_visit_order) - 1):
            total_time += self.get_route_cost(expanded_visit_order[i], expanded_visit_order[i+1])
        
        # 最早開始時刻を計算
        earliest_start_time = max(amr_ready_time, max_prev_end_time)
        
        return {
            'visit_order': expanded_visit_order,  # 中間ノードを含む展開後の経路
            'total_time': total_time,
            'earliest_start_time': earliest_start_time,
            'start_node': expanded_visit_order[0] if expanded_visit_order else current_node,
            'end_node': expanded_visit_order[-1] if expanded_visit_order else current_node
        }
    
    def _find_jobs_completing_during_route(self, route_plan: Dict, amr_ready_time: float,
                                           amr_current_node: str, max_prev_end_time: float,
                                           prev_process: str, next_process: str,
                                           ordered_jobs: List[Job], machine_assignment: Dict,
                                           job_process_end_time: Dict,
                                           existing_batch_job_names: set,
                                           max_capacity: int) -> List[Dict]:
        """
        AMRが移動中に完了するジョブを見つける（すべてのリソースを確認）
        
        Args:
            route_plan: 経路計画の辞書
            amr_ready_time: AMRの利用可能時刻
            amr_current_node: AMRの現在位置
            max_prev_end_time: バッチ内の全ジョブの前工程完了時刻の最大値
            prev_process: 前工程名
            next_process: 次工程名
            ordered_jobs: ジョブ順序リスト
            machine_assignment: マシン割り当て
            job_process_end_time: ジョブごとの各工程完了時刻
            existing_batch_job_names: 既にバッチに含まれているジョブ名のセット
            max_capacity: AMRの最大積載量
        
        Returns:
            追加で拾えるジョブのリスト
        """
        additional_jobs = []
        
        if not route_plan['visit_order']:
            return additional_jobs
        
        # AMRが経路計画の各ノードを訪問する時刻を計算
        route_visit_times = {}
        current_time = amr_ready_time
        
        # 最初の搬送元ノードへの回送時間を考慮
        first_from_node = route_plan['visit_order'][0]
        if amr_current_node != first_from_node:
            # 回送時間を計算
            return_time = self.get_route_cost(amr_current_node, first_from_node)
            # 回送開始時刻を計算（前工程完了時刻を考慮）
            return_start = max(amr_ready_time, max_prev_end_time - return_time)
            if return_start < amr_ready_time:
                return_start = amr_ready_time
            return_end = return_start + return_time
            current_time = return_end
        
        # 経路計画の各ノードを訪問する時刻を計算
        for i, node in enumerate(route_plan['visit_order']):
            if i == 0:
                # 最初のノードへの到着時刻（回送後）
                route_visit_times[node] = current_time
            else:
                # 前のノードから現在のノードへの移動時間
                move_time = self.get_route_cost(route_plan['visit_order'][i-1], node)
                current_time += move_time
                route_visit_times[node] = current_time
        
        # すべての前工程マシンの出口ポートノードを取得
        all_from_nodes = set()
        for job in ordered_jobs:
            prev_machine_name = machine_assignment.get((job.name, prev_process))
            if prev_machine_name:
                prev_machine = self.machine_by_name.get(prev_machine_name)
                if prev_machine:
                    all_from_nodes.add(prev_machine.exit_port.node)
        
        # 各ノードへの最短到着時刻を計算（経路計画に基づく、中間ノードを考慮）
        node_visit_times = {}
        for from_node in all_from_nodes:
            # 経路計画に含まれるノード（中間ノードを含む）の場合
            if from_node in route_visit_times:
                node_visit_times[from_node] = route_visit_times[from_node]
            else:
                # 経路計画に含まれないノードの場合、最短経路で到着時刻を計算
                # 経路計画の各ノード（中間ノードを含む）からこのノードへの最短経路を検索
                min_arrival_time = float('inf')
                
                # AMRの現在位置から直接
                direct_time = self.get_route_cost(amr_current_node, from_node)
                if direct_time > 0:
                    arrival_time = amr_ready_time + direct_time
                    if arrival_time < min_arrival_time:
                        min_arrival_time = arrival_time
                
                # 経路計画の各ノード（中間ノードを含む）から
                # 経路計画のvisit_orderを順番に確認し、各ノードから対象ノードへの最短経路を計算
                for route_node, route_time in route_visit_times.items():
                    # 直接移動時間
                    direct_move_time = self.get_route_cost(route_node, from_node)
                    if direct_move_time > 0:
                        arrival_time = route_time + direct_move_time
                        if arrival_time < min_arrival_time:
                            min_arrival_time = arrival_time
                    
                    # 経路上の中間ノードを経由する場合も考慮
                    # 経路計画のvisit_orderで、route_nodeの後に来るノードを確認
                    route_node_index = None
                    for idx, node in enumerate(route_plan['visit_order']):
                        if node == route_node:
                            route_node_index = idx
                            break
                    
                    # route_nodeの後のノードを確認
                    if route_node_index is not None and route_node_index < len(route_plan['visit_order']) - 1:
                        # route_nodeの後のノードから対象ノードへの経路を確認
                        for next_idx in range(route_node_index + 1, len(route_plan['visit_order'])):
                            intermediate_node = route_plan['visit_order'][next_idx]
                            if intermediate_node in route_visit_times:
                                # route_nodeからintermediate_nodeへの移動時間を計算
                                # 経路計画のvisit_orderで、route_nodeからintermediate_nodeまでの経路を確認
                                segment_time = 0.0
                                for seg_idx in range(route_node_index, next_idx):
                                    seg_from = route_plan['visit_order'][seg_idx]
                                    seg_to = route_plan['visit_order'][seg_idx + 1]
                                    seg_time = self.get_route_cost(seg_from, seg_to)
                                    if seg_time > 0:
                                        segment_time += seg_time
                                    else:
                                        segment_time = None
                                        break
                                
                                if segment_time is not None and segment_time > 0:
                                    # intermediate_nodeから対象ノードへの移動時間
                                    final_move_time = self.get_route_cost(intermediate_node, from_node)
                                    if final_move_time > 0:
                                        arrival_time = route_time + segment_time + final_move_time
                                        if arrival_time < min_arrival_time:
                                            min_arrival_time = arrival_time
                
                if min_arrival_time != float('inf'):
                    node_visit_times[from_node] = min_arrival_time
        
        # 各ノードを訪問する時刻までに完了するジョブを検索
        for job in ordered_jobs:
            # 既にバッチに含まれているジョブはスキップ
            if job.name in existing_batch_job_names:
                continue
            
            # 前工程の完了時刻を確認
            prev_end_time = job_process_end_time.get(job.name, {}).get(prev_process, 0.0)
            if prev_end_time == 0.0:
                continue
            
            # 既に次工程が完了しているジョブはスキップ
            if job_process_end_time.get(job.name, {}).get(next_process, 0.0) > 0:
                continue
            
            # マシン割り当てを確認
            prev_machine_name = machine_assignment.get((job.name, prev_process))
            next_machine_name = machine_assignment.get((job.name, next_process))
            if not prev_machine_name or not next_machine_name:
                continue
            
            # 搬送元ノードと搬送先ノードを取得
            prev_machine = self.machine_by_name.get(prev_machine_name)
            next_machine = self.machine_by_name.get(next_machine_name)
            if not prev_machine or not next_machine:
                continue
            
            from_node = prev_machine.exit_port.node
            to_node = next_machine.entrance_port.node
            
            # AMRがこのノードを訪問する時刻を確認（すべてのリソースを確認）
            if from_node in node_visit_times:
                visit_time = node_visit_times[from_node]
                # 前工程が完了している、またはAMRが到着するまでに完了する場合
                # 少し余裕を持たせる（到着時刻の少し前までに完了していれば拾える）
                tolerance = 0.1  # 0.1の余裕を持たせる
                if prev_end_time <= visit_time + tolerance:
                    # process_indexを計算（工程名から）
                    process_sequence = ['A', 'B', 'C', 'D']
                    process_idx = process_sequence.index(next_process) if next_process in process_sequence else 0
                    
                    additional_jobs.append({
                        'job': job,
                        'prev_end_time': prev_end_time,
                        'from_node': from_node,
                        'to_node': to_node,
                        'prev_process': prev_process,
                        'process_name': next_process,
                        'process_index': process_idx
                    })
        
        # 容量制限を考慮して、追加できるジョブを選択
        # 前工程完了時刻が早い順にソート
        additional_jobs.sort(key=lambda x: x['prev_end_time'])
        
        # 容量制限内で追加
        available_capacity = max_capacity - len(existing_batch_job_names)
        return additional_jobs[:available_capacity]
    
    def _group_by_destination(self, batch: List[Dict]) -> Dict[str, List[Dict]]:
        """
        バッチ内のジョブを搬送先ノードでグループ化
        
        Args:
            batch: バッチ内のジョブ情報のリスト
        
        Returns:
            搬送先ノードをキーとする辞書
        """
        groups = {}
        
        for item in batch:
            to_node = item['to_node']
            
            if to_node not in groups:
                groups[to_node] = []
            
            groups[to_node].append(item)
        
        return groups
    
    def _transport_batch_between_processes(self, batch: List[Dict], prev_process: str,
                                          next_process: str, machine_assignment: Dict,
                                          amr_states: Dict, job_current_node: Dict,
                                          job_available_time: Dict, 
                                          job_process_end_time: Dict,
                                          ordered_jobs: List[Job],
                                          transport_plans: Dict = None):
        """
        複数のジョブをまとめて運ぶバッチ処理（AMR経路計画を使用）
        
        Args:
            batch: バッチ内のジョブ情報のリスト
                [{
                    'job': Job,
                    'prev_end_time': float,
                    'from_node': str,
                    'to_node': str
                }, ...]
            prev_process: 前工程名
            next_process: 次工程名
            machine_assignment: マシン割り当て
            amr_states: AMR状態
            job_current_node: ジョブの現在位置
            job_available_time: ジョブが搬送可能になる時刻
            job_process_end_time: ジョブごとの各工程完了時刻
            transport_plans: AMR経路計画
        """
        amr_name = get_amr_for_transport(prev_process, next_process)
        if amr_name is None:
            return
        
        amr = amr_states[amr_name]
        
        # バッチ内の全ジョブの前工程完了時刻の最大値を取得
        max_prev_end_time = max([item['prev_end_time'] for item in batch])
        
        # AMRの利用可能時刻を取得
        amr_ready_time = amr.available_time
        
        # 既存のバッチジョブ名のセットを作成
        existing_batch_job_names = {item['job'].name for item in batch}
        
        # 最初の経路計画を実行（常に最近傍法を使用）
        route_plan = self._plan_route_for_batch(batch, amr.current_node, 
                                                amr_ready_time, max_prev_end_time,
                                                None, prev_process, next_process)
        
        # AMRが移動中に完了するジョブを検索
        if ordered_jobs:
            additional_jobs = self._find_jobs_completing_during_route(
                route_plan, amr_ready_time, amr.current_node, max_prev_end_time,
                prev_process, next_process, ordered_jobs, machine_assignment,
                job_process_end_time, existing_batch_job_names, amr.max_capacity
            )
            
            # 追加のジョブをバッチに追加
            if additional_jobs:
                batch.extend(additional_jobs)
                existing_batch_job_names.update({item['job'].name for item in additional_jobs})
                # バッチが更新されたので、経路計画を再計算
                max_prev_end_time = max([item['prev_end_time'] for item in batch])
                route_plan = self._plan_route_for_batch(batch, amr.current_node,
                                                        amr_ready_time, max_prev_end_time,
                                                        None, prev_process, next_process)
        
        # 搬送先ノードでグループ化
        destination_groups = self._group_by_destination(batch)
        
        # 搬送先が1つの場合：まとめて運ぶ
        if len(destination_groups) == 1:
            to_node = list(destination_groups.keys())[0]
            
            # 回送時間を計算（AMRの現在位置から最初の搬送元ノードへ）
            first_from_node = route_plan['visit_order'][0]
            return_time = 0.0
            if amr.current_node != first_from_node:
                return_time = self.get_route_cost(amr.current_node, first_from_node)
                return_start = max(amr_ready_time, max_prev_end_time - return_time)
                if return_start < amr_ready_time:
                    return_start = amr_ready_time
                return_end = return_start + return_time
                amr.add_timeline(return_start, return_end, amr.current_node, 
                               first_from_node, [], 'return')
                amr_ready_time = return_end
            
            # 搬送開始時刻を決定
            transport_start = max(amr_ready_time, max_prev_end_time, route_plan['earliest_start_time'])
            
            # 搬送時間を計算（経路計画の総時間）
            transport_end = transport_start + route_plan['total_time']
            
            # タイムラインに追加
            amr.add_timeline(transport_start, transport_end, 
                          route_plan['start_node'], to_node,
                          [item['job'].name for item in batch], 'transport')
            
            # 各ジョブの状態を更新
            for item in batch:
                job_current_node[item['job'].name] = to_node
                job_available_time[item['job'].name] = transport_end
        
        else:
            # 搬送先が複数の場合：搬送先ごとにグループ化して運ぶ
            # 簡易実装：各グループを独立して運ぶ
            for to_node, group_jobs in destination_groups.items():
                # このグループの経路計画を実行（常に最近傍法を使用）
                group_route_plan = self._plan_route_for_batch(group_jobs, amr.current_node,
                                                             amr.available_time, 
                                                             max([j['prev_end_time'] for j in group_jobs]),
                                                             None, prev_process, next_process)
                
                # 回送時間を計算
                first_from_node = group_route_plan['visit_order'][0]
                return_time = 0.0
                if amr.current_node != first_from_node:
                    return_time = self.get_route_cost(amr.current_node, first_from_node)
                    return_start = max(amr.available_time, 
                                     max([j['prev_end_time'] for j in group_jobs]) - return_time)
                    if return_start < amr.available_time:
                        return_start = amr.available_time
                    return_end = return_start + return_time
                    amr.add_timeline(return_start, return_end, amr.current_node,
                                   first_from_node, [], 'return')
                    amr_ready_time = return_end
                else:
                    amr_ready_time = amr.available_time
                
                # 搬送開始時刻を決定
                group_max_prev_end = max([j['prev_end_time'] for j in group_jobs])
                transport_start = max(amr_ready_time, group_max_prev_end)
                
                # 搬送時間を計算
                transport_end = transport_start + group_route_plan['total_time']
                
                # タイムラインに追加
                amr.add_timeline(transport_start, transport_end,
                              group_route_plan['start_node'], to_node,
                              [j['job'].name for j in group_jobs], 'transport')
                
                # 各ジョブの状態を更新
                for item in group_jobs:
                    job_current_node[item['job'].name] = to_node
                    job_available_time[item['job'].name] = transport_end
    
    def create_individual(self) -> Dict:
        """ランダムな個体を生成（経路戦略は削除、常に最近傍法を使用）"""
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
        個体の適応度（makespan）を評価（AMR経路計画を使用）
        搬送時間を考慮し、makespanは最後のジョブのENDノード到着時刻
        """
        job_order = individual['job_order']
        machine_assignment = individual['machine_assignment']
        # 経路戦略は削除、常に最近傍法を使用
        
        # ジョブ順序に基づいてジョブリストを再構築
        ordered_jobs = []
        for job_id in job_order:
            for job in self.job_list:
                if job.name == job_id:
                    ordered_jobs.append(job)
                    break
        
        # スケジュールを評価（経路戦略は削除、常に最近傍法を使用）
        machine_schedules, amr_states, end_arrival_times = \
            self._evaluate_schedule_with_transport(ordered_jobs, machine_assignment, None)
        
        # AMRタイムラインを抽出
        amr_timelines = {name: state.timeline for name, state in amr_states.items()}
        
        # makespanは最後のジョブのENDノード到着時刻
        if end_arrival_times:
            makespan = max(end_arrival_times.values())
        else:
            makespan = float('inf')
        
        return makespan, machine_schedules, amr_timelines
    
    def _evaluate_schedule_with_transport(self, ordered_jobs: List[Job],
                                          machine_assignment: Dict[Tuple[str, str], str],
                                          transport_plans: Dict = None
                                          ) -> Tuple[Dict, Dict, Dict]:
        """搬送時間を考慮したスケジュール評価（AMR経路計画を使用）"""
        
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
        
        # 経路戦略は削除、常に最近傍法を使用（transport_plansは使用しない）
        
        # Phase 1: START → 工程A（AMR1が担当）
        self._transport_start_to_A(ordered_jobs, machine_assignment, amr_states,
                                   job_current_node, job_available_time)
        
        # 前工程完了時刻を事前計算（簡易版：搬送時間を考慮しない）
        # 注：正確な計算には搬送時間も必要だが、バッチ処理のため簡易版を使用
        job_process_end_time_precalc = self._calculate_all_process_end_times(
            ordered_jobs, machine_assignment, amr_states, job_available_time)
        
        # Phase 2: 各工程の処理と工程間搬送（並列処理：前工程完了したジョブから順次次工程に運ぶ）
        # A工程の処理を実行
        for job in ordered_jobs:
            assignment_key = (job.name, 'A')
            if assignment_key not in machine_assignment:
                continue
            
            assigned_machine_name = machine_assignment[assignment_key]
            assigned_machine = self.machine_by_name.get(assigned_machine_name)
            
            if assigned_machine is None:
                continue
            
            # ジョブが搬送完了してマシンの入口に到着する時刻
            transport_arrival_time = job_available_time[job.name]
            
            # マシンが利用可能になる時刻と比較
            start_time = max(machine_available_time[assigned_machine_name], transport_arrival_time)
            process_time = 0.0
            for process in job.process_list:
                if process.label == 'A':
                    process_time = process.time
                    break
            end_time = start_time + process_time
            machine_available_time[assigned_machine_name] = end_time
            
            # スケジュール情報を保存
            schedule_item = {
                'job_name': job.name,
                'process_label': 'A',
                'process_index': 0,
                'start_time': start_time,
                'end_time': end_time,
                'duration': process_time
            }
            machine_schedules[assigned_machine_name].append(schedule_item)
            
            # ジョブの状態を更新
            job_current_node[job.name] = assigned_machine.exit_port.node
            job_available_time[job.name] = end_time
            job_process_end_time[job.name]['A'] = end_time
        
        # 工程間の並列処理：前工程完了したジョブから順次次工程に運ぶ
        # 各工程の待機キューを管理
        process_queues = {
            'B': [],  # A→Bの待機キュー
            'C': [],  # B→Cの待機キュー
            'D': [],  # C→Dの待機キュー
            'END': [] # D→ENDの待機キュー
        }
        
        # 処理済みジョブを追跡（ENDノード到着済み）
        processed_jobs = set()
        
        # 全てのジョブがENDノードに到着するまで繰り返し
        while len(processed_jobs) < len(ordered_jobs):
            # 各工程の待機キューを更新
            process_sequence = ['A', 'B', 'C', 'D']
            for process_idx, process_name in enumerate(process_sequence):
                if process_name == 'A':
                    continue
                
                prev_process = process_sequence[process_idx - 1]
                
                # 前工程完了したジョブを待機キューに追加
                for job in ordered_jobs:
                    if job.name in processed_jobs:
                        continue
                    
                    assignment_key = (job.name, process_name)
                    if assignment_key not in machine_assignment:
                        continue
                    
                    # 前工程の完了時刻を確認
                    prev_end_time = job_process_end_time.get(job.name, {}).get(prev_process, 0.0)
                    if prev_end_time == 0.0:
                        continue
                    
                    # 既にこの工程が完了しているか確認
                    if job_process_end_time.get(job.name, {}).get(process_name, 0.0) > 0:
                        continue
                    
                    # 既に待機キューに追加済みか確認
                    already_queued = any(item['job'].name == job.name for item in process_queues[process_name])
                    if already_queued:
                        continue
                    
                    # 搬送元：前工程マシンの出口ポート
                    prev_machine_name = machine_assignment.get((job.name, prev_process))
                    if prev_machine_name:
                        prev_machine = self.machine_by_name.get(prev_machine_name)
                        if prev_machine:
                            from_node = prev_machine.exit_port.node
                            
                            # 搬送先：次工程マシンの入口ポート
                            next_machine_name = machine_assignment.get((job.name, process_name))
                            if next_machine_name:
                                next_machine = self.machine_by_name.get(next_machine_name)
                                if next_machine:
                                    to_node = next_machine.entrance_port.node
                                    
                                    process_queues[process_name].append({
                                        'job': job,
                                        'prev_end_time': prev_end_time,
                                        'from_node': from_node,
                                        'to_node': to_node,
                                        'prev_process': prev_process,
                                        'process_name': process_name,
                                        'process_index': process_idx
                                    })
            
            # D工程完了したジョブをEND待機キューに追加
            for job in ordered_jobs:
                if job.name in processed_jobs:
                    continue
                
                # D工程の完了時刻を確認
                d_end_time = job_process_end_time.get(job.name, {}).get('D', 0.0)
                if d_end_time == 0.0:
                    continue
                
                # 既にENDノードに到着しているか確認
                if job.name in end_arrival_times:
                    continue
                
                # 既に待機キューに追加済みか確認
                already_queued = any(item['job'].name == job.name for item in process_queues['END'])
                if already_queued:
                    continue
                
                # 搬送元：D工程マシンの出口ポート
                d_machine_name = machine_assignment.get((job.name, 'D'))
                if d_machine_name:
                    d_machine = self.machine_by_name.get(d_machine_name)
                    if d_machine:
                        from_node = d_machine.exit_port.node
                        to_node = '7'  # ENDノード
                        
                        process_queues['END'].append({
                            'job': job,
                            'prev_end_time': d_end_time,
                            'from_node': from_node,
                            'to_node': to_node,
                            'prev_process': 'D',
                            'process_name': 'END'
                        })
            
            # 各工程のバッチ処理を実行（並列処理）
            for process_name in ['B', 'C', 'D', 'END']:
                if process_name == 'END':
                    prev_process = 'D'
                else:
                    prev_process = process_sequence[process_sequence.index(process_name) - 1]
                waiting_queue = process_queues[process_name]
                
                if not waiting_queue:
                    continue
                
                amr_name = get_amr_for_transport(prev_process, process_name)
                if not amr_name:
                    continue
                
                amr = amr_states[amr_name]
                max_capacity = amr.max_capacity
                
                # バッチ処理を実行（待機キューが空になるまで繰り返す）
                while waiting_queue:
                    # 待機キューを距離ベースでソート（AMRの現在位置から遠い順）
                    # 距離が同じ場合は前工程完了時刻が早い順（二次ソート）
                    # 注意：AMRの現在位置は前のバッチ処理後に更新されるため、毎回再ソート
                    amr_current_node = amr.current_node
                    waiting_queue.sort(
                        key=lambda x: (
                            -self.get_route_cost(amr_current_node, x['from_node']),  # 距離（遠い順：負の値で降順）
                            x['prev_end_time']  # 前工程完了時刻（早い順）
                        )
                    )
                    
                    # バッチを構築（距離ベース：遠いノードを優先、前工程完了時刻も考慮）
                    batch = []
                    
                    # 最初のジョブは必ず追加（待機キューは距離が遠い順にソート済み）
                    if waiting_queue:
                        first_job = waiting_queue[0]
                        # 前工程完了時刻を確認
                        if first_job['prev_end_time'] <= amr.available_time or len(batch) == 0:
                            first_job = waiting_queue.pop(0)
                            batch.append(first_job)
                            current_time = first_job['prev_end_time']
                            
                            # 残りのジョブを追加（距離が遠い順にソート済み、前工程完了時刻も考慮）
                            while len(batch) < max_capacity and waiting_queue:
                                next_job = waiting_queue[0]
                                # 前工程完了時刻が現在時刻以下、または許容範囲内の場合
                                time_diff = abs(next_job['prev_end_time'] - current_time)
                                if time_diff <= 10.0 or len(batch) < 2:
                                    batch.append(waiting_queue.pop(0))
                                    current_time = max(current_time, next_job['prev_end_time'])
                                else:
                                    # 前工程完了を待つ必要がある
                                    break
                    
                    if not batch:
                        # バッチが構築できない場合（前工程完了を待つ必要がある）、ループを終了
                        break
                    if process_name == 'END':
                        # D→ENDの搬送を実行（真のバッチ処理、常に最近傍法を使用）
                        # _transport_batch_between_processesを使用して複数のジョブをまとめて運ぶ
                        self._transport_batch_between_processes(
                            batch, 'D', 'END', machine_assignment,
                            amr_states, job_current_node, job_available_time, job_process_end_time,
                            ordered_jobs, None
                        )
                        
                        # ENDノード到着時刻を記録
                        processed_job_names = []
                        for item in batch:
                            job = item['job']
                            # job_available_timeにENDノード到着時刻が記録されている
                            end_arrival_times[job.name] = job_available_time[job.name]
                            processed_job_names.append(job.name)
                            processed_jobs.add(job.name)
                        
                        # バッチ処理で処理したジョブを待機キューから一括削除
                        process_queues[process_name] = [item for item in process_queues[process_name] 
                                                         if item['job'].name not in processed_job_names]
                        
                        # 待機キューの参照を更新（次のバッチ処理のために）
                        waiting_queue = process_queues[process_name]
                    else:
                        # B, C, D工程のバッチ搬送を実行（常に最近傍法を使用）
                        self._transport_batch_between_processes(
                            batch, prev_process, process_name, machine_assignment,
                            amr_states, job_current_node, job_available_time, job_process_end_time,
                            ordered_jobs, None
                        )
                        
                        # 処理したジョブの名前を記録
                        processed_job_names = []
                        
                        # 各ジョブの処理を実行
                        for item in batch:
                            job = item['job']
                            process_idx = item['process_index']
                            assigned_machine_name = machine_assignment.get((job.name, process_name))
                            if assigned_machine_name:
                                assigned_machine = self.machine_by_name.get(assigned_machine_name)
                                if assigned_machine:
                                    # ジョブが搬送完了してマシンの入口に到着する時刻
                                    transport_arrival_time = job_available_time[job.name]
                                    
                                    # マシンが利用可能になる時刻と比較
                                    start_time = max(machine_available_time[assigned_machine_name], transport_arrival_time)
                                    process_time = 0.0
                                    for process in job.process_list:
                                        if process.label == process_name:
                                            process_time = process.time
                                            break
                                    end_time = start_time + process_time
                                    machine_available_time[assigned_machine_name] = end_time
                                    
                                    # スケジュール情報を保存
                                    schedule_item = {
                                        'job_name': job.name,
                                        'process_label': process_name,
                                        'process_index': process_idx,
                                        'start_time': start_time,
                                        'end_time': end_time,
                                        'duration': process_time
                                    }
                                    machine_schedules[assigned_machine_name].append(schedule_item)
                                    
                                    # ジョブの状態を更新
                                    job_current_node[job.name] = assigned_machine.exit_port.node
                                    job_available_time[job.name] = end_time
                                    job_process_end_time[job.name][process_name] = end_time
                                    
                                    # 処理したジョブの名前を記録
                                    processed_job_names.append(job.name)
                        
                        # バッチ処理で処理したジョブを待機キューから一括削除
                        process_queues[process_name] = [item for item in process_queues[process_name] 
                                                         if item['job'].name not in processed_job_names]
                        
                        # 待機キューの参照を更新（次のバッチ処理のために）
                        waiting_queue = process_queues[process_name]
            
            # 残りのジョブを個別に処理（バッチ処理できなかったジョブ）
            for process_name in ['B', 'C', 'D', 'END']:
                if process_name == 'END':
                    prev_process = 'D'
                else:
                    prev_process = process_sequence[process_sequence.index(process_name) - 1]
                
                waiting_queue = process_queues[process_name]
                
                if not waiting_queue:
                    continue
                
                # 待機キューから個別処理可能なジョブを処理
                processed_in_this_iteration = []
                for item in waiting_queue:
                    job = item['job']
                    if job.name in processed_jobs:
                        processed_in_this_iteration.append(item)
                        continue
                    
                    # 前工程完了時刻を確認
                    prev_end_time = item['prev_end_time']
                    if prev_end_time == 0.0:
                        continue
                    
                    amr_name = get_amr_for_transport(prev_process, process_name)
                    if not amr_name:
                        continue
                    
                    amr = amr_states[amr_name]
                    
                    # AMRが利用可能で、前工程完了時刻が経過している場合
                    if prev_end_time <= amr.available_time or len(waiting_queue) == 1:
                        if process_name == 'END':
                            # D→ENDの搬送（個別処理）
                            from_node = item['from_node']
                            to_node = item['to_node']
                            
                            # AMRが利用可能になる時刻
                            amr_ready_time = amr.available_time
                            
                            # AMRの現在位置から搬送元への回送時間
                            return_time = 0.0
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
                            
                            # 処理済みのジョブを待機キューから削除
                            process_queues[process_name] = [item for item in process_queues[process_name] 
                                                             if item['job'].name != job.name]
                            
                            processed_jobs.add(job.name)
                            processed_in_this_iteration.append(item)
                        else:
                            # 工程間搬送（個別処理）
                            self._transport_between_processes(
                                job, prev_process, process_name, machine_assignment,
                                amr_states, job_current_node, job_available_time, job_process_end_time
                            )
                            
                            assigned_machine_name = machine_assignment.get((job.name, process_name))
                            if assigned_machine_name:
                                assigned_machine = self.machine_by_name.get(assigned_machine_name)
                                if assigned_machine:
                                    # ジョブが搬送完了してマシンの入口に到着する時刻
                                    transport_arrival_time = job_available_time[job.name]
                                    
                                    # マシンが利用可能になる時刻と比較
                                    start_time = max(machine_available_time[assigned_machine_name], transport_arrival_time)
                                    process_time = 0.0
                                    for process in job.process_list:
                                        if process.label == process_name:
                                            process_time = process.time
                                            break
                                    end_time = start_time + process_time
                                    machine_available_time[assigned_machine_name] = end_time
                                    
                                    # スケジュール情報を保存
                                    schedule_item = {
                                        'job_name': job.name,
                                        'process_label': process_name,
                                        'process_index': item['process_index'],
                                        'start_time': start_time,
                                        'end_time': end_time,
                                        'duration': process_time
                                    }
                                    machine_schedules[assigned_machine_name].append(schedule_item)
                                    
                                    # ジョブの状態を更新
                                    job_current_node[job.name] = assigned_machine.exit_port.node
                                    job_available_time[job.name] = end_time
                                    job_process_end_time[job.name][process_name] = end_time
                                    
                                    # 処理済みのジョブを待機キューから削除
                                    process_queues[process_name] = [item for item in process_queues[process_name] 
                                                                     if item['job'].name != job.name]
                                    
                                    processed_in_this_iteration.append(item)
                
                # 処理済みのジョブを待機キューから削除
                for item in processed_in_this_iteration:
                    if item in waiting_queue:
                        waiting_queue.remove(item)
        
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
        
        child1 = {
            'job_order': child1_job_order,
            'machine_assignment': child1_machine_assignment
        }
        child2 = {
            'job_order': child2_job_order,
            'machine_assignment': child2_machine_assignment
        }
        
        return child1, child2
    
    def mutation(self, individual: Dict) -> Dict:
        """突然変異（経路戦略は削除、常に最近傍法を使用）"""
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
        print("最適化対象: ジョブ順序 + マシン割り当て（経路計画は常に最近傍法を使用）")
        
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
    # ファイルが存在する場合は削除してから保存
    if os.path.exists(filename):
        try:
            # ファイルが開かれている可能性があるため、少し待機してから削除を試みる
            time.sleep(0.1)
            os.remove(filename)
            time.sleep(0.1)  # 削除後に少し待機
        except Exception as e:
            # 削除に失敗した場合、エラーメッセージを表示（ただし、ファイルが開かれている場合は無視）
            print(f"警告: ガントチャートファイルの削除に失敗しました: {e}")
            # ファイルが開かれている場合は、上書き保存で対応
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"ガントチャートを {filename} に保存しました。")


def create_evolution_chart(fitness_history: List[float], filename: str = "evolution_chart_transport.jpeg"):
    """
    進化の履歴（世代ごとの最良適応度）をグラフ化する
    
    Args:
        fitness_history: 各世代の最良適応度のリスト
        filename: 出力ファイル名
    """
    # 日本語フォントを設定
    plt.rcParams['font.sans-serif'] = ['MS Gothic', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 図と軸を作成
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 世代番号
    generations = list(range(1, len(fitness_history) + 1))
    
    # グラフを描画
    ax.plot(generations, fitness_history, linewidth=2, color='#2E86AB', marker='o', markersize=3, label='最良適応度')
    
    # 最小値を強調表示
    min_fitness = min(fitness_history)
    min_generation = fitness_history.index(min_fitness) + 1
    ax.scatter([min_generation], [min_fitness], color='red', s=100, zorder=5, 
               label=f'最良解 (世代{min_generation}: {min_fitness:.2f})')
    
    # 軸の設定
    ax.set_xlabel('世代', fontsize=12, fontweight='bold')
    ax.set_ylabel('適応度 (Makespan)', fontsize=12, fontweight='bold')
    ax.set_title('遺伝的アルゴリズムの進化過程（搬送時間考慮）', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10)
    
    # 初期値と最終値を表示
    initial_fitness = fitness_history[0]
    final_fitness = fitness_history[-1]
    improvement = initial_fitness - final_fitness
    improvement_rate = (improvement / initial_fitness) * 100 if initial_fitness > 0 else 0
    
    # テキストボックスに統計情報を追加
    stats_text = f'初期適応度: {initial_fitness:.2f}\n'
    stats_text += f'最終適応度: {final_fitness:.2f}\n'
    stats_text += f'改善量: {improvement:.2f} ({improvement_rate:.1f}%)'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # レイアウトを調整
    plt.tight_layout()
    
    # ファイルが存在する場合は削除してから保存
    if os.path.exists(filename):
        try:
            time.sleep(0.1)
            os.remove(filename)
            time.sleep(0.1)
        except Exception as e:
            print(f"警告: 進化グラフファイルの削除に失敗しました: {e}")
    
    # 図をJPEGファイルとして保存
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"進化グラフを {filename} に保存しました。")
    
    # 図を閉じる（メモリ節約のため表示しない）
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
        if self.file:
            self.file.flush()
            self.file.close()


def main():
    """メイン関数"""
    # 実行のたびに異なる結果が得られるようにランダムシードを設定
    random.seed(int(time.time() * 1000000) % (2**32))
    
    log_file = "execution_log_transport.txt"
    # 既存のログファイルを削除してから新しいログを開始
    if os.path.exists(log_file):
        try:
            # ファイルが開かれている可能性があるため、少し待機してから削除を試みる
            time.sleep(0.1)
            os.remove(log_file)
            time.sleep(0.1)  # 削除後に少し待機
        except Exception as e:
            # 削除に失敗した場合、エラーメッセージを表示（ただし、ファイルが開かれている場合は無視）
            print(f"警告: ログファイルの削除に失敗しました: {e}", file=sys.__stdout__)
            # ファイルが開かれている場合は、上書きモードで開くことで対応
    
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print("=" * 60)
        print("搬送あり・衝突無視ジョブスケジューリングシステム実行開始")
        print(f"実行時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"ランダムシード: {random.getstate()[1][0]}")
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
        POPULATION_SIZE = 100
        CROSSOVER_RATE = 0.8
        MUTATION_RATE = 0.1
        MAX_GENERATIONS = 300
        
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
        
        # 進化のグラフを作成
        evolution_file = "evolution_chart_transport.jpeg"
        print(f"\n進化グラフを {evolution_file} に出力中...")
        create_evolution_chart(ga.fitness_history, evolution_file)
        
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

