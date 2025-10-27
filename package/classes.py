import copy
from dataclasses import dataclass
from abc import ABC
from enum import Enum
from collections import deque


class Node:
    """
    ノード
    """
    # ノード名
    label: str
    # 接続されてるノード情報
    linked_nodes: dict[str, float]

    def __init__(self, label: str):
        self.label = label
        self.linked_nodes = {}

    def __str__(self):
        return self.label


class Edge:
    """
    枝
    """
    # 両端のノード
    nodes: list[str]
    # 枝にかかるコスト(時間)
    cost: float

    def __init__(self, nodes: list[str], cost: float):
        self.nodes = nodes
        self.cost = cost

    def __str__(self):
        return "([" + self.nodes[0] + ", " + self.nodes[1] + ']' + str(self.cost) + ')'


class Route:
    # ルートにかかるコスト(時間)
    cost: float
    #
    sequence: list[str]
    # 両端のノード
    tip_node: list[str]

    def __init__(self, sequence: list[str], cost: float):
        self.sequence = sequence
        self.cost = cost
        self.tip_node = [self.sequence[0], self.sequence[len(self.sequence) - 1]]

    def split_to_one_edge(self) -> list[list[str]]:
        result_list: list[list[str]] = []
        for i in range(len(self.sequence) - 1):
            result_list.append([self.sequence[i], self.sequence[i + 1]])
        return result_list

    def __str__(self):
        text: str = str(self.cost) + ','
        for i in range(len(self.sequence) - 1):
            text += (self.sequence[i] + ',')
        text += self.sequence[len(self.sequence) - 1]
        return text


class Process:
    label: str
    time: float
    start_time: float
    end_time: float

    def __init__(self, label: str, time: float = 0, start_time: float = 0):
        self.label = label
        self.time = time
        self.start_time = start_time
        self.end_time = self.start_time + self.time

    def set_start_time(self, update_time: float):
        self.start_time = update_time
        self.end_time = update_time + self.time

    def __str__(self):
        return "<'" + self.label + "'," + str(self.start_time) + "-(" + str(self.time) + ")-" + str(self.end_time) + ">"


class Job:
    name: str
    due_date: float
    process_list: list[Process]  # [Process('A', 10), Process('B', 20), ..]
    _current_process: int

    def __init__(self, name: str, due_data: float, process_list: list[Process]):
        self.name = name
        self.due_date = due_data
        self.process_list = process_list
        self._current_process = 0

    def execute_current_process(self, start_time: float):
        self.process_list[self._current_process].set_start_time(start_time)
        self._current_process += 1

    def is_remaining_process(self) -> bool:
        """
        処理しないといけない工程が残っているかどうか
        :return:
        """
        return self._current_process < len(self.process_list)

    def get_current_process(self) -> Process:
        """
        今行わないといけない工程の情報を返す関数
        :return:
        """
        return self.process_list[self._current_process]

    def has_before_process_finished(self, current_time: float) -> bool:
        """
        前の工程が終了しているか判定する関数．終了してたらTrue,終了して無かったらFalse
        :param current_time: 現在の時刻
        :return: bool
        """
        if self._current_process <= 0:
            return True
        if current_time < self.process_list[self._current_process - 1].end_time:
            return False
        else:
            return True

    def __str__(self):
        text: str = "['" + self.name + "', " + str(self.due_date) + ", "
        for process in self.process_list:
            text += str(process)
        text += "]"
        return text


class Port:
    node: str
    carrier: deque[Job]
    accessible_process: str

    def __init__(self, node: str, accessible_process: str = ''):
        self.node = node
        self.accessible_process = accessible_process
        self.carrier = deque()

    def get_job_num_of_inside(self):
        return len(self.carrier)

    def __str__(self):
        return self.node


class MachineEntPort(Port):
    _job_num_in_workspace: int

    def __init__(self, node: str, accessible_proces: str = ''):
        super().__init__(node, accessible_proces)
        self._job_num_in_workspace = 0

    def update_job_num_in_workspace(self, job_num: int):
        self._job_num_in_workspace = job_num

    """Override"""
    def get_job_num_of_inside(self):
        return len(self.carrier) + self._job_num_in_workspace


class Machine:
    name: str
    target_process: str
    entrance_port: MachineEntPort
    exit_port: Port
    workspace: list[Job]
    timeline: list[list[float]]  # [[0, 10],[30, 40],[50, 60]]
    timeline_label: list[str]    # ['Job1', 'Job2',  'Job3']

    def __init__(self, name: str, target_process: str, ent_node: str, exit_node: str):
        self.name = name
        self.target_process = target_process
        self.entrance_port = MachineEntPort(ent_node, target_process)
        self.exit_port = Port(exit_node)
        self.workspace = []
        self.timeline = []
        self.timeline_label = []

    def release_job(self):
        self.exit_port.carrier.append(self.workspace.pop(0))
        self.entrance_port.update_job_num_in_workspace(len(self.workspace))

    def have_stock_of_job(self) -> bool:
        if len(self.entrance_port.carrier) > 0:
            return True
        else:
            return False

    def start_processing_next_job(self, current_time: float):
        job: Job = self.entrance_port.carrier.popleft()
        process: Process = job.get_current_process()
        self.workspace.append(job)
        self.entrance_port.update_job_num_in_workspace(len(self.workspace))
        self.timeline.append([current_time, current_time + process.time])
        self.timeline_label.append(job.name)
        job.execute_current_process(current_time)

    def is_using(self, current_time: float) -> bool:
        for tl in self.timeline:
            if tl[0] <= current_time < tl[1]:
                return True
        return False

    def get_last_processing_end_time(self) -> float:
        last_history_num: int = len(self.timeline) - 1
        if last_history_num < 0:
            return -1.0
        else:
            return self.timeline[last_history_num][1]

    def get_timeline_info(self) -> str:
        t: str = "<'" + self.name + "'>"
        for i in range(len(self.timeline)):
            t += "[" + str(self.timeline[i][0]) + ", " + self.timeline_label[i] + ", " + str(self.timeline[i][1]) + "]"
        return t

    def __str__(self):
        return "name: " + self.name + ", target: '" + self.target_process + "'"


class Amr:
    # AMR名
    name: str
    # AMRが運ぶことを対象とするプロセス名
    target_process: list[str]
    # AMRの荷台
    luggage_carrier: list[Job]
    # 一度に運べるJob数
    max_capacity: int
    # タイムライン
    timeline: list[list[float]]
    timeline_label: list[str]
    # 現在地
    current_node: str

    def __init__(self, name: str, max_capacity: int, target_process: list[str], current_node: str = ''):
        self.name = name
        self.current_node = current_node
        self.target_process = target_process
        self.max_capacity = max_capacity
        self.luggage_carrier = []
        self.timeline = []
        self.timeline_label = []

    def can_carry_job(self) -> bool:
        if len(self.luggage_carrier) < self.max_capacity:
            return True
        else:
            return False

    def is_moving(self, current_time: float) -> bool:
        for tl in self.timeline:
            if tl[0] <= current_time < tl[1]:
                return True
        return False

    def get_last_moving_end_time(self) -> float:
        last_history_num: int = len(self.timeline) - 1
        if last_history_num < 0:
            return -1.0
        else:
            return self.timeline[last_history_num][1]

    def set_moving_command(self, current_time: float, nearest_port: str, cost: float):
        self.timeline.append([current_time, current_time + cost])
        self.timeline_label.append(self.current_node + "->" + nearest_port)
        self.current_node = nearest_port

    def get_timeline_info(self) -> str:
        t: str = "<'" + self.name + "'>"
        for i in range(len(self.timeline)):
            t += "[" + str(self.timeline[i][0]) + ", " + self.timeline_label[i] + ", " + str(self.timeline[i][1]) + "]"
        return t

    def get_timeline_detail_info(self, route_dict: dict[str, Route]) -> str:
        t: str = "<'" + self.name + "'>"
        for i in range(len(self.timeline)):
            t += "[" + str(self.timeline[i][0]) + ", "
            route: Route = route_dict[self.timeline_label[i]]
            if len(route.sequence) == 1:
                t += self.timeline_label[i]
            else:
                for j in range(len(route.sequence)):
                    t += route.sequence[j]
                    if j < len(route.sequence) - 1:
                        t += "->"
            t += ", " + str(self.timeline[i][1]) + "]"
        return t
