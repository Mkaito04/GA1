import copy
import csv

from package.classes import Route, Process, Job, Port, Machine, Amr


def load_job_data(file_path: str) -> list[Job]:
    """
    Jobデータを読み込んでlist[Job]に変換してくれる関数
    :param file_path: str
    :return: list[Job]
    """
    job_list: list[Job] = []
    with open('./' + file_path, mode='r', encoding='utf-8') as file:
        inp_str_list: list[list[str]] = [tmp for tmp in csv.reader(file)]
        for split_inp_str in inp_str_list:
            job_list.append(Job(
                split_inp_str[1],
                float(split_inp_str[2]),
                [Process(split_inp_str[i], float(split_inp_str[i + 1])) for i in range(3, len(split_inp_str), 2)]
            ))
    return job_list


def load_machine_data(file_path: str) -> list[Machine]:
    """
    Machineデータを読み込んでlis[Machine]に変換してくれる関数
    :param file_path:
    :return:
    """
    machine_list: list[Machine]
    with open('./' + file_path, mode='r', encoding='utf-8') as file:
        inp_str_list: list[list[str]] = [tmp for tmp in csv.reader(file)]
        machine_list = [
            Machine(inp_str_list[i][1], inp_str_list[i][2], inp_str_list[i][3], inp_str_list[i][4])
            for i in range(0, len(inp_str_list))
        ]
    return machine_list


def load_amr_data(file_path: str) -> list[Amr]:
    amr_list: list[Amr] = []
    with open('./' + file_path, mode='r', encoding='utf-8') as file:
        inp_str_list: list[list[str]] = [tmp for tmp in csv.reader(file)]
        for split_str in inp_str_list:
            amr_list.append(Amr(split_str[1], int(split_str[2]), [split_str[i] for i in range(3, len(split_str))]))

    return amr_list


def load_route_data(file_path: str) -> list[Route]:
    route_list: list[Route] = []
    with open('./' + file_path, mode='r', encoding='utf-8') as file:
        inp_str_list: list[list[str]] = [tmp for tmp in csv.reader(file)]
        for split_str in inp_str_list:
            route_list.append(
                Route(
                    [split_str[i] for i in range(2, len(split_str))],
                    float(split_str[1]))
            )
    return route_list


def convert_machine_list_into_dict(machine_list: list[Machine]) -> dict[str, list[Machine]]:
    """
    Machineリストを処理できる工程ごとに分けた辞書型に変換する関数
    :param machine_list: list[Machine]
    :return: list[Machine] ex:{'A':[Machine('A1'),Machine('A2'),Machine('A3')], 'B':[Machine('B1'),Machine('B2')]}
    """
    target_process_list: list[str] = []
    machine_dict: dict[str, list[Machine]] = {}
    for machine in machine_list:
        target_process: str = machine.target_process
        if target_process in target_process_list:
            machine_dict[target_process].append(machine)
        else:
            machine_dict[target_process] = [machine]
            target_process_list.append(target_process)

    return machine_dict


def convert_entrance_port_into_dict(machine_list: list[Machine]) -> dict[str, list[Port]]:
    target_process_list: list[str] = []
    port_dict: dict[str, list[Port]] = {}
    for machine in machine_list:
        target_process: str = machine.target_process
        if target_process in target_process_list:
            port_dict[target_process].append(machine.entrance_port)
        else:
            port_dict[target_process] = [machine.entrance_port]
            target_process_list.append(target_process)

    return port_dict


def convert_route_list_into_dict(route_list: list[Route]) -> dict[str, Route]:
    routes: list[Route] = copy.deepcopy(route_list)
    route_dict: dict[str, Route] = {}
    for route in routes:
        route_dict[route.tip_node[0] + '->' + route.tip_node[1]] = copy.deepcopy(route)
        route.sequence.reverse()
        route_dict[route.tip_node[1] + '->' + route.tip_node[0]] = copy.deepcopy(route)

    return route_dict


def load_edge_data(file_path: str) -> dict[str, dict[str, float]]:
    """
    エッジデータを読み込んでグラフ構造（隣接リスト）に変換する関数
    :param file_path: str
    :return: dict[str, dict[str, float]] - {from_node: {to_node: cost}}
    """
    graph: dict[str, dict[str, float]] = {}
    with open('./' + file_path, mode='r', encoding='utf-8') as file:
        inp_str_list: list[list[str]] = [tmp for tmp in csv.reader(file)]
        for split_str in inp_str_list:
            if len(split_str) >= 4 and split_str[0] == '<EDGE>':
                from_node = split_str[1]
                to_node = split_str[2]
                cost = float(split_str[3])
                
                if from_node not in graph:
                    graph[from_node] = {}
                graph[from_node][to_node] = cost
                
                # 双方向のエッジとして扱う（必要に応じて）
                if to_node not in graph:
                    graph[to_node] = {}
                graph[to_node][from_node] = cost
    
    return graph