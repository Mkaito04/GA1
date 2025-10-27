"""
scheduler_basic.py - マシンに基づく簡易スケジューリングとガントチャート可視化

このスクリプトは、AMR（搬送）を考慮せず、マシンのみでジョブスケジュールを組み、
ガントチャートで表示し、実行結果をテキストファイルに出力します。
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
from datetime import datetime
from package.data_manager import load_job_data, load_machine_data, convert_machine_list_into_dict
from package.classes import Job, Machine, Process
from typing import List, Dict


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


def create_gantt_chart(machine_schedules: Dict[str, List], machine_list: List[Machine]):
    """
    ガントチャートを作成して表示する
    
    Args:
        machine_schedules: マシンごとのスケジュール情報
        machine_list: マシンのリスト（表示順序を決定するため）
    """
    # 図と軸を作成
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 各ジョブに色を割り当て（同じジョブは同じ色）
    job_colors = {}
    colors = plt.cm.get_cmap('tab20', 20)
    
    # マシンリストの順序を維持して表示
    machine_order = [machine.name for machine in machine_list]
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
            process_label = schedule['process_label']
            process_idx = schedule['process_index']
            
            # ジョブごとに色を割り当て
            if job_name not in job_colors:
                job_colors[job_name] = colors(len(job_colors))
            
            # バーを作成
            duration = end - start
            bar_label = f"{job_name}-{process_label}"
            
            # ガントチャートバーを描画
            ax.barh(y_pos, duration, left=start, height=0.7, 
                   color=job_colors[job_name], edgecolor='black', linewidth=0.5)
            
            # バーの中央にラベルを追加
            if duration > 2:  # 短すぎるバーにはラベルを付けない
                ax.text(start + duration / 2, y_pos, bar_label,
                       ha='center', va='center', fontsize=8, fontweight='bold')
            
            max_time = max(max_time, end)
    
    # 軸の設定
    ax.set_yticks(list(range(len(machine_order))))
    ax.set_yticklabels(machine_order)
    ax.set_xlabel('時間', fontsize=12, fontweight='bold')
    ax.set_ylabel('マシン', fontsize=12, fontweight='bold')
    ax.set_title('ジョブスケジュール - ガントチャート', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max_time * 1.05)
    ax.grid(True, axis='x', alpha=0.3)
    
    # 凡例を作成
    legend_elements = []
    for job_name, color in job_colors.items():
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black', label=job_name))
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), title='ジョブ')
    
    # レイアウトを調整
    plt.tight_layout()
    
    # 図を表示
    plt.show()


def create_text_output(machine_schedules: Dict[str, List], machine_list: List[Machine], output_file: str = "schedule_result.txt"):
    """
    スケジュール結果をテキストファイルに出力する
    
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
                f.write(f"  {schedule['job_name']}-{schedule['process_label']}: "
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
            print(f"  {schedule['job_name']}-{schedule['process_label']}: "
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
        
        print(f"読み込まれたジョブ数: {len(job_list)}")
        print(f"読み込まれたマシン数: {len(machine_list)}")
        
        # マシンを工程ごとにグループ化
        machine_dict = convert_machine_list_into_dict(machine_list)
        
        print("\nマシンの工程別グループ:")
        for process_name, machines in machine_dict.items():
            print(f"  {process_name}: {[m.name for m in machines]}")
        
        # スケジュールを作成
        print("\nスケジュール作成中...")
        machine_schedules = schedule_jobs(job_list, machine_dict)
        
        # スケジュールの概要を表示
        print_schedule_summary(machine_schedules)
        
        # テキストファイルに詳細結果を出力
        print("\nテキストファイル出力中...")
        create_text_output(machine_schedules, machine_list)
        
        # ガントチャートを作成
        print("\nガントチャート生成中...")
        create_gantt_chart(machine_schedules, machine_list)
        
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

