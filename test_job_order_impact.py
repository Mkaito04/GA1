"""
ジョブ順序がmakespanに与える影響を検証するスクリプト
"""
import random
from package.data_manager import load_job_data, load_machine_data, convert_machine_list_into_dict
from scheduler_basic import GeneticAlgorithm

def test_different_job_orders():
    """異なるジョブ順序でmakespanがどう変わるかをテスト"""
    
    # データを読み込み
    job_list = load_job_data('data/case01_job.csv')
    machine_list = load_machine_data('data/case01_machine.csv')
    machine_dict = convert_machine_list_into_dict(machine_list)
    
    # GAインスタンスを作成（評価関数だけ使う）
    ga = GeneticAlgorithm(
        job_list=job_list,
        machine_dict=machine_dict,
        population_size=10,
        max_generations=1
    )
    
    # 異なるジョブ順序をテスト
    job_ids = [job.name for job in job_list]
    
    print("=" * 60)
    print("ジョブ順序がmakespanに与える影響の検証")
    print("=" * 60)
    print(f"ジョブ数: {len(job_list)}")
    print(f"マシン数: {len(machine_list)}")
    print()
    
    # テスト1: 元の順序
    original_order = job_ids.copy()
    fitness1 = ga.evaluate_fitness(original_order)
    print(f"テスト1 - 元の順序: {original_order}")
    print(f"  Makespan: {fitness1:.2f}")
    print()
    
    # テスト2: 逆順
    reversed_order = job_ids[::-1]
    fitness2 = ga.evaluate_fitness(reversed_order)
    print(f"テスト2 - 逆順: {reversed_order}")
    print(f"  Makespan: {fitness2:.2f}")
    print()
    
    # テスト3-10: ランダムな順序（5パターン）
    random_orders = []
    for i in range(5):
        random_order = job_ids.copy()
        random.shuffle(random_order)
        random_orders.append(random_order)
        fitness = ga.evaluate_fitness(random_order)
        print(f"テスト{3+i} - ランダム順序{i+1}: {random_order}")
        print(f"  Makespan: {fitness:.2f}")
        print()
    
    # 結果の統計
    all_fitnesses = [fitness1, fitness2] + [ga.evaluate_fitness(order) for order in random_orders]
    
    print("=" * 60)
    print("結果の統計")
    print("=" * 60)
    print(f"最小makespan: {min(all_fitnesses):.2f}")
    print(f"最大makespan: {max(all_fitnesses):.2f}")
    print(f"平均makespan: {sum(all_fitnesses) / len(all_fitnesses):.2f}")
    print(f"すべて同じ値か: {len(set(all_fitnesses)) == 1}")
    
    if len(set(all_fitnesses)) == 1:
        print("\n⚠️  警告: すべてのジョブ順序で同じmakespanになりました。")
        print("   ジョブ順序の最適化が効果を発揮していない可能性があります。")
    else:
        print(f"\n✓ ジョブ順序によってmakespanが変動します（差: {max(all_fitnesses) - min(all_fitnesses):.2f}）")

if __name__ == "__main__":
    test_different_job_orders()

