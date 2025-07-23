import numpy as np
import time
import matplotlib.pyplot as plt
from pyswarms.single.global_best import GlobalBestPSO

# -----------------------------
# Task Configuration
NUM_TASKS = 15
NUM_VMS = 3
task_times = np.random.randint(100, 500, size=NUM_TASKS)

# -----------------------------
# Hybrid GA-RM
from hybrid_ga_rm import run_ga_rm  # Must return (allocation, makespan)

# -----------------------------
# Min-Min Scheduler
def min_min_scheduler(task_times, num_vms):
    task_list = list(range(len(task_times)))
    vm_loads = [0] * num_vms
    allocation = [-1] * len(task_times)
    while task_list:
        min_completion = float('inf')
        best_task, best_vm = None, None
        for task in task_list:
            for vm in range(num_vms):
                completion = vm_loads[vm] + task_times[task]
                if completion < min_completion:
                    min_completion = completion
                    best_task, best_vm = task, vm
        allocation[best_task] = best_vm
        vm_loads[best_vm] += task_times[best_task]
        task_list.remove(best_task)
    return allocation, max(vm_loads)

# -----------------------------
# PSO Scheduler
def evaluate_particle(x, task_times, num_vms):
    fitness = []
    for particle in x:
        vm_loads = [0] * num_vms
        for i, val in enumerate(particle):
            vm = int(val * num_vms) % num_vms
            vm_loads[vm] += task_times[i]
        fitness.append(max(vm_loads))
    return np.array(fitness)

def pso_scheduler(task_times, num_vms):
    dim = len(task_times)
    options = {'c1': 1.5, 'c2': 1.5, 'w': 0.7}
    optimizer = GlobalBestPSO(n_particles=20, dimensions=dim, options=options,
                              bounds=(np.zeros(dim), np.ones(dim)))
    cost, pos = optimizer.optimize(lambda x: evaluate_particle(x, task_times, num_vms), iters=50)
    allocation = [int(vm * num_vms) % num_vms for vm in pos]
    return allocation, cost

# -----------------------------
# ACO Scheduler
def aco_scheduler(task_times, num_vms, num_ants=20, iterations=50, alpha=1, beta=2, rho=0.5):
    num_tasks = len(task_times)
    pheromone = np.ones((num_tasks, num_vms))
    best_allocation = None
    best_makespan = float('inf')
    for _ in range(iterations):
        for _ in range(num_ants):
            allocation = []
            vm_loads = [0] * num_vms
            for t in range(num_tasks):
                heuristic = 1 / (np.array(vm_loads) + 1)
                probs = pheromone[t] ** alpha * heuristic ** beta
                probs /= probs.sum()
                chosen_vm = np.random.choice(range(num_vms), p=probs)
                allocation.append(chosen_vm)
                vm_loads[chosen_vm] += task_times[t]
            makespan = max(vm_loads)
            if makespan < best_makespan:
                best_makespan = makespan
                best_allocation = allocation
        pheromone *= (1 - rho)
        for t in range(num_tasks):
            pheromone[t][best_allocation[t]] += 1 / best_makespan
    return best_allocation, best_makespan

# -----------------------------
# Metric Calculation
def get_metrics(alloc, makespan, runtime, task_times, num_vms):
    vm_loads = [0] * num_vms
    for i, vm in enumerate(alloc):
        vm_loads[vm] += task_times[i]
    lbf = np.std(vm_loads) / np.mean(vm_loads)
    utilization = sum(task_times) / (makespan * num_vms) * 100
    throughput = len(task_times) / makespan
    idle_times = [makespan - load for load in vm_loads]
    return {
        'makespan': makespan,
        'runtime': runtime,
        'lbf': lbf,
        'utilization': utilization,
        'throughput': throughput,
        'idle_times': idle_times
    }

# -----------------------------
# Runner
def run_all():
    print("Task Times:", task_times)
    metrics = {}

    # GA-RM
    print("\nRunning GA-RM...")
    start = time.time()
    alloc_ga, ms_ga = run_ga_rm(task_times, NUM_VMS)
    end = time.time()
    metrics['GA-RM'] = get_metrics(alloc_ga, ms_ga, end - start, task_times, NUM_VMS)

    # Min-Min
    print("Running Min-Min...")
    start = time.time()
    alloc_mm, ms_mm = min_min_scheduler(task_times, NUM_VMS)
    end = time.time()
    metrics['Min-Min'] = get_metrics(alloc_mm, ms_mm, end - start, task_times, NUM_VMS)

    # PSO
    print("Running PSO...")
    start = time.time()
    alloc_pso, ms_pso = pso_scheduler(task_times, NUM_VMS)
    end = time.time()
    metrics['PSO'] = get_metrics(alloc_pso, ms_pso, end - start, task_times, NUM_VMS)

    # ACO
    print("Running ACO...")
    start = time.time()
    alloc_aco, ms_aco = aco_scheduler(task_times, NUM_VMS)
    end = time.time()
    metrics['ACO'] = get_metrics(alloc_aco, ms_aco, end - start, task_times, NUM_VMS)

    # Display results
    print("\n=== Comparative Results ===")
    for algo, m in metrics.items():
        print(f"{algo}: Makespan={m['makespan']}, Runtime={m['runtime']:.2f}s, LBF={m['lbf']:.4f}, "
              f"Util={m['utilization']:.2f}%, TP={m['throughput']:.4f}, Idle={m['idle_times']}")

    # Plotting
    def plot_metric(metric_name, ylabel):
        values = [metrics[a][metric_name] for a in metrics]
        plt.figure(figsize=(8, 5))
        plt.bar(metrics.keys(), values, color=['blue', 'green', 'orange', 'purple'])
        plt.title(f"{metric_name.capitalize()} Comparison")
        plt.ylabel(ylabel)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    plot_metric("makespan", "Makespan")
    plot_metric("lbf", "Load Balancing Factor")
    plot_metric("utilization", "Utilization (%)")
    plot_metric("throughput", "Throughput (Tasks/Time Unit)")
    plot_metric("runtime", "Execution Time (s)")

# -----------------------------
# Run the full runner
if __name__ == "__main__":
    run_all()
