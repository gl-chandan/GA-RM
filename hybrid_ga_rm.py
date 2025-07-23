# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 19:57:23 2025

@author: Dell
"""

import numpy as np
import random
import time

# --- Configuration
POP_SIZE = 30
NUM_GENERATIONS = 100
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.8

# --- Fitness Function
def fitness(chromosome, task_times, num_vms):
    vm_loads = [0] * num_vms
    for task_idx, vm in enumerate(chromosome):
        vm_loads[vm] += task_times[task_idx]
    return max(vm_loads)

# --- Heuristic Initialization
def heuristic_allocation(task_times, num_vms):
    num_tasks = len(task_times)
    vm_loads = [0] * num_vms
    allocation = [-1] * num_tasks
    task_indices = sorted(range(num_tasks), key=lambda i: -task_times[i])
    for idx in task_indices:
        min_vm = np.argmin(vm_loads)
        allocation[idx] = min_vm
        vm_loads[min_vm] += task_times[idx]
    return allocation

# --- Uniform Crossover
def uniform_crossover(p1, p2):
    child1, child2 = [], []
    for i in range(len(p1)):
        if random.random() < 0.5:
            child1.append(p1[i])
            child2.append(p2[i])
        else:
            child1.append(p2[i])
            child2.append(p1[i])
    return child1, child2

# --- Restricted Mutation (protect top 2 longest tasks)
def restricted_mutation(chromosome, task_times, num_vms):
    mutated = chromosome[:]
    num_tasks = len(task_times)
    protected_tasks = sorted(range(num_tasks), key=lambda x: -task_times[x])[:2]
    for i in range(num_tasks):
        if i in protected_tasks:
            continue
        if random.random() < MUTATION_RATE:
            mutated[i] = random.randint(0, num_vms - 1)
    return mutated

# --- Roulette Selection
def roulette_selection(population, fitnesses):
    total_fitness = sum([1 / f for f in fitnesses])
    probs = [(1 / f) / total_fitness for f in fitnesses]
    selected = random.choices(population, weights=probs, k=len(population))
    return selected

# --- Additional Metrics
def load_balancing_factor(vm_loads):
    return np.std(vm_loads) / np.mean(vm_loads)

def resource_utilization(task_times, makespan, num_vms):
    return (sum(task_times) / (makespan * num_vms)) * 100

def throughput(num_tasks, makespan):
    return num_tasks / makespan

def idle_times(vm_loads, makespan):
    return [makespan - load for load in vm_loads]

# --- Main GA Function with metrics
def run_ga_rm(task_times, num_vms):
    num_tasks = len(task_times)
    population = [heuristic_allocation(task_times, num_vms) for _ in range(POP_SIZE)]
    best_solution = None
    best_fitness = float('inf')
    fitness_history = []

    start_time = time.time()

    for gen in range(NUM_GENERATIONS):
        fitnesses = [fitness(ch, task_times, num_vms) for ch in population]
        best_idx = np.argmin(fitnesses)
        current_best = fitnesses[best_idx]

        # Update global best
        if current_best < best_fitness:
            best_solution = population[best_idx][:]
            best_fitness = current_best

        fitness_history.append(best_fitness)

        # Restart mechanism if stuck for 10 generations
        if gen > 10 and all(f == best_fitness for f in fitness_history[-10:]):
            for i in range(POP_SIZE):
                if i != best_idx:
                    population[i] = restricted_mutation(population[i], task_times, num_vms)

        # Selection
        parents = roulette_selection(population, fitnesses)

        # Crossover + Mutation
        next_gen = []
        for i in range(0, POP_SIZE, 2):
            p1, p2 = parents[i], parents[i + 1]
            if random.random() < CROSSOVER_RATE:
                c1, c2 = uniform_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            c1 = restricted_mutation(c1, task_times, num_vms)
            c2 = restricted_mutation(c2, task_times, num_vms)
            next_gen.extend([c1, c2])

        # Elitism: Keep the best if better
        new_fitnesses = [fitness(ch, task_times, num_vms) for ch in next_gen]
        if best_fitness < max(new_fitnesses):
            worst_idx = np.argmax(new_fitnesses)
            next_gen[worst_idx] = best_solution[:]

        population = next_gen

        print(f"Gen {gen + 1} | Best Makespan: {best_fitness}")

    end_time = time.time()
    exec_time = end_time - start_time

    # Compute Final VM Loads
    vm_loads = [0] * num_vms
    for i, vm in enumerate(best_solution):
        vm_loads[vm] += task_times[i]

    # Compute and print metrics
    lbf = load_balancing_factor(vm_loads)
    utilization = resource_utilization(task_times, best_fitness, num_vms)
    tp = throughput(num_tasks, best_fitness)
    idle = idle_times(vm_loads, best_fitness)

    print("\nBest Allocation (Task → VM):")
    for i, vm in enumerate(best_solution):
        print(f"Task {i + 1} (runtime={task_times[i]}) → VM {vm + 1}")
    print("\nFinal VM Loads:", vm_loads)
    print("Final Makespan:", best_fitness)
    print(f"Load Balancing Factor: {lbf:.4f}")
    print(f"Resource Utilization: {utilization:.2f}%")
    print(f"Throughput: {tp:.4f} tasks/unit time")
    print(f"Execution Time: {exec_time:.2f}s")
    print(f"Idle Times per VM: {idle}")

    return best_solution, best_fitness


