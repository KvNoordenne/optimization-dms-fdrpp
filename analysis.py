import json
import os

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def result_analysis():
    files = []
    for x in os.listdir():
        if x.endswith(".json"):
            files.append(x)

    size_map = {'30': 'Small', '50': 'Medium', '70': 'Large'}
    stoch_map = {'0.0': 'No', '0.1': 'Medium', '0.2': 'High'}
    models = ['GRASP', 'SA', 'GA', 'ABC', 'TABU', 'ALNS']

    results = {s: {st: {m: [] for m in models} for st in stoch_map.values()} for s in size_map.values()}
    driver_costs = {s: {st: {m: [] for m in models} for st in stoch_map.values()} for s in size_map.values()}
    customer_costs = {s: {st: {m: [] for m in models} for st in stoch_map.values()} for s in size_map.values()}
    all_runtimes = {s: {st: {m: [] for m in models} for st in stoch_map.values()} for s in size_map.values()}
    all_evaluations = {s: {st: {m: [] for m in models} for st in stoch_map.values()} for s in size_map.values()}
    all_batching = {s: {st: {m: [] for m in models} for st in stoch_map.values()} for s in size_map.values()}

    pres_results = {'Small': {m: [] for m in models}, 'Medium': {m: [] for m in models}}
    orders = {}
    avg_run_times = {}
    avg_evaluations = {}
    comp_batch = {}
    batching = {}

    for file in files:
        with open(file, 'r') as f:
            log = json.load(f)

        name = file[:-5].split('_')
        size = size_map[name[5]]
        stoch = stoch_map[name[2]]
        pressure = round(int(name[6]) / int(name[7]))

        file_tot_batching = {m: [] for m in models}
        file_act_batching = {m: [] for m in models}

        for run in log.values():
            for method, steps in run.items():
                if method not in models:
                    continue

                last_step = list(steps.values())[-3]
                timesteps = int(list(steps.keys())[-3])
                optimizations = (timesteps / 5) + 1
                num_drivers = len(list(steps['0']['INVENTORIES'].values()))

                fitness = last_step['TC']
                c_costs = last_step['CC']
                d_costs = last_step['DC']
                avg_run = steps['time'] / optimizations
                avg_eval = steps['evaluations'] / optimizations

                carried = 0
                for t in list(steps.keys())[:-2]:
                    inv = sum(list(int(x) for x in steps[t]['INVENTORIES'].values()))
                    carried += inv

                tbu = carried / (timesteps * num_drivers)  # total batching utilization

                active_inv = 0
                active_t = 0
                for t in list(steps.keys())[:-2]:
                    for d in range(num_drivers):
                        if len(steps[t]['ROUTES'][str(d)]) > 1:
                            active_inv += int(steps[t]['INVENTORIES'][str(d)])
                            active_t += 1
                abu = active_inv / active_t  # active batching utilization

                if pressure == 10:
                    results[size][stoch][method].append(fitness)
                    customer_costs[size][stoch][method].append(c_costs)
                    driver_costs[size][stoch][method].append(d_costs)
                    all_runtimes[size][stoch][method].append(avg_run)
                    all_evaluations[size][stoch][method].append(avg_eval)
                    all_batching[size][stoch][method].append(abu)
                    file_tot_batching[method].append(tbu)
                    file_act_batching[method].append(abu)
                else:
                    pres_results[size][method].append(fitness)

        if pressure == 10:
            if size in ['Small', 'Large']:
                if size not in avg_run_times:
                    avg_run_times[size] = {}
                    avg_evaluations[size] = {}
                if stoch in ['No', 'High']:
                    avg_run_times[size][stoch] = all_runtimes[size][stoch]
                    avg_evaluations[size][stoch] = all_evaluations[size][stoch]
            if size not in orders:
                orders[size] = log['0']['ORDERS']
            if size == 'Large':
                if stoch == 'No':
                    comp_batch['Total'] = file_tot_batching
                    comp_batch['Active'] = file_act_batching
                batching[stoch] = file_act_batching

    print('RESULTS', results)
    print('PRES RESULTS', pres_results)
    print('AVG RUN TIMES', avg_run_times)
    print('AVG EVALUATIONS', avg_evaluations)
    print('BATCHING COMP', comp_batch)
    print('ORDERS', orders)

    colors = ['#88CCEE', '#CC6677', '#DDCC77', '#117733', '#332288', '#AA4499']

    datasets = {
        "Total Fitness": {"data": results, "prec": "%.1f"},
        "Customer Costs": {"data": customer_costs, "prec": "%.1f"},
        "Driver Costs": {"data": driver_costs, "prec": "%.1f"},
        "Average Runtimes (s)": {"data": all_runtimes, "prec": "%.3f"},
        "Average Evaluations": {"data": all_evaluations, "prec": "%.0f"},
        "Active Batching": {"data": all_batching, "prec": "%.2f"}
    }

    for name, config in datasets.items():
        print(f"GENERATING FULL TABLE {name}")
        metric_data = config['data']
        total_rows = []
        # Create table with results and fitness boxplots for regular environments
        for s_key in ['Small', 'Medium', 'Large']:
            if name == 'Total Fitness':
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            for i, st_key in enumerate(['No', 'Medium', 'High']):
                for model in models:
                    runs = metric_data[s_key][st_key][model]

                    v_min = np.min(runs)
                    q1 = np.percentile(runs, 25)
                    median = np.median(runs)
                    q3 = np.percentile(runs, 75)
                    v_max = np.max(runs)

                    total_rows.append({
                        'Size': s_key,
                        'Stoch.': st_key,
                        'Model': model,
                        'Min': v_min,
                        '1st Q': q1,
                        'Median': median,
                        '3rd Q': q3,
                        'Max': v_max,
                        'Range': v_max - v_min,
                        'Mean': np.mean(runs),
                        'Std.': np.std(runs)
                    })
                if name == 'Total Fitness':
                    ax = axes[i]
                    bp = ax.boxplot(metric_data[s_key][st_key].values(),
                                    tick_labels=list(metric_data[s_key][st_key].keys()), patch_artist=True)
                    for box, color in zip(bp['boxes'], colors):
                        box.set_facecolor(color)
                        box.set_alpha(0.7)
                    ax.set_title(f'{st_key} Stochasticity')
                    ax.set_xlabel('Model')
                    if i == 0:
                        ax.set_ylabel('Fitness')
            if name == 'Total Fitness':
                fig.suptitle(f'Fitness of models in {s_key} environment', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(f'figures/{s_key}_fitness.png')
                plt.show()

        df = pd.DataFrame(total_rows)
        pd.set_option('display.width', None)
        print(df)
        latex_code = df.to_latex(
            column_format="lll | rrrrrrrrr",
            multicolumn=True,
            bold_rows=False,
            float_format=config['prec'],
            index=False
        )
        print(latex_code)

    for name, config in datasets.items():
        print(f'GENERATE MEDIAN TABLE FOR {name}')
        data = config['data']
        table_data = {}
        for size in ["Small", "Medium", "Large"]:
            no_stoch = data[size]["No"]
            med_stoch = data[size]["Medium"]
            high_stoch = data[size]["High"]

            no_medians = [np.median(no_stoch[m]) for m in models]
            med_medians = [np.median(med_stoch[m]) for m in models]
            high_medians = [np.median(high_stoch[m]) for m in models]

            med_p_change = [
                f"{((med - no) / no) * 100:.1f}\\%" for med, no in zip(med_medians, no_medians)
            ]
            high_p_change = [
                f"{((high - no) / no) * 100:.1f}\\%" for high, no in zip(high_medians, no_medians)
            ]

            table_data[(size, "No")] = no_medians
            table_data[(size, "Med.")] = med_medians
            table_data[(size, "Med. %")] = med_p_change
            table_data[(size, "High")] = high_medians
            table_data[(size, "High %")] = high_p_change

        columns = pd.MultiIndex.from_tuples(table_data.keys(), names=["Size", "Stoch."])
        df = pd.DataFrame(table_data, index=models, columns=columns)
        df = df.rename(columns={"Med. %": "", "High %": ""}, level=1)
        print(df)

        latex_code = df.to_latex(
            column_format="l | rrr rrr rrr",
            multicolumn=True,
            bold_rows=False,
            float_format=config['prec']
        )
        print(latex_code)

    # Create boxplots for pressured environments
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for i, s_key in enumerate(['Small', 'Medium']):
        ax = axes[i]
        bp = ax.boxplot(pres_results[s_key].values(), tick_labels=list(pres_results[s_key].keys()),
                        patch_artist=True)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        ax.set_title(f'{s_key} environment')
        ax.set_xlabel('Model')
        if i == 0:
            ax.set_ylabel('Fitness')
    fig.suptitle(f'Fitness of models in High pressure environments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/pressure_fitness.png')
    plt.show()

    # Create boxplots for runtime and evaluation comparisons
    for plot_type, var_dict in zip(['run time', 'evaluations'], [avg_run_times, avg_evaluations]):
        print('VARRR', var_dict)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        for i, s_key in enumerate(['Small', 'Large']):
            for j, st_key in enumerate(['No', 'High']):
                ax = axes[i][j]
                bp = ax.boxplot(var_dict[s_key][st_key].values(), tick_labels=list(var_dict[s_key][st_key].keys()),
                                patch_artist=True)
                for box, color in zip(bp['boxes'], colors):
                    box.set_facecolor(color)
                    box.set_alpha(0.7)
                ax.set_title(f'{s_key} environment: {st_key} stochasticity')
                ax.set_xlabel('Model')
                if j == 0:
                    if plot_type == 'run time':
                        ax.set_ylabel('Avg run time (s)')
                    else:
                        ax.set_ylabel('Avg evaluations')
        top_ax_1, top_ax_2 = axes[0][0], axes[0][1]
        bottom_ax_1, bottom_ax_2 = axes[1][0], axes[1][1]
        top_ax_1.sharey(top_ax_2)
        bottom_ax_1.sharey(bottom_ax_2)
        fig.suptitle(f'Average {plot_type} per optimization of models in different environments', fontsize=16,
                     fontweight='bold')
        plt.tight_layout()
        if plot_type == 'run time':
            plt.savefig(f'figures/runtime.png')
        else:
            plt.savefig(f'figures/evaluations.png')
        plt.show()

    # Plot comparison in batching metrics
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    for i, b_key in enumerate(['Total', 'Active']):
        ax = axes[i]
        bp = ax.boxplot(comp_batch[b_key].values(), tick_labels=list(comp_batch[b_key].keys()),
                        patch_artist=True)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        ax.set_title(f'{b_key} Batching Utilization')
        ax.set_xlabel('Model')
        if i == 0:
            ax.set_ylabel('Value')
    fig.suptitle(f'Comparison of batching utilization metrics in deterministic large environments', fontsize=16,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/batching_comparison.png')
    plt.show()

    # Plot active batching in large environments
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, s_key in enumerate(['No', 'Medium', 'High']):
        ax = axes[i]
        bp = ax.boxplot(batching[s_key].values(), tick_labels=list(batching[s_key].keys()),
                        patch_artist=True)
        for box, color in zip(bp['boxes'], colors):
            box.set_facecolor(color)
            box.set_alpha(0.7)
        ax.set_title(f'{s_key} stochasticity')
        ax.set_xlabel('Model')
        if i == 0:
            ax.set_ylabel('Active Batching Utilization')
    fig.suptitle(f'Comparison of ABU in large environments with different stochasticity levels', fontsize=16,
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/active_batching.png')
    plt.show()

    # Create order distribution graph
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    inv_size_map = {'Small': 30, 'Medium': 50, 'Large': 70}

    for i, size in enumerate(list(orders.keys())):
        ax = axes[i]
        limit = inv_size_map[size]

        xp, yp, xd, yd = [], [], [], []
        for order_str in orders[size]:
            coordinates = [int(d) for d in re.findall(r'\d+', order_str)]

            xp.append(coordinates[0])
            yp.append(coordinates[1])
            xd.append(coordinates[2])
            yd.append(coordinates[3])
        xp = np.array(xp)
        yp = np.array(yp)
        xd = np.array(xd)
        yd = np.array(yd)
        ax.plot(xp, yp, 'ro', label='Pickups')
        ax.plot(xd, yd, 'bo', label='Deliveries')
        ax.grid()
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_title(f"Environment size: {size}")
        ax.set_xlabel('X Coordinate')
        if i == 0:
            ax.set_ylabel('Y Coordinate')
        ax.legend(loc='upper right')

    fig.suptitle('Distribution of pickup and delivery locations', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'figures/order_distributions.png')
    plt.show()