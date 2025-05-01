import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randrange
import matplotlib.animation as animation
from collections import Counter
import itertools
import copy

def initialize_graph(graph_type, n, initial_mutants, num_muts, center_mutant=False):
    """
    Initialize a graph based on the specified type and set some initial mutants.
    
    Parameters:
    - graph_type: str, type of the graph ('complete', 'star', 'line', 'circle', '2D_lattice', '3D_lattice')
    - n: int, parameter for graph size (interpreted differently for different graph types)
    - initial_mutants: list, vector of each type of initial mutant
    - center_mutant: bool, whether the center of the star graph should be a mutant (only applicable for 'star' graph)
    
    Returns:
    - G: networkx.Graph, the initialized graph with mutant states set
    """
    if graph_type == 'complete':
        G = nx.complete_graph(n)
    elif graph_type == 'star':
        G = nx.star_graph(n - 1)
    elif graph_type == 'line':
        G = nx.path_graph(n)
    elif graph_type == 'circle':
        G = nx.cycle_graph(n)
    elif graph_type == '2D_lattice':
        G = nx.grid_2d_graph(n, n, periodic=True)
        mapping = {node: i for i, node in enumerate(G.nodes())}
        G = nx.relabel_nodes(G, mapping)
    elif graph_type == '3D_lattice':
        G = nx.grid_graph(dim=[n, n, n], periodic=True)
    else:
        raise ValueError("Invalid graph type specified.")
       
    nodes = list(G.nodes()) 

    wild_type = [0, 0, 0]
    #nx.set_node_attributes(G, 0, 'state')
    nx.set_node_attributes(G, copy.deepcopy(wild_type), 'state')

    if graph_type == 'star' and center_mutant:
        # rand_index = random.randint(0, num_muts - 1)
        G.nodes[0]['state'] = [1, 0, 0]
        # Center node of star is a mutant

        remaining_mutants = sum(initial_mutants) - 1
        if remaining_mutants > 0:
            mutants = np.random.choice(nodes[1:], remaining_mutants, replace=False)
        else:
            mutants = []
    else:
        mutants = np.random.choice(nodes, sum(initial_mutants), replace=False)
    """
    mut1_locations = [411, 412, 386, 387, 361, 362, 336, 337, 312, 287, 288, 262, 263, 237, 238, 212, 213]
    mut2_locations = [388, 389, 390, 391, 363, 364, 365, 366, 338, 339, 340, 341, 313, 314, 315, 316]
    mut3_locations = [308, 309, 310, 311, 283, 284, 285, 286, 258, 259, 260, 261, 233, 234, 235, 236]
    for index in mut1_locations:
        G.nodes[index]['state'] = [1, 0, 0]
    for index in mut2_locations:
        G.nodes[index]['state'] = [0, 1, 0]
    for index in mut3_locations:
        G.nodes[index]['state'] = [0, 0, 1]
    """
    mut1, mut2, mut3 = initial_mutants
    for node in mutants[0:mut1]:
        G.nodes[node]['state'] = [1, 0, 0]
    for node in mutants[mut1:mut1+mut2]:
        G.nodes[node]['state'] = [0, 1, 0]
    for node in mutants[mut1+mut2:]:
        G.nodes[node]['state'] = [0, 0, 1]
    return G

def step(G, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight):
    num_states = np.power(2, num_muts)

    N = G.number_of_nodes() #number of nodes/vertices. ex. 15

    # randomly decide if the step is a mutation state increase or if it's moran
    # p = 0.2 # p is the chance that we just increase instead of doing BD process
    count_max = sum(1 for node in G.nodes() if G.nodes[node]['state'] == list(np.ones(num_muts)))
    #p = (N - count_max)/(N - count_max + N* (N-1))
    p = 0.1
    if random.random() < p:
        model = "increase"

    """
    Perform one step of the Moran process.
    """
    # num_mutants = sum(nx.get_node_attributes(G, 'state').values())
    
    b = bfitness_vector 
    d = dfitness_vector

    if model == 'BD':
        bfitness = np.zeros(N) # len 15
        for i in range(N):
            state = G.nodes[i]['state']
            # state = [1, 0, 0] for example
            bfitness[i] = 1 + sum(fitness for m, fitness in zip(state, b) if m) / N
            # bfitness[i] = BF[state]
        # bfitness = np.array([BF1 if G.nodes[node]['state'] == 1 else BF2 if G.nodes[node]['state'] == 2 else BF3 if G.nodes[node]['state'] == 3 else  1 for node in G.nodes()])
        total_bfitness = np.sum(bfitness)
        bprobabilities = bfitness / total_bfitness
        reproducing_node = np.random.choice(list(G.nodes()), p=bprobabilities)
            
        neighbors = list(G.neighbors(reproducing_node))
        if neighbors:
            dfitness = np.zeros(len(neighbors)) # len 15
            for i in range(len(neighbors)):
                state = G.nodes[i]['state']
                dfitness[i] = 1 + sum(fitness for m, fitness in zip(state, d) if m) / N
            # dfitness = np.array([DF1 if G.nodes[node]['state'] == 1 else DF2 if G.nodes[node]['state'] == 2 else DF3 if G.nodes[node]['state'] == 3 else 1 for node in neighbors])
            total_dfitness = np.sum(dfitness)
            dprobabilities = dfitness / total_dfitness
            replacing_node = np.random.choice(neighbors, p=dprobabilities)
            G.nodes[replacing_node]['state'] = G.nodes[reproducing_node]['state']
    elif model == 'DB':
        dfitness = np.zeros(N) # len = # of neighbors
        for i in range(N):
            state = G.nodes[i]['state']
            dfitness[i] = 1 - sum(fitness for m, fitness in zip(state, d) if m) / N
        #dfitness = np.array([DF if G.nodes[node]['state'] == 1 else 1 for node in G.nodes()])
        total_dfitness = np.sum(dfitness)
        dprobabilities = dfitness / total_dfitness
        replacing_node = np.random.choice(list(G.nodes()), p=dprobabilities)
        
        neighbors = list(G.neighbors(replacing_node))
        if neighbors:
            bfitness = np.zeros(len(neighbors)) # length = # of neighbors
            for i in range(len(neighbors)):
                state = G.nodes[i]['state']
                bfitness[i] = 1 + sum(fitness for m, fitness in zip(state, b) if m) / N
            # bfitness = np.array([BF if G.nodes[node]['state'] == 1 else 1 for node in neighbors])
            total_bfitness = np.sum(bfitness)
            bprobabilities = bfitness / total_bfitness
            reproducing_node = np.random.choice(neighbors, p=bprobabilities)
            G.nodes[replacing_node]['state'] = G.nodes[reproducing_node]['state']
    elif model == 'increase':
        nodes = list(G.nodes())
        #print("   ")
        nonfinal_node = [node for node in nodes if G.nodes[node]['state'] != [1, 1, 1]]
        random_index = random.randint(0, len(nonfinal_node)-1)
        random_node = nonfinal_node[random_index]
        # random_node = node number 5
        mutations = G.nodes[random_node]['state']
        #print(mutations)
        # mutations = [0, 1, 0] --> state of node 5

        denom = 0
        for i in range(len(mutations)):
            if mutations[i] == 0:
                denom += mutation_weight[i]
        # sum_weights = sum(weight for m, weight in zip(mutations, mutation_weight) if not m)
        # denom == q1 + q3

        prob_selection = []
        for i in range(len(mutations)):
            if mutations[i] == 0:   
                prob_selection.append(mutation_weight[i] / denom)
            else: 
                prob_selection.append(0)
        # prob_selection = [q1/(q1+q3), 0, q3/(q1+q3)]

        random_mut = random.choices(range(len(prob_selection)), weights=prob_selection)[0]
        # random_mut = 1 or 3
        temp = copy.deepcopy(mutations)
        temp[random_mut] = 1
        G.nodes[random_node]['state'] = temp
        #print("change node number", random_index)
        #print(G.nodes[random_node]['state'])
    else:
        raise ValueError("Invalid model type specified.")
    nodes = list(G.nodes())
    #print("all states")
    #for node in nodes:
        #print("node number", node, "is", G.nodes[node]['state'])
def assign_colors_to_states(G, num_muts):
    # G.nodes[0]['state'] = 1 
    permutations = list(itertools.product([0, 1], repeat=num_muts))
    color_bank = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan']
    all_colors = {}
    for i in range(len(permutations)):
        state = permutations[i]
        all_colors[state] = color_bank[i]
    # all_colors is a dictionary now - [0, 0, 0]: 'blue', etc.

    node_colors = []
    nodes = list(G.nodes())
    for node in nodes:
        mut_state = G.nodes[node]['state']
        node_colors.append(all_colors.get(tuple(mut_state)))
    return node_colors

def plot_graph(G, pos, ax, step_count, mode, num_muts):
    """
    Plot the graph with node states.
    """
    # update this because we need more colors

    colors = assign_colors_to_states(G, num_muts)

    #colors = ['green' if G.nodes[node]['state'] == 3 else 'orange' if G.nodes[node]['state'] == 2 else 'red' if G.nodes[node]['state'] == 1 else 'blue' for node in G.nodes()]
    ax.clear()
    nx.draw(G, pos, node_color=colors, node_size=100, with_labels=False, edge_color='gray', ax=ax)
    ax.set_title(f'{model} process: {graph_type.capitalize()} Graph - Generation= {step_count}')


def animate(G, pos, ax1, ax2, history, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight):
    """
    # Generator function for the animation.
    """
    num_states = np.power(2, num_muts)
    n = G.number_of_nodes()
    step_count = 0
    num_of_state = list(np.zeros(num_states)) # number of cells per state for all states
    curr = True
    permutations_tuples = list(itertools.product([0, 1], repeat=num_muts))
    permutations = [list(t) for t in permutations_tuples] #[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], each represents a state
    while curr:
        step(G, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight)
        step_count += 1
        plot_graph(G, pos, ax1, step_count, model, num_muts)
        for i in range(num_states):
            num_of_state[i] = list(nx.get_node_attributes(G, 'state').values()).count(permutations[i])
            proportion_mutants = num_of_state[i] / n
            history[i].append(proportion_mutants)

        # history.append(proportion_mutants)
        color_bank = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan']
        ax2.clear()
        for i in range(len(history)):
            ax2.plot(history[i], color = color_bank[i], label = "State" + str(i))
        # ax2.plot(history, color='red')
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Proportion of Mutants')
        ax2.set_title('Proportion of Mutants Over Time')
        ax2.legend()
        #print(num_of_state)
        if n in num_of_state:
            curr = False
        
        yield

def prompt_user_for_parameters():
    """
    #Prompt the user to input parameters for the Moran process animation.
    """
    graph_type = input("Enter the graph type ('complete', 'star', 'line', 'circle', '2D_lattice', '3D_lattice'): ")
    n = int(input("Enter the size parameter 'n': "))

    input1 = input("Enter the initial number of mutants as a vector: ")
    stripped1 = input1.strip('[] ')
    num1 = stripped1.split(',')
    initial_mutants = [int(num) for num in num1]

 
    model = input("Enter the model type ('BD', 'DB'): ")
    

    center_mutant = False
    if graph_type == 'star':
        center_mutant_input = input("Should the center of the star graph be a mutant? (yes/no): ").strip().lower()
        center_mutant = center_mutant_input == 'yes'
    
    num_muts = int(input("Enter the number of mutations 'm': "))
    if num_muts > 3:
        print("Maximum number of mutations exceeded. Number of mutations has been set to 3.")
        num_muts = 3

    input_str = input("Enter the birth fitness vector for each mutation: ")
    stripped_str = input_str.strip('[] ')
    num_strs = stripped_str.split(',')
    bfitness_vector = [int(num_str) for num_str in num_strs]


    input_str2 = input("Enter the death fitness vector for each mutation: ")
    stripped_str2 = input_str2.strip('[] ')
    num_strs2 = stripped_str2.split(',')
    dfitness_vector = [int(num_str2) for num_str2 in num_strs2]

    input_str3 = input("Enter the weight of each mutation as a vector: ")
    stripped_str3 = input_str3.strip('[] ')
    num_strs3 = stripped_str3.split(',')
    raw_mutation_weight = [float(num_str3) for num_str3 in num_strs3]
    total_weight = sum(raw_mutation_weight)
    mutation_weight = [x/total_weight for x in raw_mutation_weight]

    return graph_type, n, initial_mutants, model, center_mutant, num_muts, bfitness_vector, dfitness_vector, mutation_weight

def run_animation():
    G = initialize_graph(graph_type, n, initial_mutants, num_muts, center_mutant)
    # Position the nodes for plotting
    if graph_type == '2D_lattice':
        pos = {node: (node % n, node // n) for node in G.nodes()}
    elif graph_type == '3D_lattice':
        pos = {node: (node[0] + node[1]*n, node[2]) for node in G.nodes()}
    elif graph_type == 'circle':   
        pos = nx.circular_layout(G)
    elif graph_type == 'line':
        pos = {node: (node, 0) for node in G.nodes()}
    else:
        pos = nx.spring_layout(G)


    num_states = np.power(2, num_muts)
    permutations_tuples = list(itertools.product([0, 1], repeat=num_muts))
    permutations = [list(t) for t in permutations_tuples] #[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], each represents a state
    history = []

    for i in range(num_states):
        history.append([sum(1 for node in G.nodes() if G.nodes[node]['state'] == permutations[i]) / n])




    # Create a figure for the animation with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [2, 1]})

    # Plot the initial state of the graph
    plot_graph(G, pos, ax1, 0, model, num_muts)

    # Create the generator
    animation_generator = animate(G, pos, ax1, ax2, history, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight)

    # Create the animation
    ani = animation.FuncAnimation(fig, lambda i: next(animation_generator), repeat=False, interval=100)

    # # Save the animation
    #ani.save('moran_process_animation_mutations.mp4', writer='ffmpeg', fps=80)

    plt.show()

def mut_type_average(trials,steps,graph_type, n, initial_mutants, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight):
    sample_1 = [621, 622, 623, 624, 596, 597, 598, 599, 571, 572, 573, 574, 546, 547, 548, 549]
    sample_2 = [75, 76, 77, 78, 50, 51, 52, 53, 25, 26, 27, 28, 0, 1, 2, 3]
    
    #G = initialize_graph(graph_type, n, initial_mutants, num_muts, center_mutant)
    num_states = np.power(2, num_muts)
    history_accumulator = [np.zeros(steps) for _ in range(num_states)]
    mut_type_accumulator = [np.zeros(steps) for _ in range(3)]
    z_vector_accumulator = np.zeros(steps)
    initiation_time_accumulator = np.zeros(trials)
    
    permutations_tuples = list(itertools.product([0, 1], repeat=num_muts))
    permutations = [list(t) for t in permutations_tuples] #[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], each represents a state
    state_111 = permutations.index([1, 1, 1])

    for trial in range(trials):
        G = initialize_graph(graph_type, n, initial_mutants, center_mutant)
        N = G.number_of_nodes()
        history = [[] for _ in range(num_states)] #list of lists, each list is the proportion of each type (proportion_mutants) across every step
        mut_type = [[] for _ in range(3)]
        z_vector = np.zeros(steps)
        initiation_time = -1
        for j in range(steps):
            step(G, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight)
            num_of_state = list(np.zeros(num_states)) # number of cells per state for all states
            for i in range(num_states):
                for index in sample_1: 
                    if G.nodes[index]['state'] == permutations[i]:
                        num_of_state[i]+= 1
                proportion_mutants = num_of_state[i] / len(sample_1)
                # num_of_state[i] = list(nx.get_node_attributes(G, 'state').values()).count(permutations[i])
                # proportion_mutants = num_of_state[i] / N
                history[i].append(proportion_mutants) #history[0] = list of number of cells in each state 
            mut_type[0].append(sum(history[k][j] for k in [4, 5, 6, 7]))
            mut_type[1].append(sum(history[k][j] for k in [2, 3, 6, 7]))
            mut_type[2].append(sum(history[k][j] for k in [1, 3, 5, 7]))
            
            if num_of_state[state_111] > 0:
                z_vector[j] = 1
                if initiation_time == -1:
                    initiation_time = j  

        # Accumulate results
        for i in range(num_states):
            history_accumulator[i] += np.array(history[i]) # add history for each trial to the general running list
        
        for k in range(3):
            mut_type_accumulator[k] += np.array(mut_type[k])
    
        z_vector_accumulator += z_vector
        initiation_time_accumulator[trial] = initiation_time
        print(f'Trial {trial} complete.')

    # Calculate averages
    average_history = [history_accumulator[i] / trials for i in range(num_states)] #list of length 8, each list is avg of proportion of each type across steps
    # ex. avg_history[0] = avg proportion of type 0 from step 1 to 2000
    average_mut_type = [mut_type_accumulator[k] / trials for k in range(3)] # list of length 3, each list is avg proportion of each mutation across steps
    # ex. avg_mut_type[0] = avg proportion of mutation 1 from step 1 to 2000
    average_z_vector = z_vector_accumulator / trials
    average_initiation_time = np.mean(initiation_time_accumulator)

    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_history_lattice.npy', average_history)
    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_mut_type_lattice.npy', average_mut_type)
    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_z_vector_lattice.npy', average_z_vector)
    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_init_time_lattice.npy', average_initiation_time)


    # Plot average_history
    plt.figure(figsize=(14, 7))
    color_bank = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan']
    for i in range(num_states):
        plt.plot(average_history[i],color=color_bank[i], label=f'State {i}')
    plt.title('Expected Proportion of Mutants Over Time')
    plt.xlabel('Time')
    plt.ylabel('Expected Proportion of Mutants')
    plt.legend()
    # plt.savefig('/Users/kyletsai/Desktop/simulation_data/avg_history_lattice.png')
    plt.show()

    # Plot average_mut_type
    plt.figure(figsize=(14, 7))
    for k in range(3):
        plt.plot(average_mut_type[k], label=f'Type {k+1}')
    plt.title('Expected Proportion of Types Over Time')
    plt.xlabel('Time')
    plt.ylabel('Expected Proportion of Types')
    plt.legend()
    # plt.savefig('/Users/kyletsai/Desktop/simulation_data/avg_mut_type_lattice.png')
    plt.show()
    
    # Plot average_z_vector
    plt.figure(figsize=(14, 7))
    plt.plot(average_z_vector, label='z_vector')
    #plt.title('Probability of Initiation Up to Time t')
    plt.xlabel('Time')
    plt.ylabel('Probability of Initiation Up to Time t')
    plt.legend()
    # plt.savefig('/Users/kyletsai/Desktop/simulation_data/avg_z_vector_lattice.png')
    plt.show()
    
    # Print average initiation time
    print(f'Average Initiation Time: {average_initiation_time}') 

    #return average_history, average_mut_type, average_z_vector, average_initiation_time
    return  average_initiation_time

# Prompt user for parameters
graph_type, n, initial_mutants, model, center_mutant, num_muts, bfitness_vector, dfitness_vector, mutation_weight = prompt_user_for_parameters()

# Initialize and run the Moran process
G = initialize_graph(graph_type, n, initial_mutants, num_muts, center_mutant)

#print(mut_type_average(100,6000,graph_type, n, initial_mutants, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight))
run_animation()