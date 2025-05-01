import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
from random import randrange
import matplotlib.animation as animation
from collections import Counter

def initialize_graph(graph_type, n, initial_mutants, center_mutant=False):
    """
    Initialize a graph based on the specified type and set some initial mutants.
    
    Parameters:
    - graph_type: str, type of the graph ('complete', 'star', 'line', 'circle', '2D_lattice', '3D_lattice')
    - n: int, parameter for graph size (interpreted differently for different graph types)
    - initial_mutants: int, number of initial mutants
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
    
    nx.set_node_attributes(G, 0, 'state')
    
    nodes = list(G.nodes())
    
    if graph_type == 'star' and center_mutant:
        G.nodes[0]['state'] = 1  # Center node of star is a mutant
        remaining_mutants = initial_mutants - 1
        if remaining_mutants > 0:
            mutants = np.random.choice(nodes[1:], remaining_mutants, replace=False)
        else:
            mutants = []
    else:
        mutants = np.random.choice(nodes, initial_mutants, replace=False)
    
    for node in mutants:
        G.nodes[node]['state'] = 1  # 1 represents mutant
    
    return G

# def DB():
#     bfitness = np.array([BF1 if G.nodes[node]['state'] == 1 else BF2 if G.nodes[node]['state'] == 2 else BF3 if G.nodes[node]['state'] == 3 else  1 for node in G.nodes()])
#     total_bfitness = np.sum(bfitness)
#     bprobabilities = bfitness / total_bfitness
#     reproducing_node = np.random.choice(list(G.nodes()), p=bprobabilities)
        
#     neighbors = list(G.neighbors(reproducing_node))
#     if neighbors:
#         dfitness = np.array([DF1 if G.nodes[node]['state'] == 1 else DF2 if G.nodes[node]['state'] == 2 else DF3 if G.nodes[node]['state'] == 3 else 1 for node in neighbors])
#         total_dfitness = np.sum(dfitness)
#         dprobabilities = dfitness / total_dfitness
#         replacing_node = np.random.choice(neighbors, p=dprobabilities)
#         G.nodes[replacing_node]['state'] = G.nodes[reproducing_node]['state']
    

def step(G, model, num_muts):
    num_states = num_muts + 1

    N = G.number_of_nodes() #number of nodes/vertices. ex. 15

    # randomly decide if the step is a mutation state increase or if it's moran
    # p = 0.2 # p is the chance that we just increase instead of doing BD process
    count_max = sum(1 for node in G.nodes() if G.nodes[node]['state'] == num_states - 1)
    p = (N - count_max)/(N - count_max + N* (N-1))
    if random.random() < p:
        model = "increase"

    """
    Perform one step of the Moran process.
    """
    # b = bfitness_vector #vector of length 7 with birth fitness of a cell with x number of mutations
    # d = dfitness_vector
    BF = np.ones(num_states) # length is number of states. ex. 7
    DF = np.ones(num_states)
    BF[0] = 1
    BF[1] = 1 + 0.3
    BF[2] = 1 + 0.6
    BF[3] = 1 + 0.9
    DF[0] = 1
    DF[1] = 1 - 0.12
    DF[2] = 1 - 0.24
    DF[3] = 1 - 0.36
    # for i in range(num_states):
    #     BF[i] += sum(b[0:i]) / N
    #     DF[i] -= sum(d[0:i]) / N 
    if model == 'BD':
        bfitness = np.zeros(N) # len 15
        for i in range(N):
            state = G.nodes[i]['state']
            # state = 2 for example
            bfitness[i] = BF[state]
        total_bfitness = np.sum(bfitness)
        bprobabilities = bfitness / total_bfitness
        reproducing_node = np.random.choice(list(G.nodes()), p=bprobabilities)
            
        neighbors = list(G.neighbors(reproducing_node))
        if neighbors:
            dfitness = np.zeros(len(neighbors)) # len 15
            for i in range(len(neighbors)):
                state = G.nodes[i]['state']
                dfitness[i] = DF[state]
            total_dfitness = np.sum(dfitness)
            dprobabilities = dfitness / total_dfitness
            print("death probabilities:", dprobabilities)
            replacing_node = np.random.choice(neighbors, p=dprobabilities)
            G.nodes[replacing_node]['state'] = G.nodes[reproducing_node]['state']
    elif model == 'DB':
        dfitness = np.zeros(N) 
        for i in range(N):
            state = G.nodes[i]['state']
            dfitness[i] = DF[state]
        total_dfitness = np.sum(dfitness)
        dprobabilities = dfitness / total_dfitness
        replacing_node = np.random.choice(list(G.nodes()), p=dprobabilities)
        
        neighbors = list(G.neighbors(replacing_node))
        if neighbors:
            bfitness = np.zeros(len(neighbors)) # length = # of neighbors
            for i in range(len(neighbors)):
                state = G.nodes[i]['state']
                bfitness[i] = BF[state]
            total_bfitness = np.sum(bfitness)
            bprobabilities = bfitness / total_bfitness
            reproducing_node = np.random.choice(neighbors, p=bprobabilities)
            G.nodes[replacing_node]['state'] = G.nodes[reproducing_node]['state']
    elif model == 'increase':
        nodes = list(G.nodes())
        nonfinal_node = [node for node in nodes if G.nodes[node]['state'] != num_states - 1]
        random_index = random.randint(0, len(nonfinal_node)-1)
        random_node = nonfinal_node[random_index]
        G.nodes[random_node]['state'] += 1 
    else:
        raise ValueError("Invalid model type specified.")

def assign_colors_to_states(G):
    all_colors = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow', 
                  4: 'purple', 5: 'orange', 6: 'indigo', 7: 'black', 8: 'cyan', 9: 'violet', 10:'pink'}
    node_colors = []
    nodes = list(G.nodes())
    for node in nodes:
        node_colors.append(all_colors.get(G.nodes[node]['state']))   
    return node_colors

def plot_graph(G, pos, ax, step_count, model):
    """
    Plot the graph with node states.
    """
    colors = assign_colors_to_states(G)
    nx.draw(G, pos, node_color=colors, node_size=100, with_labels=False, edge_color='gray', ax=ax)
    ax.set_title(f'{model} process: {graph_type.capitalize()} Graph - Generation= {step_count}')


def animate(G, pos, ax1, ax2, history, model, num_muts):
    """
    # Generator function for the animation.
    """
    num_states = num_muts + 1
    n = G.number_of_nodes()
    step_count = 0
    num_of_state = np.zeros(num_states) # number of cells in state i from 0 to num_states-1
    curr = True
    while curr:
        step(G, model, num_muts)
        step_count += 1
        plot_graph(G, pos, ax1, step_count, model)
        for i in range(num_states):
            num_of_state[i] = list(nx.get_node_attributes(G, 'state').values()).count(i)
        # normal = list(nx.get_node_attributes(G, 'state').values()).count(0)
        # state_1 = list(nx.get_node_attributes(G, 'state').values()).count(1)
        # state_2 = list(nx.get_node_attributes(G, 'state').values()).count(2)
        # mutants = list(nx.get_node_attributes(G, 'state').values()).count(num_states-1)
        # proportion_mutants = mutants / G.number_of_nodes()
        proportion_mutants = num_of_state[num_states-1] / n
        history.append(proportion_mutants)
        
        ax2.clear()
        ax2.plot(history, color='red')
        ax2.set_ylim(0, 1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Proportion of Mutants')
        ax2.set_title('Proportion of Mutants Over Time')
    
        if any(count == n for count in num_of_state):
            curr = False

        
        yield

def prompt_user_for_parameters():
    """
    #Prompt the user to input parameters for the Moran process animation.
    """
    graph_type = input("Enter the graph type ('complete', 'star', 'line', 'circle', '2D_lattice', '3D_lattice'): ")
    n = int(input("Enter the size parameter 'n': "))
    initial_mutants = int(input("Enter the initial number of mutants: "))
    model = input("Enter the model type ('BD', 'DB'): ")
    

    center_mutant = False
    if graph_type == 'star':
        center_mutant_input = input("Should the center of the star graph be a mutant? (yes/no): ").strip().lower()
        center_mutant = center_mutant_input == 'yes'
    
    num_muts = int(input("Enter the number of mutations 'm': "))
    if num_muts > 9:
        print("Maximum number of mutations exceeded. Number of mutations has been set to 10.")
        num_muts = 9

    input_str = input("Enter the birth fitness vector for each mutation: ")
    stripped_str = input_str.strip('[] ')
    num_strs = stripped_str.split(',')
    bfitness_vector = [int(num_str) for num_str in num_strs]


    input_str2 = input("Enter the death fitness vector for each mutation: ")
    stripped_str2 = input_str2.strip('[] ')
    num_strs2 = stripped_str2.split(',')
    dfitness_vector = [int(num_str2) for num_str2 in num_strs2]
    return graph_type, n, initial_mutants, model, center_mutant, num_muts, bfitness_vector, dfitness_vector



def run_animation():

    # Initialize and run the Moran process
    G = initialize_graph(graph_type, n, initial_mutants, center_mutant)

    # Position the nodes for plotting
    if graph_type == '2D_lattice':
        pos = {node: (node % n, node // n) for node in G.nodes()}
    elif graph_type == '3D_lattice':
        pos = {node: (node[0] + node[1]*n, node[2]) for node in G.nodes()}
    else:
        pos = nx.spring_layout(G)

    history = [list(nx.get_node_attributes(G, 'state').values()).count(num_muts) / G.number_of_nodes()]
    print(history)


    # Create a figure for the animation with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12), gridspec_kw={'height_ratios': [2, 1]})

    # Plot the initial state of the graph
    plot_graph(G, pos, ax1, 0, model)

    # Create the generator
    animation_generator = animate(G, pos, ax1, ax2, history, model, num_muts)

    # Create the animation
    ani = animation.FuncAnimation(fig, lambda i: next(animation_generator), repeat=False, interval=100)

    # # Save the animation
    # ani.save('moran_process_animation.mp4', writer='ffmpeg', fps=80)

    plt.show()



def mut_type_average(trials,steps,graph_type, n, initial_mutants, model, num_muts):
    G = initialize_graph(graph_type, n, initial_mutants, num_muts, center_mutant)
    num_states = num_muts + 1
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
            step(G, model, num_muts)
            num_of_state = list(np.zeros(num_states)) # number of cells per state for all states
            for i in range(num_states):
                num_of_state[i] = list(nx.get_node_attributes(G, 'state').values()).count(permutations[i])
                proportion_mutants = num_of_state[i] / N
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

    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_history_circle1.npy', average_history)
    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_mut_type_circle1.npy', average_mut_type)
    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_z_vector_circle1.npy', average_z_vector)
    # np.save('/Users/kyletsai/Desktop/simulation_data/avg_init_time_circle1.npy', average_initiation_time)


    # Plot average_history
    plt.figure(figsize=(14, 7))
    color_bank = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'pink', 'black', 'cyan']
    for i in range(num_states):
        plt.plot(average_history[i],color=color_bank[i], label=f'State {i}')
    plt.title('Expected Proportion of Mutants Over Time')
    plt.xlabel('Time')
    plt.ylabel('Expected Proportion of Mutants')
    plt.legend()
    # plt.savefig('/Users/kyletsai/Desktop/simulation_data/avg_history_circle1.png')
    plt.show()

    # Plot average_mut_type
    plt.figure(figsize=(14, 7))
    for k in range(3):
        plt.plot(average_mut_type[k], label=f'Type {k+1}')
    plt.title('Expected Proportion of Types Over Time')
    plt.xlabel('Time')
    plt.ylabel('Expected Proportion of Types')
    plt.legend()
    # plt.savefig('/Users/kyletsai/Desktop/simulation_data/avg_mut_type_circle1.png')
    plt.show()
    
    # Plot average_z_vector
    plt.figure(figsize=(14, 7))
    plt.plot(average_z_vector, label='z_vector')
    #plt.title('Probability of Initiation Up to Time t')
    plt.xlabel('Time')
    plt.ylabel('Probability of Initiation Up to Time t')
    plt.legend()
    # plt.savefig('/Users/kyletsai/Desktop/simulation_data/avg_z_vector_circle1.png')
    plt.show()
    
    # Print average initiation time
    print(f'Average Initiation Time: {average_initiation_time}') 

    #return average_history, average_mut_type, average_z_vector, average_initiation_time
    return average_initiation_time

# Prompt user for parameters
# graph_type, n, initial_mutants, model, center_mutant, num_muts, bfitness_vector, dfitness_vector, mutation_weight = prompt_user_for_parameters()

# Initialize and run the Moran process
#G = initialize_graph(graph_type, n, initial_mutants, num_muts, center_mutant)
# Prompt user for parameters
graph_type, n, initial_mutants, model, center_mutant, num_muts, bfitness_vector, dfitness_vector = prompt_user_for_parameters()
run_animation()
#print(mut_type_average(100,2000,graph_type, n, initial_mutants, model, num_muts, bfitness_vector, dfitness_vector, mutation_weight))


