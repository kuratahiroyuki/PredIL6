#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:32:39 2024

@author: user
"""

import numpy as np



import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fitness_function(weights, predictions_baseline, true_labels):
    # Normalize weights to sum to 1
    weights = softmax(weights) # / np.sum(weights)
    
    # Combine predictions using the normalized weights
    combined_predictions = np.dot(predictions_baseline, weights)
    
    # Clip combined predictions to ensure values are between 0 and 1
    combined_predictions = np.clip(combined_predictions, 0, 1)
    #combined_predictions = sigmoid(combined_predictions)
    # Calculate accuracy
    accuracy = np.mean((combined_predictions > 0.5) == true_labels)
    return accuracy

def genetic_algorithm_ensemble(dataset, num_generations=100, population_size=50, mutation_rate=0.01):
    num_samples, num_classifiers = dataset.shape[0], dataset.shape[1] - 1
    predictions_baseline = dataset[:, :-1]
    true_labels = dataset[:, -1].astype(bool)

    # Initialize population
    population = np.random.rand(population_size, num_classifiers)

    for generation in range(num_generations):
        # Evaluate fitness of the population
        fitness_scores = np.array([fitness_function(individual, predictions_baseline, true_labels) for individual in population])

        # Select the best individuals (elitism)
        best_indices = np.argsort(fitness_scores)[-2:]
        best_individuals = population[best_indices]

        # Generate new population through crossover and mutation
        new_population = []
        for _ in range(population_size):
            parent1, parent2 = best_individuals[np.random.choice(2, 2)]
            # Crossover
            crossover_point = np.random.randint(1, num_classifiers - 1)
            child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            # Mutation
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(num_classifiers)
                child[mutation_point] = np.random.rand()
            new_population.append(child)
        
        population = np.array(new_population)

    # Get the best individual from the final population
    final_fitness_scores = np.array([fitness_function(individual, predictions_baseline, true_labels) for individual in population])
    best_individual = population[np.argmax(final_fitness_scores)]

    # Generate final prediction scores
    best_weights = softmax(best_individual) #best_individual / np.sum(best_individual)
    final_scores = np.dot(predictions_baseline, best_weights)
    #final_scores_sigmoid = sigmoid(final_scores)
    #final_scores = np.clip(final_scores, 0, 1)

    return final_scores, best_weights
       
def predict_with_gae(test_dataset, best_weights):
    test_predictions = test_dataset[:, :-1]
    # Normalize weights (if needed)
    best_weights = best_weights # / np.sum(best_weights)
    
    # Generate final prediction scores for the test set
    final_scores_test = np.dot(test_predictions, best_weights)
    final_scores_test_sigmoid = sigmoid(final_scores_test)
    return final_scores_test

# Example usage
#dataset = np.random.rand(100, 101)  # Dummy dataset with 100 samples, 100 classifiers' predictions + 1 true label column
#final_scores = genetic_algorithm_ensemble(dataset)
#print(final_scores)


