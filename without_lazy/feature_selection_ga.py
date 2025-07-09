import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import random

class GeneticFeatureSelector:
    """
    Selects the best subset of features using a genetic algorithm.

    Args:
        model: A scikit-learn compatible estimator (classifier or regressor).
        X (pd.DataFrame): The training feature data.
        y (pd.Series): The training target data.
        n_generations (int): The number of generations to run the evolution.
        population_size (int): The number of individuals in each generation.
        crossover_rate (float): The probability of two parents performing crossover.
        mutation_rate (float): The probability of a gene (feature) mutating.
        tournament_size (int): The number of individuals to select for a tournament.
        scoring (str): The scoring metric for model evaluation (e.g., 'accuracy', 'r2').
        random_state (int, optional): Seed for reproducibility.
        cv (int): The number of cross-validation folds to use for fitness evaluation.
    """
    def __init__(self, model, X, y, n_generations=20, population_size=50, crossover_rate=0.8, mutation_rate=0.1, tournament_size=3, scoring='accuracy', cv=3, random_state=None):
        self.model = model
        self.X = X
        self.y = y
        self.n_generations = n_generations
        self.population_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.scoring = scoring
        self.cv = cv
        self.feature_names = X.columns.tolist()
        self.n_features = len(self.feature_names)
        self.random_state = random_state
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)
        
        self.population = []
        self.fitness_scores = []
        self.best_individual = None
        self.best_score = -np.inf

    def _initialize_population(self):
        """Creates the initial population of individuals (feature subsets)."""
        self.population = [np.random.randint(0, 2, self.n_features) for _ in range(self.population_size)]

    def _calculate_fitness(self, individual):
        """Calculates the fitness of an individual based on model performance and complexity."""
        num_features = np.sum(individual)
        if num_features == 0:
            return 0.0  # No features selected, score is 0

        selected_features = [feature for feature, bit in zip(self.feature_names, individual) if bit == 1]
        X_subset = self.X[selected_features]
        
        try:
            # Use cross-validation for a robust score
            score = np.mean(cross_val_score(clone(self.model), X_subset, self.y, cv=self.cv, scoring=self.scoring))
        except Exception:
            # Penalize if the model fails to train with the subset
            return 0.0

        # Penalty for complexity (more features) to encourage simpler models
        complexity_penalty = 0.001 * num_features
        return score - complexity_penalty

    def _select_parents(self):
        """Selects two parents from the population using tournament selection."""
        selection = []
        for _ in range(2): # Select two parents
            tournament_indices = np.random.choice(range(self.population_size), self.tournament_size, replace=False)
            tournament_fitness = [self.fitness_scores[i] for i in tournament_indices]
            winner_index = tournament_indices[np.argmax(tournament_fitness)]
            selection.append(self.population[winner_index])
        return selection[0], selection[1]

    def _crossover(self, parent1, parent2):
        """Performs single-point crossover between two parents."""
        if random.random() < self.crossover_rate:
            point = random.randint(1, self.n_features - 1)
            child1 = np.concatenate([parent1[:point], parent2[point:]])
            child2 = np.concatenate([parent2[:point], parent1[point:]])
            return child1, child2
        return parent1.copy(), parent2.copy()

    def _mutate(self, individual):
        """Performs bit-flip mutation on an individual."""
        for i in range(self.n_features):
            if random.random() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def run(self):
        """Runs the genetic algorithm evolution process."""
        self._initialize_population()

        for gen in range(self.n_generations):
            self.fitness_scores = [self._calculate_fitness(ind) for ind in self.population]

            # Update best individual found so far
            gen_best_idx = np.argmax(self.fitness_scores)
            if self.fitness_scores[gen_best_idx] > self.best_score:
                self.best_score = self.fitness_scores[gen_best_idx]
                self.best_individual = self.population[gen_best_idx].copy()

            new_population = [self.population[gen_best_idx]] # Elitism

            while len(new_population) < self.population_size:
                parent1, parent2 = self._select_parents()
                child1, child2 = self._crossover(parent1, parent2)
                new_population.append(self._mutate(child1))
                if len(new_population) < self.population_size:
                    new_population.append(self._mutate(child2))
            
            self.population = new_population[:self.population_size]
            print(f"  - Generation {gen + 1}/{self.n_generations} | Best Score: {self.best_score:.4f}", end='\r')
        
        print() # Newline after progress bar
        
        selected_features = [feature for feature, bit in zip(self.feature_names, self.best_individual) if bit == 1]
        return selected_features