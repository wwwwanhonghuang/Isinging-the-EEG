import numpy as np

class ConfigurationIterator:
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self.current_configuration = np.zeros(n_sites, dtype=int)
        self.current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_index >= (1 << self.n_sites):
            raise StopIteration
        if self.current_index == 0:
            # Return the initial configuration (all zeros)
            self.current_index += 1
            return self.current_configuration.copy()
        else:
            # Find the bit that changes between the current and previous index
            changed_bit = int(np.log2(self.current_index ^ (self.current_index - 1)))
            # Flip the bit in the configuration
            self.current_configuration[changed_bit] = 1 - self.current_configuration[changed_bit]
            self.current_index += 1
            return self.current_configuration.copy()
        
class PairwiseIsingModel(object):
    def __init__(self, n_sites):
        self.n_sites = n_sites
        self.J = np.zeros((n_sites, n_sites))  # Coupling matrix
        self.H = np.zeros(n_sites)  # External field
        
class PairwiseIsingModelInferencer:
    def __init__(self, ising_model: PairwiseIsingModel):
        self.Z = 0  # Partition function
        self.ising_model = ising_model
    
    def _check_partition_function(self):
        """Ensure the partition function is computed."""
        if self.Z == 0:
            self.update_partition_function()
    
    def update_partition_function(self):
        """Update the partition function Z by iteratively generating configurations using Gray codes."""
        self.Z = 0
        config_iterator = ConfigurationIterator(self.ising_model.n_sites)
        for configuration in config_iterator:
            self.Z += np.exp(-self.energy(configuration))
    
    def energy(self, configuration):
        """Compute the energy of a given configuration."""
        configuration = np.array(configuration)
        return -np.sum(configuration @ self.ising_model.J @ configuration.T) - np.dot(self.ising_model.H, configuration)

    def probability(self, configuration):
        """Compute the probability of a given configuration."""
        if self.Z == 0:
            print("Warning: Ising model's partition function Z == 0. The partition function may not have been initialized yet.")
            return 0
        return np.exp(-self.energy(configuration)) / self.Z

    def essembly_average_si(self):
        """Compute the ensemble average <s_i> for each spin."""
        self._check_partition_function()  # Ensure the partition function is computed
        n = self.ising_model.n_sites
        essembly_average_si = np.zeros(n)  # Initialize the result vector

        config_iterator = ConfigurationIterator(n)
        for configuration in config_iterator:
            prob = self.probability(configuration)
            essembly_average_si += prob * configuration

        return essembly_average_si

    def essembly_average_sisj(self):
        """Compute the ensemble average <s_i s_j> for all pairs of spins."""
        self._check_partition_function()  # Ensure the partition function is computed
        n = self.ising_model.n_sites
        essembly_average_sisj = np.zeros((n, n))  # Initialize the result matrix

        config_iterator = ConfigurationIterator(n)
        for configuration in config_iterator:
            prob = self.probability(configuration)
            essembly_average_sisj += prob * np.outer(configuration, configuration)

        return essembly_average_sisj