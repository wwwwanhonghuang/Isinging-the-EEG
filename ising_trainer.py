from ising_model import PairwiseIsingModelInferencer, PairwiseIsingModel
import numpy as np
class TrainingContext():
    def __init__(self, epoch = None, epochs = None, loss = None, model = None, l2_loss = None, kl_loss = None):
        self.epoch = epoch
        self.epochs = epochs
        self.loss = loss
        self.model = model
        self.kl_loss = kl_loss
        self.l2_loss = l2_loss

class PairwiseIsingModelTrainer:
    def __init__(self, ising_model: PairwiseIsingModel, inferencer: PairwiseIsingModelInferencer = None):
        self.ising_model = ising_model
        self.inferencer = PairwiseIsingModelInferencer(self.ising_model) if inferencer is None else inferencer


    def loss(self, observation_dataset, kl_weight = 1.0, lambda_weight = 1.0):
        """Compute the KL divergence loss and its gradients."""
        assert observation_dataset.shape[1] == self.ising_model.n_sites
        inferencer = self.inferencer

        n_samples = observation_dataset.shape[0]
        n_sites = self.ising_model.n_sites

        # Compute empirical averages
        essembly_average_obs_sisj = np.einsum('ki,kj->ij', observation_dataset, observation_dataset) / n_samples
        essembly_average_obs_si = np.mean(observation_dataset, axis=0)

        # Compute model averages using the inferencer
        essembly_average_model_sisj = inferencer.essembly_average_sisj()
        essembly_average_model_si = inferencer.essembly_average_si()

        # Compute the loss
        kl_loss = np.sum(essembly_average_obs_sisj * self.ising_model.J) + np.sum(essembly_average_obs_si * self.ising_model.H) + np.log(inferencer.Z)


         # Compute L2 loss on averages
        l2_loss_sisj = np.sum((essembly_average_obs_sisj - essembly_average_model_sisj) ** 2)
        l2_loss_si = np.sum((essembly_average_obs_si - essembly_average_model_si) ** 2)
        l2_loss = l2_loss_sisj + l2_loss_si

        # Compute gradients
        grad_J = kl_weight * (- essembly_average_obs_sisj + essembly_average_model_sisj) + 2 * lambda_weight * (essembly_average_model_sisj - essembly_average_obs_sisj)
        grad_H = kl_weight * (- essembly_average_obs_si + essembly_average_model_si) + 2 * lambda_weight * (essembly_average_model_si - essembly_average_obs_si)

        combined_loss =  kl_loss + lambda_weight * l2_loss

        return combined_loss, grad_J, grad_H, kl_loss, l2_loss

    def train(self, observation_dataset, epochs=5, learning_rate=0.01, epoch_callback=None):
        """Train the Ising model using gradient descent."""
        assert observation_dataset.shape[1] == self.ising_model.n_sites

        if self.inferencer is None:
            self.inferencer = PairwiseIsingModelInferencer(self.ising_model)
        

        for epoch in range(epochs):
            self.inferencer.update_partition_function()

            # Compute the loss and gradients
            loss, grad_J, grad_H, kl_loss, l2_loss = self.loss(observation_dataset)

            # Update the model parameters
            self.ising_model.J -= learning_rate * grad_J
            self.ising_model.H -= learning_rate * grad_H

            # Print the loss for monitoring
            if epoch_callback is not None:
                ctx = TrainingContext(epoch, epochs, loss=loss, model=self.ising_model, l2_loss=l2_loss, kl_loss=kl_loss)
                epoch_callback(ctx)
        return self.ising_model