import numpy as np
import random

from PyNetwork.layers import Layer, Flatten, Input, BatchNorm
from PyNetwork import ErrorFunctions, MetricFunctions
from PyNetwork.optimizers import Optimizer, get_optimizer
from .utils.utils import buffer_str

import pyopencl.array as cl_array

class Sequential:
    """ Build a neural network sequentially by using the .add function

        Attributes
        ----------
        layers : dict of int - class
            Store the individual layers as a form of classes
    """

    def __init__(self):
        """ Initialise a sequential class
        """
        self.layers = {}

        self.loss_function = None
        self.metric_function = None
        self.optimizer_bias = None
        self.optimizer_weights = None

    def add(self, layer):
        """ Add a new layer

            Notes
            -----
            The layers are stacked in the order that they are added in, hence the name sequential

            Also, sequential models must always start with the `Input` class

            Parameters
            ----------
            layer : :obj:Layer
                A single layer in the network
        """

        if not isinstance(layer, Layer):
            raise TypeError('Only instances of Layer can be added to the sequential class.')

        self.layers[self.n + 1] = layer
        
    def build(self, context, queue, loss_function, optimizer, metrics=None):
        """ Once the model layers have been added this method must be called to initialise
            all the `sequential_layer` classes

            loss_function : str
                The loss function
            metrics : str, optional
                The name of the function to judge performance of the model
        """
        self.context = context
        self.queue = queue

        if isinstance(optimizer, Optimizer):
            self.optimizer_bias = optimizer
            self.optimizer_weights = self.optimizer_bias.new_instance()
        elif isinstance(optimizer, str):
            self.optimizer_bias = get_optimizer(optimizer)
            self.optimizer_weights = self.optimizer_bias.new_instance()
        else:
            raise ValueError("optimizer must be an instance of Optimizer or str")

        self.loss_function = loss_function
        self.metric_function = metrics

        previous_output_shape = None
        for i in range(1, self.n + 1):
            layer = self.layers[i]

            layer.build(self.context, self.queue, previous_output_shape)
            previous_output_shape = layer.output_shape

        self.gpu_metric = MetricFunctions(self.context, self.queue)
        
    def _validate_x_shape(self, x_shape, name):
        input_shape = self.layers[1].input_shape
        target_shape = (-1, *input_shape)
        
        assert len(x_shape) > 1, f"{name} has shape {x_shape} but {target_shape} was expected. Try {name}.reshape({target_shape})."
        assert input_shape == x_shape[1:], f"{name} has shape {x_shape} but {target_shape} was expected. Try {name}.reshape({target_shape})." 
                
    def _validate_y_shape(self, y_shape, name):
        output_shape = self.layers[self.n].output_shape
        target_shape = (-1, *output_shape)
        
        if len(y_shape) == 1: 
            assert len(output_shape == 1) and output_shape[0] == 1, f"{name} has shape {y_shape} but {target_shape} was expected. Try {name}.reshape({target_shape})."
        else:
            assert output_shape == y_shape[1:], f"{name} has shape {y_shape} but {target_shape} was expected. Try {name}.reshape({target_shape})."

    def predict(self, x_train_gpu, output_only=True):
        """ Forward propagate the input through the layers

            Parameters
            ----------
            x_train_gpu : pyopencl.array
                x_train_gpu is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the 0th index of x_train_gpu is the index that inputs is accessed by

            output_only : :obj:`bool`, optional
                If set to true, then this function will return only the prediction of the neural
                network. If set to false then this will return the outputs of the individual
                layers. Unless back propagation is being performed, this should be set to true.

            Returns
            -------
            pyopencl.array
                The final layer output of the neural netowork

            OR (if `output_only = False`)

            dict of int - pyopencl.array, dict of int - pyopencl.array
                The first dictionary will store outputs of all layers before it is passed through the activation
                function.
                The second dictionary will store the outputs of all layers after it has passed through the
                activation function.
        """
        self._validate_x_shape(x_train_gpu.shape, "x_train_gpu")

        if output_only:
            working_z = x_train_gpu
            for i in range(1, self.n + 1):  # Propagate working_Z throughout the layers
                working_z = self.layers[i].predict(working_z, output_only=True)
            return working_z
        else:
            a_dict = {0: None}
            z_dict = {0: x_train_gpu}

            for i in range(1, self.n + 1):
                current_layer = self.layers[i]
                a_dict[i], z_dict[i] = current_layer.predict(z_dict[i - 1], output_only=False,
                                                             pre_activation_of_input=a_dict[i - 1])

                i += 1
            return a_dict, z_dict

    def evaluate(self, x_test_gpu, y_test_gpu):
        """ Return the MSE of the model prediction

            Parameters
            ----------
            x_test_gpu : pyopencl.array
                x_test_gpu is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of x_test_gpu is the index that inputs is accessed by
            y_test_gpu : pyopencl.array
                y_test_gpu is the associated list of outputs to the list of inputs x_test_gpu.

            Returns
            -------
            str
                The error
        """
        self._validate_x_shape(x_test_gpu.shape, "x_test_gpu")
        self._validate_y_shape(y_test_gpu.shape, "y_test_gpu")
        
        prediction = self.predict(x_test_gpu)

        loss_val = self.loss(prediction, y_test_gpu)

        eval_str = f'{self.loss_function}: {format(loss_val.get(), ".4f")}'

        if self.metric_function is not None:
            metric_val = self.metric(prediction, y_test_gpu)
            eval_str += f' - {self.metric_function}: {format(metric_val.get(), ".4f")}'

        return eval_str

    def train(self, x_train_gpu, y_train_gpu, epochs=100, batch_size=None, verbose=True):
        """ Train the neural network by means of back propagation

            Parameters
            ----------
            x_train_gpu : pyopencl.array
                x_train_gpu is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of x_train_gpu is the index that inputs is accessed by
            y_train_gpu : pyopencl.array
                y_train_gpu is the associated list of outputs to the list of inputs x_train_gpu. More specifically,
                the neural network will be trained to find the map x_train_gpu -> y_train_gpu

            epochs : :obj:`int`, optional
                Number of times the neural network will see the entire dataset
            batch_size : :obj:`int`, optional
                The batch size for gradient descent. If not defined then `batch_size` is set to the
                length of the dataset, i.e. traditional gradient descent.
            verbose : bool, optional
                If set to `True` then the model performance will be printed after each epoch

        """
        self._validate_x_shape(x_train_gpu.shape, "x_train_gpu")
        self._validate_y_shape(y_train_gpu.shape, "y_train_gpu")

        training_length = len(x_train_gpu)
        if batch_size is None:
            batch_size = training_length
        index = np.arange(training_length)

        for _ in range(epochs):
            if verbose:
                print(f'Training on {len(x_train_gpu)} samples')

            random.shuffle(index)
            for i in range(np.ceil(training_length / batch_size).astype(int)):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_index = index[start:end]

                batch_x = cl_array.stack([x_train_gpu[j] for j in batch_index], axis=0)
                batch_y = cl_array.stack([y_train_gpu[j] for j in batch_index], axis=0)

                self._back_prop(batch_x, batch_y)

            if verbose:
                start, end = 0, batch_size
                batch_index = index[start:end]

                batch_x = cl_array.stack([x_train_gpu[j] for j in batch_index], axis=0)
                batch_y = cl_array.stack([y_train_gpu[j] for j in batch_index], axis=0)
                
                evaluation = self.evaluate(batch_x, batch_y)
                print(f'Epoch {_ + 1}/{epochs}')
                print(evaluation)

    def summary(self):
        """ Prints a summary of the layers in the `sequential` class
        """
        max_str_len = max([len(layer.summary_()[0]) for layer in self.layers.values()])

        for layer in self.layers.values():
            layer_type = buffer_str(layer.summary_()[0], max_buffer=max_str_len + 3)
            layer_out = layer.summary_()[1]
            print(f'{layer_type} :    {layer_out}')

    @property
    def loss(self):
        return ErrorFunctions.get_error_function(self.loss_function)

    @property
    def metric(self):
        if self.metric_function is not None:
            return self.gpu_metric.get_metric_function(self.metric_function)
        return None

    @property
    def n(self):
        return len(self.layers)

    def __len__(self):
        return self.n

    def _back_prop(self, x_train_gpu, y_train_gpu):
        """ Perform a single back propagation of the batch `x_train_gpu`, `y_train_gpu`.

            Parameters
            ----------
            x_train_gpu : pyopencl.array
                x_train_gpu is assumed to be a list of all the inputs to be forward propagated. In particular
                it is assumed that the first index of x_train_gpu is the index that inputs is accessed by
            y_train_gpu : pyopencl.array
                y_train_gpu is the associated list of outputs to the list of inputs x_train_gpu. More specifically,
                the neural network will be trained to find the map x_train_gpu -> y_train_gpu
        """

        # Forward propagate
        a_dict, z_dict = self.predict(x_train_gpu, output_only=False)
        delta_dict, grad_dict = {}, {}

        # Compute delta for the last layer
        delta_dict[self.n] = self.loss(z_dict[self.n], y_train_gpu, grad=True)  # Gradient of output
        if self.loss_function == 'cross_entropy':
            delta_dict[self.n] = z_dict[self.n] - y_train_gpu

        # Compute the weight gradients for the i-th layer and then compute delta_{i-1} for the
        # next layer in the network
        for i in range(self.n, 2, -1):  # i = 1 is the Input class, no backpropagation needed
            grad_dict[i] = self.layers[i].get_weight_grad_(delta_dict[i], z_dict[i - 1])
            g_prime = self.layers[i - 1].activation_function_(a_dict[i - 1], grad=True)

            if isinstance(self.layers[i], BatchNorm):
                delta_dict[i - 1] = self.layers[i].get_delta_backprop_(g_prime, delta_dict[i], z_dict[i - 1])
            else:
                delta_dict[i - 1] = self.layers[i].get_delta_backprop_(g_prime, delta_dict[i])

        # Use Optimizer to find the gradient update
        bias_grads = self.optimizer_bias.step({key: val[0] for (key, val) in grad_dict.items()})
        weights_grads = self.optimizer_weights.step({key: val[1] for (key, val) in grad_dict.items()})

        # Update layer parameters
        for i in range(self.n, 2, -1):
            current_layer = self.layers[i]
            current_layer.update_parameters_(bias_grads[i], weights_grads[i])
