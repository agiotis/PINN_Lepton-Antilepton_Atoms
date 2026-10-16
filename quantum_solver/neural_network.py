# neural_network.py

import numpy as np


# import torch


# class RadialNN(torch.nn.Module):
#     def __init__(self, units, activation):
#         super().__init__()
#         self.fc1 = torch.nn.Linear(1, units)
#         self.fc2 = torch.nn.Linear(units, 1)
#         self.activation = activation
#
#     def forward(self, x):
#         x = self.activation(self.fc1(x))
#         x = self.fc2(x)
#         return x


class NeuralNetwork:
    def __init__(self):
        pass

    # in following fundamental math methods such as exp() and log, they have been replaced by np.exp and np.log, since
    # they seem to handle better cases where math range error arises from numerical overflow.
    @staticmethod
    # ΑΡΧΙΚΗ ΠΡΟΣΕΓΓΙΣΗ
    # def sigmoid(s: float) -> float:
    #     """Sigmoid function 1/(1 + e^-s)"""
    #     return 1 / (1 + np.exp(-s))

    # @staticmethod
    # def sigmoid(s: float) -> float:
    #     """Sigmoid Activation Function with Clipping"""
    #     s = np.clip(s, -30, 30)  # Περιορισμός τιμών για αποφυγή υπερχείλισης
    #     return 1 / (1 + np.exp(-s))
    #
    # @staticmethod
    # def firdersigmoid(s: float) -> float:
    #     """"First derivative of the sigmoid function"""
    #     return NeuralNetwork.sigmoid(s) * (1 - NeuralNetwork.sigmoid(s))
    #
    # @staticmethod
    # def secdersigmoid(s: float) -> float:
    #     """Second derivative of the sigmoid function"""
    #     return NeuralNetwork.sigmoid(s) * (1 - NeuralNetwork.sigmoid(s)) * (1 - 2 * NeuralNetwork.sigmoid(s))
    #
    # @staticmethod
    # def thirdersigmoid(s: float) -> float:
    #     """Third derivative of the sigmoid function"""
    #     term1 = (NeuralNetwork.sigmoid(s) * np.power(1 - NeuralNetwork.sigmoid(s), 2)
    #              * (1 - 2 * NeuralNetwork.sigmoid(s)))
    #     term2 = (- np.power(NeuralNetwork.sigmoid(s), 2) * (1 - NeuralNetwork.sigmoid(s))
    #              * (1 - 2 * NeuralNetwork.sigmoid(s)))
    #     term3 = - 2 * np.power(NeuralNetwork.sigmoid(s), 2) * np.power(1 - NeuralNetwork.sigmoid(s), 2)
    #     return term1 + term2 + term3
    #
    # @staticmethod
    # def fourdersigmoid(s: float) -> float:
    #     """Fourth derivative of the sigmoid function"""
    #     term11 = (NeuralNetwork.sigmoid(s) * np.power(1 - NeuralNetwork.sigmoid(s), 3)
    #               * (1 - 2 * NeuralNetwork.sigmoid(s)))
    #     term12 = (-2 * np.power(NeuralNetwork.sigmoid(s), 2) * np.power(1 - NeuralNetwork.sigmoid(s), 3)
    #               * (1 - 2 * NeuralNetwork.sigmoid(s)))
    #     term13 = -2 * np.power(NeuralNetwork.sigmoid(s), 2) * np.power(1 - NeuralNetwork.sigmoid(s), 3)
    #
    #     term21 = (- 2 * np.power(NeuralNetwork.sigmoid(s), 2) * np.power(1 - NeuralNetwork.sigmoid(s), 2)
    #               * (1 - 2 * NeuralNetwork.sigmoid(s)))
    #     term22 = (np.power(NeuralNetwork.sigmoid(s), 3) * np.power(1 - NeuralNetwork.sigmoid(s), 2)
    #               * (1 - 2 * NeuralNetwork.sigmoid(s)))
    #     term23 = 2 * np.power(NeuralNetwork.sigmoid(s), 3) * np.power(1 - NeuralNetwork.sigmoid(s), 2)
    #
    #     term31 = - 4 * np.power(NeuralNetwork.sigmoid(s), 2) * np.power(1 - NeuralNetwork.sigmoid(s), 3)
    #     term32 = 4 * np.power(NeuralNetwork.sigmoid(s), 3) * np.power(1 - NeuralNetwork.sigmoid(s), 2)
    #     return term11 + term12 + term13 + term21 + term22 + term23 + term31 + term32

    # ORIGINAL IMPLEMENTATION
    # @staticmethod
    def sigmoid(s: float) -> float:
        """Sigmoid function."""
        return 1 / (1 + np.exp(-s))

    @staticmethod
    # def sigmoid(s: float) -> float:
    #     """Stable sigmoid function."""
    #     if s >= 0:
    #         z = np.exp(-s)
    #         return 1 / (1 + z)
    #     else:
    #         z = np.exp(s)
    #         return z / (1 + z)

    @staticmethod
    def firdersigmoid(s: float) -> float:
        """First derivative of the sigmoid function."""
        sig = NeuralNetwork.sigmoid(s)
        return sig * (1 - sig)

    @staticmethod
    def secdersigmoid(s: float) -> float:
        """Second derivative of the sigmoid function."""
        sig = NeuralNetwork.sigmoid(s)
        sig1 = 1 - sig
        return sig * sig1 * (1 - 2 * sig)

    @staticmethod
    def thirdersigmoid(s: float) -> float:
        """Third derivative of the sigmoid function."""
        sig = NeuralNetwork.sigmoid(s)
        return sig * (1 - sig) * (1 - 6 * sig + 6 * sig ** 2)

    @staticmethod
    def fourdersigmoid(s: float) -> float:
        """Fourth derivative of the sigmoid function."""
        sig = NeuralNetwork.sigmoid(s)
        return sig * (1 - sig) * (1 - 14 * sig + 36 * sig ** 2 - 24 * sig ** 3)

    # Original changed in 19/12/2025
    # @staticmethod
    # def relu(s: float):
    #     """Relu activation function"""
    #     return np.max([0, s])

    @staticmethod
    def relu(s):
        return np.maximum(0, s)

    @staticmethod
    def firderrelu(s: float):
        """First derivative of Relu activation function"""
        if s > 0:
            return 1
        else:
            return 0

    @staticmethod
    def secderrelu(s: float) -> float:
        """Second derivative of Relu activation function"""
        return 0

    @staticmethod
    def thirderrelu(s: float) -> float:
        """Third derivative of Relu activation function"""
        return 0

    @staticmethod
    def fourderrelu(s: float) -> float:
        """Fourth derivative of Relu activation function"""
        return 0

    @staticmethod
    def smplus(s: float) -> float:
        """Smooth plus activation function"""
        return np.log(1 + np.exp(s))

    @staticmethod
    def firdersmplus(s: float) -> float:
        """First derivative of Smooth plus activation function"""
        return NeuralNetwork.sigmoid(s)

    @staticmethod
    def secdersmplus(s: float) -> float:
        """Second derivative of Smooth plus activation function"""
        return NeuralNetwork.firdersigmoid(s)

    @staticmethod
    def thirdersmplus(s: float) -> float:
        """Third derivative of Smooth plus activation function"""
        return NeuralNetwork.secdersigmoid(s)

    @staticmethod
    def fourdersmplus(s: float) -> float:
        """Fourth derivative of Smooth plus activation function"""
        return NeuralNetwork.thirdersigmoid(s)

    # @staticmethod  (original)
    # def firdertanh(s: float) -> float:
    #     """First Derivative of Hyperbolic Tangent"""
    #     return 1 / np.power(np.cosh(s), 2)
    # @staticmethod
    # def firdertanh(s: float) -> float:
    #     if abs(s) > 20:
    #         return 0.0  # Approaches zero for large |s|
    #     return 1 / np.power(np.cosh(s), 2)
    #
    # @staticmethod
    # def secdertanh(s: float) -> float:
    #     """Second Derivative of Hyperbolic Tangent"""
    #     return -2 * np.sinh(s) / np.power(np.cosh(s), 2)
    #
    # @staticmethod
    # def thirdertanh(s: float) -> float:
    #     return - 2 / np.cosh(s) + 4 * np.power(np.sinh(s), 2) / np.power(np.cosh(s), 4)
    #
    # @staticmethod
    # def fourdertanh(s: float) -> float:
    #     return 2 * np.sinh(s) / np.power(np.cosh(s), 2) + 4

    @staticmethod
    def tanh(s: float) -> float:
        """Hyperbolic Tangent Function"""
        return np.tanh(s)

    @staticmethod
    def firdertanh(s: float) -> float:
        """First Derivative of Hyperbolic Tangent (optimized)"""
        tanh_val = np.tanh(s)
        return 1 - tanh_val * tanh_val  # Efficient and numerically stable

    @staticmethod
    def secdertanh(s: float) -> float:
        """Second Derivative of Hyperbolic Tangent"""
        tanh_val = np.tanh(s)
        tanh_squared = tanh_val ** 2
        one_minus_tanh_squared = 1 - tanh_squared
        return -2 * tanh_val * one_minus_tanh_squared

    @staticmethod
    def thirdertanh(s: float) -> float:
        """Third derivative of the hyperbolic tangent."""
        tanh_val = np.tanh(s)
        return 2 * (1 - tanh_val ** 2) * (3 * tanh_val ** 2 - 1)

    @staticmethod
    def fourdertanh(s: float) -> float:
        """Fourth derivative of the hyperbolic tangent."""
        tanh_val = np.tanh(s)
        return 8 * tanh_val * (1 - tanh_val ** 2) * (2 - 3 * tanh_val ** 2)

    @staticmethod
    def activfunc(act_function, deg: int) -> callable:
        if act_function == 'sigmoid':
            return [NeuralNetwork.sigmoid, NeuralNetwork.firdersigmoid, NeuralNetwork.secdersigmoid,
                    NeuralNetwork.thirdersigmoid, NeuralNetwork.fourdersigmoid][deg]
        elif act_function == 'relu':
            return [NeuralNetwork.relu, NeuralNetwork.firderrelu, NeuralNetwork.secderrelu,
                    NeuralNetwork.thirderrelu, NeuralNetwork.fourderrelu][deg]
        elif act_function == 'smplus':
            return [NeuralNetwork.smplus, NeuralNetwork.firdersmplus, NeuralNetwork.secdersmplus,
                    NeuralNetwork.thirdersmplus, NeuralNetwork.fourdersmplus][deg]
        elif act_function == 'tanh':
            return [np.tanh, NeuralNetwork.firdertanh, NeuralNetwork.secdertanh, NeuralNetwork.thirdertanh,
                    NeuralNetwork.fourdertanh][deg]
        else:
            raise print('no activation function set! Please enter sigmoid or relu or smplus')
    # ORIGINAL
    # @staticmethod
    # def neuralfunc(r: float, u: list, actfunc: callable, ite: int) -> float:
    #     """Trial function"""
    #     nfc0 = 0
    #     for i in range(ite):
    #         nfc0 += u[i] * actfunc(u[i + ite] * r + u[i + 2 * ite])
    #         #  nfc0 += actfunc(u[i] * r + u[ite])
    #     return nfc0
    @staticmethod
    def neuralfunc(r: float, u: list, actfunc: callable, ite: int) -> float:
        """Vectorized neural function"""
        u = np.array(u)
        indices = np.arange(ite)
        return np.sum(u[indices] * actfunc(u[indices + ite] * r + u[indices + 2 * ite]))

    # ORIGINAL
    # @staticmethod
    # def firderneuralfunc(r: float, u: list, firderactfunc: callable, ite: int) -> float:
    #     """First derivative of the neural function"""
    #     fnfc0 = 0
    #     for i in range(ite):
    #         coeff = u[i] * u[i + ite]
    #         fnfc0 += coeff * firderactfunc(u[i + ite] * r + u[i + 2 * ite])
    #         #  fnfc0 += u[i] * firderactfunc(u[i] * r + u[ite])
    #     return fnfc0
    @staticmethod
    def firderneuralfunc(r: float, u: list, firderactfunc: callable, ite: int) -> float:
        """First derivative of the neural function (optimized)"""
        fnfc0 = 0
        for i in range(ite):
            s = u[i + ite] * r + u[i + 2 * ite]  # Προαποθήκευση της συνάρτησης εισόδου
            fnfc0 += u[i] * u[i + ite] * firderactfunc(s)
        return fnfc0

    @staticmethod
    def secderneuralfunc(r: float, u: list, secderactfunc: callable, ite: int) -> float:
        """Second derivative of the neural function"""
        snfc0 = 0
        for i in range(ite):
            # maybe np.power(u[i+ite],2) ??
            coeff = u[i] * u[i + ite] * u[i + ite]
            #  coeff = np.power(u[i],2)
            snfc0 += coeff * secderactfunc(u[ite + i] * r + u[i + 2 * ite])
        return snfc0

    @staticmethod
    def thirderneuralfunc(r: float, u: list, thirderactfunc: callable, ite: int) -> float:
        """Third derivative of the neural function"""
        tnfc0 = 0
        for i in range(ite):
            coeff = u[i] * np.power(u[i + ite], 3)
            tnfc0 += coeff * thirderactfunc(u[ite + i] * r + u[i + 2 * ite])
        return tnfc0

    @staticmethod
    def fourderneuralfunc(r: float, u: list, fourderactfunc: callable, ite: int) -> float:
        """Fourth derivative of the neural function"""
        fonfc0 = 0
        for i in range(ite):
            coeff = u[i] * np.power(u[i + ite], 4)
            fonfc0 += coeff * fourderactfunc(u[ite + i] * r + u[i + 2 * ite])
        return fonfc0

    # @staticmethod
    # def arfunc(r: float, u: list, kpam: float, ite: int, act_function) -> float:
    #     """Numerical solution (for suitable u)"""
    #     return NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite) * np.exp(-kpam * r) * r
    @staticmethod
    def arfunc(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """Numerical solution (optimized)"""
        exp_term = np.exp(-kpam * r)
        return NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite) * exp_term * r

    @staticmethod
    def arfuncDirac(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """Numerical solution for Dirac"""
        exp_term = np.exp(-kpam * r)
        return NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite) * exp_term

    # @staticmethod
    # def firderarfunc(r: float, u: list, kpam: float, ite: int, act_function) -> float:
    #     """First derivative of the numerical solution"""
    #     f1p = (NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite)
    #            * np.exp(-kpam * r) * r)
    #     f2p = (-kpam * NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite)
    #            * np.exp(-kpam * r) * r)
    #     f3p = NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite) * np.exp(-kpam * r)
    #     return f1p + f2p + f3p
    @staticmethod
    def firderarfunc(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """First derivative of the numerical solution (optimized)"""
        exp_term = np.exp(-kpam * r)
        neural_value = NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite)
        firder_neural = NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite)

        f1p = firder_neural * exp_term * r
        f2p = -kpam * neural_value * exp_term * r
        f3p = neural_value * exp_term

        return f1p + f2p + f3p

    @staticmethod
    def firderarfuncDirac(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """First derivative of the numerical solution for Dirac"""
        # Πρώτος όρος
        f1p = -kpam * NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite) * np.exp(-kpam * r)

        # Δεύτερος όρος
        f2p = np.exp(-kpam * r) * NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite)

        return f1p + f2p

    @staticmethod
    def secderarfunc(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """Second derivative of the numerical solution"""
        sec1p = (r * np.exp(-kpam * r) *
                 NeuralNetwork.secderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 2), ite))
        sec2p = (2 * np.exp(-kpam * r) *
                 NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite)
                 * (1 - kpam * r))
        sec3p = (kpam * np.exp(-kpam * r) *
                 NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite) * (kpam * r - 2))
        return sec1p + sec2p + sec3p

    @staticmethod
    def secderarfuncDirac(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """Second derivative of the numerical solution for Dirac"""
        # Πρώτος όρος
        sec1p = (kpam ** 2 * np.exp(-kpam * r) *
                 NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite)) - \
                (kpam * np.exp(-kpam * r) *
                 NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite))

        # Δεύτερος όρος
        sec2p = (-kpam * np.exp(-kpam * r) *
                 NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite)) + \
                (np.exp(-kpam * r) *
                 NeuralNetwork.secderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 2), ite))

        return sec1p + sec2p

    @staticmethod
    def thirderarfunc(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """Third derivative of U(r) = r exp(-kpam r) N(r)."""
        n0 = NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite)
        n1 = NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite)
        n2 = NeuralNetwork.secderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 2), ite)
        n3 = NeuralNetwork.thirderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 3), ite)
        return np.exp(-kpam * r) * (
            r * n3 + (3 - 3 * kpam * r) * n2
            + (3 * kpam ** 2 * r - 6 * kpam) * n1
            + (3 * kpam ** 2 - kpam ** 3 * r) * n0
        )

    @staticmethod
    def fourderarfunc(r: float, u: list, kpam: float, ite: int, act_function) -> float:
        """Fourth derivative of U(r) = r exp(-kpam r) N(r)."""
        n0 = NeuralNetwork.neuralfunc(r, u, NeuralNetwork.activfunc(act_function, 0), ite)
        n1 = NeuralNetwork.firderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 1), ite)
        n2 = NeuralNetwork.secderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 2), ite)
        n3 = NeuralNetwork.thirderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 3), ite)
        n4 = NeuralNetwork.fourderneuralfunc(r, u, NeuralNetwork.activfunc(act_function, 4), ite)
        return np.exp(-kpam * r) * (
            r * n4 + (4 - 4 * kpam * r) * n3
            + (6 * kpam ** 2 * r - 12 * kpam) * n2
            + (12 * kpam ** 2 - 4 * kpam ** 3 * r) * n1
            + (kpam ** 4 * r - 4 * kpam ** 3) * n0
        )

    @staticmethod
    def random_initial_value_u0(num_of_hidden_units, f_g_model):
        # assign (30 * f_g_model + f_g_model) random values in [0..1] to model Dirac's f and g components.
        # 30 + f_g_model/2 = 31 numbers for each decade of values for u, v, w: f, plus 1 value for coefficient c_f
        #  30 + f_g_model/2 = 31 numbers for each decade of values for u, v, w: g, plus 1 value for coefficient c_g
        if f_g_model != 0:
            u0_random = np.random.rand(3 * f_g_model * num_of_hidden_units + f_g_model)
            # Here we assign default value = 1 for cf coefficient (needs testing when learned not fixed...)
            # u0_random[3 * num_of_hidden_units] = 1.0
            # Here we assign default value = 1 for cg coefficient (needs testing when learned not fixed...)
            # u0_random[-1] = 1.0
        else:
            # assign 30 random values in [0 1] corresponding to each decade of u,v,w to model Schrödinger's f component.
            u0_random = np.random.rand(3 * num_of_hidden_units)
        #  u0_random = np.random.rand(num_of_hidden_units+1)  # 10 weights (1 per hidden unit) + 1 bias term
        # Normalize random values so that they are comparable with each other
        u0_random = u0_random / np.max(np.abs(u0_random))

        return u0_random

    @staticmethod
    def fixed_initial_value_u0(num_of_hidden_units, f_g_model):
        # assign 30 fixed values in [u=1.2, v=0.8, w=1] for each decade of values for u, v, w, plus 1 for c_f and
        # another 30 fixed values in [u=1.2, v=0.8, w=1] for each decade of values for u, v, w, plus 1 for c_g
        ones = np.ones(num_of_hidden_units)
        if f_g_model != 0:
            u0_fixed = np.concatenate([
                1.2 * ones,  # [1.2] * num_of_hidden_units  ---> u
                0.8 * ones,  # [0.8] * num_of_hidden_units  ---> v
                ones,  # [1] * (num_of_hidden_units + 1)    ---> w + c_f
                1.2 * ones,  # [1.2] * num_of_hidden_units  ---> u
                0.8 * ones,  # [0.8] * num_of_hidden_units  ---> v
                ones,  # [1] * (num_of_hidden_units + 1)    ---> w + c_g
            ])
        else:
            u0_fixed = np.concatenate([1.2 * ones, 0.8 * ones, ones])
        return u0_fixed

    @staticmethod
    def bounded_initial_value_u0(num_of_hidden_units, f_g_model):
        # assign i.e. 10 random values in [-10, 10], for u
        # [-9, 9] for v and [-8.9 8.9] for w + 1 for f and g, respectively.
        if f_g_model != 0:
            u0_bounded_f = np.concatenate([np.array(np.random.uniform(-10, 10, num_of_hidden_units)),
                                           np.array(np.random.uniform(-9, 9, num_of_hidden_units)),
                                           np.array(np.random.uniform(-8.9, 8.9, num_of_hidden_units + 1))])
            u0_bounded_g = np.concatenate([np.array(np.random.uniform(-10, 10, num_of_hidden_units)),
                                           np.array(np.random.uniform(-9, 9, num_of_hidden_units)),
                                           np.array(np.random.uniform(-8.9, 8.9, num_of_hidden_units + 1))])
            u0_bounded = np.concatenate([u0_bounded_f, u0_bounded_g])
        else:
            u0_bounded = np.concatenate([np.array(np.random.uniform(-10, 10, num_of_hidden_units)),
                                         np.array(np.random.uniform(-9, 9, num_of_hidden_units)),
                                         np.array(np.random.uniform(-8.9, 8.9, num_of_hidden_units))])
        return u0_bounded

    @staticmethod
    def xavier_init_u0(size):
        # Initialize parameters for a 2-layer network with 30 input units [u v w], 10 hidden units, and 1 output unit
        # u_xavier = xavier_init_u0((30, 10))  # Parameters for the input to hidden layer
        # v_xavier = xavier_init_u0((10, 1))   # Parameters for the hidden to output layer
        in_dim, out_dim = size
        xavier_stddev = 1.0 / np.sqrt(in_dim + out_dim)

        #   return np.random.normal(0, xavier_stddev, size)
        return np.abs(np.random.normal(0, xavier_stddev, size))

    @staticmethod
    def manual_initial_value_u0(u0_manual, num_of_hidden_units, f_g_model):
        """
        Use explicitly provided u0 vector (cardinal / fixed for debugging).

        Parameters
        ----------
        u0_manual : array-like
            Explicit initial vector u0.
        num_of_hidden_units : int
            Number of hidden units.
        f_g_model : int
            0 for Schrödinger, !=0 for Dirac (f+g).
        """
        u0_manual = np.asarray(u0_manual, dtype=float)

        # Expected size check (fail fast)
        if f_g_model != 0:
            expected_size = 3 * f_g_model * num_of_hidden_units + f_g_model
        else:
            expected_size = 3 * num_of_hidden_units

        if u0_manual.size != expected_size:
            raise ValueError(
                f"[manual_initial_value_u0] Wrong u0 size: "
                f"got {u0_manual.size}, expected {expected_size}"
            )

        # Normalize exactly like random init (keeps scale comparable)
        u0_manual = u0_manual / np.max(np.abs(u0_manual))

        return u0_manual

