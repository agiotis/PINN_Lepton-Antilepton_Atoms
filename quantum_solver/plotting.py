# plotting.py

import os
import csv
import numpy as np

from matplotlib import pyplot as plt


class PlottingModule:
    def __init__(self, partl, eq, verbose=False, index=None):
        self.partl = partl
        self.eq = eq
        self.verbose = verbose
        self.experiment_idx = index

    def create_folders(self):
        """Creates respected folders if they don't exist"""
        path = f'./data/{self.partl}-{self.eq}/'
        if not os.path.exists(path):
            os.makedirs(path)
        return path


class SchrodingerPlotter(PlottingModule):
    def __init__(self, partl, nq, lq, alpha0, verbose, index):
        super().__init__(partl, 'Sch')
        self.nq = nq
        self.lq = lq
        self.alpha0 = alpha0
        self.verbose = verbose
        self.experiment_idx = index

    def plot(self, func_analytical, func_numerical, anum, lb, rb, ipoints, overlay=False, idx=None):
        """Exporting function values to .csv and .txt formats"""
        path = self.create_folders()
        lbound = lb if lb < rb else 1e-5 * self.alpha0
        rbound = rb if lb < rb else 8 * self.alpha0
        steps = ipoints if ipoints > 0 else 1000
        xran = np.linspace(lbound, rbound, steps)

        # ---to-do----
        # extra arg verbose = true then show the plot else only save the plot (used for batch experiments)
        # Define the axis labels using LaTeX notation
        x_label = r'$r / \alpha_0$'
        y_label = r'$R_{%d%d} \, \alpha_0^{3/2}$' % (self.nq, self.lq)

        if not overlay:
            plt.figure()  # Create a figure for analytical plot

            # Plotting
            plt.plot(xran, [func_analytical(r) for r in xran],
                     label=f'Analytical Function for {anum}n={self.nq}, l={self.lq}', linestyle='-')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/analytical_plot.pdf')  # Save the figure for analytical solution plot
            if self.verbose:
                plt.show()  # Display the plot for non-overlayed plots

            plt.figure()  # Create a new figure for non-overlayed plots
            plt.plot(xran, [func_numerical(r) for r in xran],
                     label=f'Numerical Function for {anum}n={self.nq}, l={self.lq}', linestyle='--')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/numerical_plot.pdf')  # Save the figure for numerical solution plot
            if self.verbose:
                plt.show()  # Display the plot for non-overlayed plots
        else:
            plt.figure()  # Create a figure for analytical plot
            threshold = 3e-07
            # Plotting
            # Calculate the analytical and numerical values
            analytical_values = [func_analytical(r) * pow(self.alpha0, 3 / 2) for r in xran]
            numerical_values = [func_numerical(r) * pow(self.alpha0, 3 / 2) for r in xran]

            # Check the difference between the first numerical value and the corresponding analytical value
            if abs(numerical_values[0] - analytical_values[0]) > threshold:
                # If the difference is greater than a certain threshold, ignore the first point
                xran = xran[1:]
                analytical_values = analytical_values[1:]
                numerical_values = numerical_values[1:]

            # Plot the values
            plt.plot(xran / self.alpha0, analytical_values,
                     label=f'Analytical Function for {anum}n={self.nq}, l={self.lq}', linestyle='-')
            plt.plot(xran / self.alpha0, numerical_values,
                     label=f'Numerical Function for {anum}n={self.nq}, l={self.lq}', linestyle='--')

            # plt.plot(xran / self.alpha0, [func_analytical(r) * pow(self.alpha0, 3 / 2) for r in xran],
            #          label=f'Analytical Function for {anum}n={self.nq}, l={self.lq}', linestyle='-')
            # plt.plot(xran / self.alpha0, [func_numerical(r) * pow(self.alpha0, 3 / 2) for r in xran],
            #          label=f'Numerical Function for {anum}n={self.nq}, l={self.lq}', linestyle='--')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/Experiment_{idx}_overlayed_plot.pdf')  # Save the figure for overlayed plots
            if self.verbose:
                plt.show()  # Display the plot for overlayed plots

        # .csv format
        with open(f'{path}/Experiment_{idx}_{self.nq}_{self.lq}.csv', 'w', newline='') as file_csv:
            writer_csv = csv.writer(file_csv)
            for r in xran:
                rnl_analytical = func_analytical(r)
                rnl_numerical = func_numerical(r)
                writer_csv.writerow([r / self.alpha0, rnl_analytical * pow(self.alpha0, 3 / 2),
                                     rnl_numerical * pow(self.alpha0, 3 / 2)])

        # .txt format
        with open(f'{path}/Experiment_{idx}_{self.nq}_{self.lq}.txt', 'w', newline='') as file_txt:
            for r in xran:
                rnl_analytical = func_analytical(r)
                rnl_numerical = func_numerical(r)
                file_txt.write(
                    f"{r / self.alpha0:10f} {rnl_analytical * pow(self.alpha0, 3 / 2)} {rnl_numerical * pow(self.alpha0, 3 / 2)}\n")


class DiracPlotter(PlottingModule):
    def __init__(self, partl, nq, lq, jq, verbose, idx=None):
        super().__init__(partl, 'DC')
        self.nq = nq
        self.lq = lq
        self.jq = jq
        self.verbose = verbose
        self.experiment_idx = idx

    def plot(self, analytical_F, numerical_F, analytical_G, numerical_G, anum, nq, lq, jq, lbound=1e-2, rbound=5e5,
             ipoints=900, overlay=False, idx=None, canonical_numerical_data=None, dirac_coefficient_mode=None,
             dirac_initialization_method=None):
        """Plot F and G analytical and numerical functions separately or overlayed if specified."""
        path = self.create_folders()
        if canonical_numerical_data is None:
            xrange = np.linspace(lbound, rbound, ipoints)
            canonical_F = None
            canonical_G = None
        else:
            canonical_numerical_data = np.asarray(canonical_numerical_data, dtype=float)
            if canonical_numerical_data.ndim != 2 or canonical_numerical_data.shape[1] != 3:
                raise ValueError("canonical_numerical_data must have columns r,F,G")
            xrange = canonical_numerical_data[:, 0]
            canonical_F = canonical_numerical_data[:, 1]
            canonical_G = canonical_numerical_data[:, 2]
        xran_F = xrange
        xran_G = xrange
        if not overlay:
            # Create separate plots for F analytical and numerical
            plt.figure()
            plt.plot(xrange, [analytical_F(r) for r in xrange],
                     label=f'F analytical for {anum}n={nq}, l={lq}, j={jq}')
            plt.xlabel('r')
            plt.ylabel('Function Value')
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/F_analytical_{nq}_{lq}_{jq}_plot.pdf')
            if self.verbose:
                plt.show()

            plt.figure()
            plt.plot(xrange, canonical_F if canonical_F is not None else [numerical_F(r) for r in xrange],
                     label=f'F numerical for {anum}n={nq}, l={lq}, j={jq}', linestyle='dashed')
            plt.xlabel('r')
            plt.ylabel('Function Value')
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/F_numerical_{nq}_{lq}_{jq}_plot.pdf')
            if self.verbose:
                plt.show()

            # Create separate plots for G analytical and numerical
            plt.figure()
            plt.plot(xrange, [analytical_G(r) for r in xrange],
                     label=f'G analytical for {anum}n={nq}, l={lq}, j={jq}')
            plt.xlabel('r')
            plt.ylabel('Function Value')
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/G_analytical_{nq}_{lq}_{jq}_plot.pdf')
            if self.verbose:
                plt.show()

            plt.figure()
            plt.plot(xrange, canonical_G if canonical_G is not None else [numerical_G(r) for r in xrange],
                     label=f'G numerical for {anum}n={nq}, l={lq}, j={jq}', linestyle='dashed')
            plt.xlabel('r')
            plt.ylabel('Function Value')
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/G_numerical_{nq}_{lq}_{jq}_plot.pdf')
            if self.verbose:
                plt.show()
        else:
            # Define dynamic thresholds based on the analytical values
            scale_F = 0.001  # Adjust for F threshold scaling
            scale_G = 0.01  # Adjust for G threshold scaling (higher for G)

            # Generate analytical and numerical values for F and G
            analytical_values_F = [analytical_F(r) for r in xrange]
            numerical_values_F = list(canonical_F) if canonical_F is not None else [numerical_F(r) for r in xrange]
            analytical_values_G = [analytical_G(r) for r in xrange]
            numerical_values_G = list(canonical_G) if canonical_G is not None else [numerical_G(r) for r in xrange]

            # threshold_F = scale_F * max(abs(v) for v in analytical_values_F)
            # threshold_G = scale_G * max(abs(v) for v in analytical_values_G)
            threshold_F = np.max(np.abs(analytical_values_F)) * 1e-3
            threshold_G = np.max(np.abs(analytical_values_G)) * 1e-2

            # Initialize filtered ranges and values
            xran_F = xrange[:]
            xran_G = xrange[:]

            # Check difference for F
            if canonical_F is None and abs(numerical_values_F[0] - analytical_values_F[0]) > threshold_F:
                xran_F = xrange[1:]
                analytical_values_F = analytical_values_F[1:]
                numerical_values_F = numerical_values_F[1:]

            # Check difference for G
            if canonical_G is None and abs(numerical_values_G[0] - analytical_values_G[0]) > threshold_G:
                xran_G = xrange[1:]
                analytical_values_G = analytical_values_G[1:]
                numerical_values_G = numerical_values_G[1:]

            # Create overlayed plots for F
            plt.figure()
            plt.plot(xran_F, analytical_values_F,
                     label=f'F analytical for {anum}n={nq}, l={lq}, j={jq}')
            plt.plot(xran_F, numerical_values_F,
                     label=f'F numerical for {anum}n={nq}, l={lq}, j={jq}', linestyle='dashed')
            plt.xlabel('r')
            plt.ylabel('Function Value')
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/F_overlayed_{nq}_{lq}_{jq}_plot.pdf')
            if self.verbose:
                plt.show()

            # Create overlayed plots for G
            plt.figure()
            plt.plot(xran_G, analytical_values_G,
                     label=f'G analytical for {anum}n={nq}, l={lq}, j={jq}')
            plt.plot(xran_G, numerical_values_G,
                     label=f'G numerical for {anum}n={nq}, l={lq}, j={jq}', linestyle='dashed')
            plt.xlabel('r')
            plt.ylabel('Function Value')
            plt.legend(loc='upper right')
            plt.savefig(f'{path}/G_overlayed_{nq}_{lq}_{jq}_plot.pdf')
            if self.verbose:
                plt.show()

        # .csv format for F analytical
        with open(f'{path}/Experiment_{idx}_F_analytical_{nq}_{lq}_{jq}.csv', 'w', newline='') as file_csv:
            writer_csv = csv.writer(file_csv)
            for r in xrange:
                writer_csv.writerow([r, analytical_F(r)])

        # .csv format for F numerical
        with open(f'{path}/Experiment_{idx}_F_numerical_{nq}_{lq}_{jq}.csv', 'w', newline='') as file_csv:
            writer_csv = csv.writer(file_csv)
            for i, r in enumerate(xrange):
                writer_csv.writerow([r, canonical_F[i] if canonical_F is not None else numerical_F(r)])

        # .csv format for G analytical
        with open(f'{path}/Experiment_{idx}_G_analytical_{nq}_{lq}_{jq}.csv', 'w', newline='') as file_csv:
            writer_csv = csv.writer(file_csv)
            for r in xrange:
                writer_csv.writerow([r, analytical_G(r)])

        # .csv format for G numerical
        with open(f'{path}/Experiment_{idx}_G_numerical_{nq}_{lq}_{jq}.csv', 'w', newline='') as file_csv:
            writer_csv = csv.writer(file_csv)
            for i, r in enumerate(xrange):
                writer_csv.writerow([r, canonical_G[i] if canonical_G is not None else numerical_G(r)])

        if dirac_coefficient_mode is not None:
            with open(f'{path}/Experiment_{idx}_dirac_metadata_{nq}_{lq}_{jq}.csv', 'w', newline='') as file_csv:
                writer_csv = csv.writer(file_csv)
                writer_csv.writerow(['dirac_coefficient_mode', dirac_coefficient_mode])
                if dirac_initialization_method is not None:
                    writer_csv.writerow(['dirac_initialization_method', dirac_initialization_method])
                writer_csv.writerow(['numerical_array_source', 'canonical_reduced_F_G'])


class KGPlotter(PlottingModule):
    def __init__(self, partl, nq, lq, alpha0, verbose, index, method):
        super().__init__(partl, 'KG')
        self.nq = nq
        self.lq = lq
        self.alpha0 = alpha0
        self.verbose = verbose
        self.experiment_idx = index
        self.method = method

    def plot(self, func_analytical, func_numerical, anum, lb, rb, ipoints, overlay=False, idx=None):
        """Plotting and exporting function values to .csv and .txt formats"""
        path = self.create_folders()

        # Define physical range of r in units of alpha0
        steps = ipoints if ipoints > 0 else 1000
        r_min = lb if lb > 0 else 1e-6 * self.alpha0  # Lower bound of r
        r_max = rb if rb > 0 else 8 * self.alpha0  # Upper bound of r
        xran = np.linspace(r_min, r_max, steps)  # Radial range in r units (not scaled yet)

        # Scale xran for plotting in units of 2*alpha0
        xran_scaled = xran / (2 * self.alpha0)  # Divide by 2*alpha0 to scale

        # Define axis labels using LaTeX notation
        x_label = r'$\frac{r}{2\alpha_0}$'
        y_label = r'$Y_{%d%d}(r)$' % (self.nq, self.lq)

        if not overlay:
            if self.method == 'NeuralNetworks':
                # Plot analytical solution for Neural Networks
                plt.figure()
                plt.plot(xran_scaled, [func_analytical(r) for r in xran],
                         label=f'Analytical Function for {anum}n={self.nq}, l={self.lq}', linestyle='-')
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend(loc='upper right')
                plt.savefig(f'{path}/NN_analytical_plot_{idx}.pdf')
                if self.verbose:
                    plt.show()

                # Plot numerical solution (Neural Networks)
                plt.figure()
                plt.plot(xran_scaled, [func_numerical(r) for r in xran],
                         label=f'Numerical (NN) Function for {anum}n={self.nq}, l={self.lq}', linestyle='--')
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend(loc='upper right')
                plt.savefig(f'{path}/NN_numerical_plot_{idx}.pdf')
                if self.verbose:
                    plt.show()
            elif self.method == 'FEM':
                # Plot analytical solution for FEM
                plt.figure()
                plt.plot(xran_scaled, [func_analytical(r) for r in xran],
                         label=f'Analytical Function for {anum}n={self.nq}, l={self.lq}', linestyle='-')
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend(loc='upper right')
                plt.savefig(f'{path}/FEM_analytical_plot_{idx}.pdf')
                if self.verbose:
                    plt.show()

                # Plot numerical solution (FEM)
                plt.figure()
                plt.plot(xran_scaled, [func_numerical(r) for r in xran],
                         label=f'Numerical (FEM) Function for {anum}n={self.nq}, l={self.lq}', linestyle='--')
                plt.xlabel(x_label)
                plt.ylabel(y_label)
                plt.legend(loc='upper right')
                plt.savefig(f'{path}/FEM_numerical_plot_{idx}.pdf')
                if self.verbose:
                    plt.show()
        else:
            # Create overlayed plot for both Neural Networks and FEM
            self._plot_overlay(path, xran_scaled, func_analytical, func_numerical, anum, idx, self.method)

        # Export to .csv format
        with open(f'./data/Experiment_{idx}_{self.nq}_{self.lq}_{self.method}.csv', 'w', newline='') as file_csv:
            writer_csv = csv.writer(file_csv)
            for r in xran:
                rnl_analytical = func_analytical(r)
                rnl_numerical = func_numerical(r)
                writer_csv.writerow([r / self.alpha0, rnl_analytical * pow(self.alpha0, 3 / 2),
                                     rnl_numerical * pow(self.alpha0, 3 / 2)])

        # Export to .txt format
        with open(f'./data/Experiment_{idx}_{self.nq}_{self.lq}_{self.method}.txt', 'w', newline='') as file_txt:
            for r in xran:
                rnl_analytical = func_analytical(r)
                rnl_numerical = func_numerical(r)
                file_txt.write(
                    f"{r / self.alpha0:10f} {rnl_analytical * pow(self.alpha0, 3 / 2)} {rnl_numerical * pow(self.alpha0, 3 / 2)}\n")

    def _plot_overlay(self, path, xran, func_analytical, func_numerical, anum, idx, method_label):
        plt.figure()
        threshold = 3e-03
        analytical_values = [func_analytical(r) for r in xran]
        numerical_values = [func_numerical(r) for r in xran]

        if abs(numerical_values[0] - analytical_values[0]) > threshold:
            xran = xran[1:]
            analytical_values = analytical_values[1:]
            numerical_values = numerical_values[1:]

        plt.plot(xran, analytical_values,
                 label=f'Analytical Function for {anum}n={self.nq}, l={self.lq}', linestyle='-')
        plt.plot(xran, numerical_values,
                 label=f'Numerical ({method_label}) Function for {anum}n={self.nq}, l={self.lq}', linestyle='--')

        plt.xlabel(r'$\frac{r}{2*\alpha_0}$')
        plt.ylabel(r'$Y_{%d%d}(r)$' % (self.nq, self.lq))
        plt.legend(loc='upper right')
        plt.savefig(f'{path}/data/Experiment_{idx}_overlayed_plot_{method_label}.pdf')
        if self.verbose:
            plt.show()

