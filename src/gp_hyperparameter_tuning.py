# --------------------------------------------------
# Automatic GP hyperparameter tuning
# --------------------------------------------------
#
# This block tunes: 
#   - kernel type
#   - Matern nu
#   - GP noise alpha
#   - length-scale bounds
#   - RationalQuadratic alpha
#
# Selection metric:
#   - Leave-One-Out Cross-Validation MSE
#
# Notes:
#   - For very small datasets, LOO is usually reasonable.
#   - For larger datasets, consider KFold instead of LeaveOneOut.
#   - alpha is now actually passed into GaussianProcessRegressor.

from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_score

# This function creates a GP kernel based on the specified kernel name and hyperparameters, 
# allowing for flexible construction of different kernel types with appropriate parameters.   
def make_gp_kernel(
    kernel_name,
    length_scale,
    length_scale_bounds,
    matern_nu=None,
    rq_alpha=0.5
):
    # Create a GP kernel from a compact hyperparameter config.
    # The constant kernel is included in all configurations to allow the GP to 
    # learn an overall signal variance, which can improve performance by 
    # scaling the kernel appropriately. The constant value bounds are set 
    # to a wide range to allow for flexibility in learning the signal variance 
    # from the data.   
    constant = ConstantKernel(
        1.0,
        constant_value_bounds=(1e-3, 1e3)
    )

    # The kernel is constructed based on the specified kernel name, 
    # with the appropriate hyperparameters passed in for each kernel type.
    if kernel_name == "RBF":
        return constant * RBF(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds
        )

    # The Matern kernel includes the nu parameter, which controls the smoothness of the function.
    if kernel_name == "Matern":
        return constant * Matern(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            nu=matern_nu
        )

    # The RationalQuadratic kernel includes the alpha parameter, which controls the 
    # relative weighting of large-scale and small-scale variations in the function. 
    # The alpha_bounds are set to "fixed" to keep it constant during optimization, 
    # as varying it can lead to more complex optimization landscapes and potential 
    # convergence issues. If you want to allow it to vary, you can set alpha_bounds=(1e-3, 1e3) 
    # or another appropriate range based on your data and problem domain.   
    if kernel_name == "RationalQuadratic":
        return constant * RationalQuadratic(
            length_scale=length_scale,
            length_scale_bounds=length_scale_bounds,
            alpha=rq_alpha,
            alpha_bounds="fixed"#alpha_bounds=(1e-3, 1e3)
        )

    raise ValueError(f"Unknown kernel_name: {kernel_name}")

# This function performs hyperparameter tuning for a Gaussian Process model using 
# Leave-One-Out Cross-Validation to evaluate different configurations of kernel type, 
# Matern nu, GP noise alpha, length-scale bounds, and RationalQuadratic alpha. 
# The results are sorted by LOO MSE to identify the best hyperparameter configuration 
# for the given dataset.   
def tune_gp_hyperparameters(
    X,
    y,
    length_scale,
    load_dataset,
    random_state=0
):
    #Tune GP hyperparameters using Leave-One-Out CV.

    # Dataset-specific defaults can still be used to keep the search sensible
    # and avoid output warning about bad hyperparameters.
    if load_dataset == 2:
        kernel_names = ["RBF", "Matern"]
        alpha_values = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
        length_scale_bounds_options = [
            (0.02, 1e2),
            (0.05, 1e2),
            (1e-2, 1e2)
        ]
    if load_dataset in [3, 6]:
        kernel_names = ["RBF", "Matern", "RationalQuadratic"]
        alpha_values = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1]
        length_scale_bounds_options = [
            (1e-3, 1e2),
            (1e-2, 1e2),
            (0.02, 1e2)
        ]
    else:
        kernel_names = ["RBF", "Matern", "RationalQuadratic"]
        alpha_values = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
        length_scale_bounds_options = [
            (1e-3, 1e2),
            (1e-2, 1e2),
            (0.02, 1e2)
        ]

    # Matern nu values are chosen to cover a range of smoothness assumptions, 
    # with 0.5 corresponding to an exponential kernel (less smooth), 1.5 providing 
    # a moderate level of smoothness, and 2.5 allowing for even smoother functions. 
    # RationalQuadratic alpha values are selected to explore different balances 
    # between large-scale and small-scale variations in the function, with lower 
    # values giving more weight to large-scale variations and higher values allowing 
    # for more small-scale variation. By including these hyperparameters in the 
    # tuning process, we can better capture the underlying structure of the data 
    # and improve the GP's predictive performance.  
    matern_nu_values = [0.5, 1.5, 2.5]

    # RationalQuadratic alpha values are chosen to explore a range of balances 
    # between large-scale and small-scale variations in the function, with lower 
    # values giving more weight to large-scale variations and higher values allowing 
    # for more small-scale variation. By including these hyperparameters in the 
    # tuning process, we can better capture the underlying structure of the data 
    # and improve the GP's predictive performance.  
    rq_alpha_values = [0.1, 0.5, 1.0, 2.0]
    
    search_results = []

    print(
        "\n================================ "
        "AUTOMATIC GP HYPERPARAMETER TUNING "
        "==================================\n"
    )

    # Iterate over all combinations of hyperparameters and evaluate using Leave-One-Out CV MSE.
    for kernel_name in kernel_names:
        # The hyperparameters are iterated in a way that allows for conditional parameters based on the kernel type.
        for alpha_value in alpha_values:
            # Length scale bounds are always varied to allow the optimizer to find the best length scale within 
            # a reasonable range, which can significantly impact the GP's performance.
            for ls_bounds in length_scale_bounds_options:
                # Matern nu is only relevant for the Matern kernel, so we conditionally set the values to try based on the kernel type.
                if kernel_name == "Matern":
                    nu_values_to_try = matern_nu_values
                else:
                    nu_values_to_try = [None]
                # RationalQuadratic alpha is only relevant for the RationalQuadratic kernel, so we conditionally set the values to try based on the kernel type.
                if kernel_name == "RationalQuadratic":
                    rq_alpha_values_to_try = rq_alpha_values
                else:
                    rq_alpha_values_to_try = [None]
                # Now we iterate over the relevant values for Matern nu and RationalQuadratic alpha based on the kernel
                #  type, allowing us to explore the hyperparameter space effectively while avoiding irrelevant combinations.
                for matern_nu in nu_values_to_try:
                    for rq_alpha in rq_alpha_values_to_try:
                        # Create the GP kernel based on the current combination of hyperparameters, 
                        # using the make_gp_kernel function defined above to construct the appropriate 
                        # kernel object for the GaussianProcessRegressor.
                        kernel = make_gp_kernel(
                            kernel_name=kernel_name,
                            length_scale=length_scale,
                            length_scale_bounds=ls_bounds,
                            matern_nu=matern_nu,
                            rq_alpha=0.5 if rq_alpha is None else rq_alpha
                        )
                        # Initialize the GaussianProcessRegressor with the current kernel and alpha value,
                        # and set normalize_y to True to normalize the target values, n_restarts_optimizer to 5 
                        # to allow for multiple restarts of the optimizer to find better hyperparameters, 
                        # and random_state for reproducibility.
                        gp = GaussianProcessRegressor(
                            kernel=kernel,
                            alpha=alpha_value,
                            normalize_y=True,
                            n_restarts_optimizer=5,
                            random_state=random_state
                        )

                        try:
                            # Evaluate the GP using Leave-One-Out Cross-Validation and calculate the mean 
                            # and standard deviation of the negative mean squared error scores, which are 
                            # then negated to get the LOO MSE. This provides a robust estimate of the GP's 
                            # performance for the current hyperparameter configuration.    
                            scores = cross_val_score(
                                gp,
                                X,
                                y,
                                cv=LeaveOneOut(),
                                scoring="neg_mean_squared_error",
                                n_jobs=-1,
                                pre_dispatch="2*n_jobs"
                            )

                            loo_mse = -scores.mean()
                            loo_mse_std = scores.std()

                            # Record the results for this hyperparameter configuration, including the kernel name, 
                            # alpha value, length scale bounds, Matern nu, RationalQuadratic alpha, and the LOO MSE 
                            # and its standard deviation. This allows us to compare different configurations and 
                            # select the best one based on the lowest LOO MSE.    
                            search_results.append({
                                "kernel_name": kernel_name,
                                "alpha": alpha_value,
                                "length_scale_bounds": ls_bounds,
                                "matern_nu": matern_nu,
                                "rq_alpha": rq_alpha,
                                "loo_mse": loo_mse,
                                "loo_mse_std": loo_mse_std,
                                "kernel": kernel
                            })

                            print(
                                f"{kernel_name:18s} "
                                f"alpha={alpha_value:<9.1e} "
                                f"bounds={str(ls_bounds):14s} "
                                f"nu={str(matern_nu):4s} "
                                f"rq_alpha={str(rq_alpha):4s} "
                                f"LOO MSE={loo_mse:.6f}"
                            )

                        except Exception as exc:
                            print(
                                f"Skipping {kernel_name}, "
                                f"alpha={alpha_value}, "
                                f"bounds={ls_bounds}, "
                                f"nu={matern_nu}, "
                                f"rq_alpha={rq_alpha}: {exc}"
                            )

    if not search_results:
        raise RuntimeError("No valid GP hyperparameter configurations were found.")

    # Sort by lowest LOO MSE.
    search_results = sorted(search_results, key=lambda row: row["loo_mse"])

    best_config = search_results[0]

    print(
        "\n================ "
        "BEST GP HYPERPARAMETER CONFIGURATION "
        "=================\n"
    )
    print(f"Best kernel: {best_config['kernel_name']}")
    print(f"Best alpha: {best_config['alpha']}")
    print(f"Best length_scale_bounds: {best_config['length_scale_bounds']}")
    print(f"Best Matern nu: {best_config['matern_nu']}")
    print(f"Best RationalQuadratic alpha: {best_config['rq_alpha']}")
    print(f"Best LOO MSE: {best_config['loo_mse']:.6f}")

    print(
        "\nTop 5 GP configurations:"
    )
    for rank, row in enumerate(search_results[:5], start=1):
        print(
            f"{rank}. {row['kernel_name']} | "
            f"alpha={row['alpha']} | "
            f"bounds={row['length_scale_bounds']} | "
            f"nu={row['matern_nu']} | "
            f"rq_alpha={row['rq_alpha']} | "
            f"LOO MSE={row['loo_mse']:.6f}"
        )

    return best_config, search_results