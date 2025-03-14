from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import patsy as pt
import scanpy as sc
import seaborn as sns
from adjustText import adjust_text
from anndata import AnnData
from jax import config, random
from lamin_utils import logger
from matplotlib import cm, rcParams
from matplotlib import image as mpimg
from matplotlib.colors import ListedColormap
from mudata import MuData
from numpyro.infer import HMC, MCMC, NUTS, initialization
from rich import box, print
from rich.console import Console
from rich.table import Table
from scipy.cluster import hierarchy as sp_hierarchy

from pertpy._doc import _doc_params, doc_common_plot_args

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpyro as npy
    import toytree as tt
    from ete3 import Tree
    from jax._src.typing import Array
    from matplotlib.axes import Axes
    from matplotlib.colors import Colormap
    from matplotlib.figure import Figure

config.update("jax_enable_x64", True)

class CompositionalModel2(ABC):
    """General compositional model framework for scCODA-type models.

    This class serves as a template for scCODA-style models. It handles:

    - General data preprocessing

    - Inference

    - Result creation

    - Inference algorithms

    An instance of this class has two main attributes.
    `sample_adata` is an `AnnData` object that contains the aggregated counts of N samples and P features (e.g. cell types),
    with N on the `obs` axis and P on the `var` axis. All other information about the model, such as initial parameters,
    references, or the model formula are stored in `sample_adata.uns`.
    After running a numpyro inference algorithm, such as MCMC sampling, the result will be stored in another class attribute.

    Particular models can be implemented as children of this class.
    The following parameters must be set during subclass initialization:

    - `sample_adata.uns["param_names"]`:
    List with the names of all tracked latent model parameters (through `npy.sample` or `npy.deterministic`)

    - `sample_adata.uns["scCODA_params"]["model_type"]`:
    String indicating the model type ("classic" or "tree_agg")

    - `sample_adata.uns["scCODA_params"]["select_type"]`:
    String indicating the type of spike_and_slab selection ("spikeslab" or "sslasso")

    Additionally, a subclass must implement at least these functions (see subclasses for examples):

    - `model`: The model formulation

    - `set_init_mcmc_states`: A function to set the initial state of the MCMC algorithm

    - `make_arviz`: A function to generate an arviz result object
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def make_arviz(self, *args, **kwargs):
        pass

    @abstractmethod
    def model(self, *args, **kwargs):
        pass

    @abstractmethod
    def set_init_mcmc_states(self, *args, **kwargs):
        pass

    def prepare(
        self,
        sample_adata: AnnData,
        formula: str,
        reference_cell_type: str = "automatic",
        automatic_reference_absence_threshold: float = 0.05,
    ) -> AnnData:
        """Handles data preprocessing, covariate matrix creation, reference selection, and zero count replacement.

        Args:
            sample_adata: anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            formula: R-style formula for building the covariate matrix.
                Categorical covariates are handled automatically, with the covariate value of the first sample being used as the reference category.
                To set a different level as the base category for a categorical covariate, use "C(<CovariateName>, Treatment('<ReferenceLevelName>'))"
            reference_cell_type: Column name that sets the reference cell type.
                Reference the name of a column. If "automatic", the cell type with the lowest dispersion in relative abundance that is present in at least 90% of samlpes will be chosen.
            automatic_reference_absence_threshold: If using reference_cell_type = "automatic", determine the maximum fraction of zero entries for a cell type
                to be considered as a possible reference cell type.

        Returns:
            AnnData object that is ready for CODA models.
        """
        dtype = "float64"

        # Convert count data to float64 (needed for correct inference)
        sample_adata.X = sample_adata.X.astype(dtype)

        # Build covariate matrix from R-like formula, save in obsm
        covariate_matrix = pt.dmatrix(formula, sample_adata.obs)
        covariate_names = covariate_matrix.design_info.column_names[1:]
        sample_adata.obsm["covariate_matrix"] = np.array(covariate_matrix[:, 1:]).astype(dtype)

        cell_types = sample_adata.var.index.to_list()

        # Invoke instance of the correct model depending on reference cell type
        # Automatic reference selection (dispersion-based)
        if reference_cell_type == "automatic":
            percent_zero = np.sum(sample_adata.X == 0, axis=0) / sample_adata.X.shape[0]
            nonrare_ct = np.where(percent_zero < automatic_reference_absence_threshold)[0]

            if len(nonrare_ct) == 0:
                raise ValueError(
                    "No cell types that have large enough presence! Please increase automatic_reference_absence_threshold"
                )

            rel_abun = sample_adata.X / np.sum(sample_adata.X, axis=1, keepdims=True)

            # select reference
            cell_type_disp = np.var(rel_abun, axis=0) / np.mean(rel_abun, axis=0)
            min_var = np.min(cell_type_disp[nonrare_ct])
            ref_index = np.where(cell_type_disp == min_var)[0][0]

            ref_cell_type = cell_types[ref_index]
            logger.info(f"Automatic reference selection! Reference cell type set to {ref_cell_type}")

        # Column name as reference cell type
        elif reference_cell_type in cell_types:
            ref_index = cell_types.index(reference_cell_type)

        # None of the above: Throw error
        else:
            raise NameError("Reference index is not a valid cell type name or numerical index!")

        # Add pseudocount if zeroes are present.
        if np.count_nonzero(sample_adata.X) != np.size(sample_adata.X):
            logger.info("Zero counts encountered in data! Added a pseudocount of 0.5.")
            sample_adata.X[sample_adata.X == 0] = 0.5

        sample_adata.obsm["sample_counts"] = np.sum(sample_adata.X, axis=1)

        # Check input data
        if covariate_matrix.shape[0] != sample_adata.X.shape[0]:
            row_len_covariate_matrix = sample_adata.obsm["covariate_matrix"].shape[0]
            row_len_sample_adata = sample_adata.X.shape[0]
            raise ValueError(f"Wrong input dimensions X[{row_len_covariate_matrix},:] != y[{row_len_sample_adata},:]")
        if covariate_matrix.shape[0] != len(sample_adata.obsm["sample_counts"]):
            covariate_matrix = sample_adata.obsm["covariate_matrix"]
            len_sample_counts = len(sample_adata.obsm["sample_counts"])
            raise ValueError(f"Wrong input dimensions X[{covariate_matrix},:] != n_total[{len_sample_counts}]")

        # Save important model parameters in uns
        sample_adata.uns["scCODA_params"] = {
            "formula": formula,
            "reference_cell_type": cell_types[ref_index],
            "reference_index": ref_index,
            "automatic_reference_absence_threshold": automatic_reference_absence_threshold,
            "covariate_names": covariate_names,
            "mcmc": {"init_params": []},
        }
        return sample_adata

    def __run_mcmc(
        self,
        sample_adata: AnnData,
        kernel: npy.infer.mcmc.MCMCKernel,
        rng_key: Array,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Background function that executes any numpyro MCMC algorithm and processes its results

        Args:
            sample_adata: anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            kernel: A `numpyro.infer.mcmc.MCMCKernel` object
            rng_key: The rng state used. If None, a random state will be selected
            copy: Return a copy instead of writing to adata.
            args: Passed to `numpyro.infer.mcmc.MCMC`
            kwargs: Passed to `numpyro.infer.mcmc.MCMC`

        Returns:
            Saves all results into `sample_adata` and generates `self.mcmc` as a class attribute. If copy==True, return a copy of adata.
        """
        dtype = "float64"
        # The tracked MCMC parameters for diagnostic checks
        extra_fields = (
            "potential_energy",
            "num_steps",
            "adapt_state.step_size",
            "accept_prob",
            "mean_accept_prob",
        )

        # Convert all data needed for modeling to numpyro arrays
        numpyro_counts = jnp.array(sample_adata.X, dtype=dtype)
        numpyro_covariates = jnp.array(sample_adata.obsm["covariate_matrix"], dtype=dtype)
        numpyro_n_total = jnp.array(sample_adata.obsm["sample_counts"], dtype=dtype)

        # Extract one-hot encoded region labels
        region_labels_onehot = jnp.array(sample_adata.obsm["region_labels"])  # Shape (N, R)
        num_regions = region_labels_onehot.shape[1]  # Extract number of regions
    
        # Extract the cell type presence mask
        cell_type_region_mask = jnp.array(sample_adata.obsm["cell_type_region_mask"])


        # Create mcmc attribute and run inference
        self.mcmc = MCMC(kernel, chain_method="vectorized", *args, **kwargs)
        self.mcmc.run(
            rng_key,
            numpyro_counts,
            numpyro_covariates,
            numpyro_n_total,
            sample_adata.uns["scCODA_params"]["reference_index"],
            region_labels_onehot,  # Pass one-hot encoded regions
            cell_type_region_mask,
            extra_fields=extra_fields,
        )
        
        acc_rate = np.mean(self.mcmc.last_state.mean_accept_prob)

        if acc_rate < 0.6:
            logger.warning(
                f"Acceptance rate unusually low ({acc_rate} < 0.5)! Results might be incorrect! "
                f"Please check feasibility of results and re-run the sampling step with a different rng_key if necessary."
            )
        if acc_rate > 0.95:
            logger.warning(
                f"Acceptance rate unusually high ({acc_rate} > 0.95)! Results might be incorrect! "
                f"Please check feasibility of results and re-run the sampling step with a different rng_key if necessary."
            )

        
        # Set acceptance rate and save sampled values to `sample_adata.uns`
        sample_adata.uns["scCODA_params"]["mcmc"]["acceptance_rate"] = np.array(self.mcmc.last_state.mean_accept_prob)
        samples = self.mcmc.get_samples()
        for k, v in samples.items():
            samples[k] = np.array(v)
        sample_adata.uns["scCODA_params"]["mcmc"]["samples"] = samples
        
        # Evaluate results and create result dataframes (based on tree-aggregation or not)
        if sample_adata.uns["scCODA_params"]["model_type"] == "classic":
            intercept_df, effect_df = self.summary_prepare(sample_adata)  # type: ignore
        else:
            raise ValueError("No valid model type!")
        
        # Save intercept and effect dfs in `sample_adata.varm` (one effect df per covariate)
        for region in intercept_df.index.get_level_values("Region").unique():
            sample_adata.varm[f"intercept_df_{region}"] = intercept_df.loc[region, :]

        for region in effect_df.index.get_level_values("Region").unique():
            for cov in effect_df.index.get_level_values("Covariate").unique():
                sample_adata.varm[f"effect_df_{region}_{cov}"] = effect_df.loc[(region, cov), :]
        
        if copy:
            return sample_adata

    def run_nuts(
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        num_chains: int = 1,
        num_samples: int = 10000,
        num_warmup: int = 1000,
        rng_key: int = 0,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Run No-U-turn sampling (Hoffman and Gelman, 2014), an efficient version of Hamiltonian Monte Carlo sampling to infer optimal model parameters.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.
            num_samples: Number of sampled values after burn-in.
            num_warmup: Number of burn-in (warmup) samples.
            rng_key: The rng state used.
            copy: Return a copy instead of writing to adata.

        Returns:
            Calls `self.__run_mcmc`
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data
        if copy:
            sample_adata = sample_adata.copy()

        rng_key_array = random.key_data(random.key(rng_key))
        sample_adata.uns["scCODA_params"]["mcmc"]["rng_key"] = np.array(rng_key_array)

        # Set up NUTS kernel
        sample_adata = self.set_init_mcmc_states(
            rng_key, sample_adata.uns["scCODA_params"]["reference_index"], sample_adata
        )
        init_params = sample_adata.uns["scCODA_params"]["mcmc"]["init_params"]
        nuts_kernel = NUTS(self.model, *args, init_strategy=initialization.init_to_value(values=init_params), **kwargs)
        # Save important parameters in `sample_adata.uns`
        sample_adata.uns["scCODA_params"]["mcmc"]["num_samples"] = num_samples
        sample_adata.uns["scCODA_params"]["mcmc"]["num_warmup"] = num_warmup
        sample_adata.uns["scCODA_params"]["mcmc"]["algorithm"] = "NUTS"

        return self.__run_mcmc(
            sample_adata, nuts_kernel, num_chains=num_chains, num_samples=num_samples, num_warmup=num_warmup, rng_key=rng_key_array, copy=copy
        )

    def run_hmc(
        self,
        data: AnnData | MuData,
        modality_key: str = "coda",
        num_samples: int = 20000,
        num_warmup: int = 5000,
        rng_key=None,
        copy: bool = False,
        *args,
        **kwargs,
    ):
        """Run standard Hamiltonian Monte Carlo sampling (Neal, 2011) to infer optimal model parameters.

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.
            num_samples: Number of sampled values after burn-in.
            num_warmup: Number of burn-in (warmup) samples.
            rng_key: The rng state used. If None, a random state will be selected.
            copy: Return a copy instead of writing to adata.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_hmc(mdata, num_warmup=100, num_samples=1000)
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data
        if copy:
            sample_adata = sample_adata.copy()

        # Set rng key if needed
        if rng_key is None:
            rng = np.random.default_rng()
            rng_key = random.key(rng.integers(0, 10000))
            sample_adata.uns["scCODA_params"]["mcmc"]["rng_key"] = rng_key
        else:
            rng_key = random.key(rng_key)

        # Set up HMC kernel
        sample_adata = self.set_init_mcmc_states(
            rng_key, sample_adata.uns["scCODA_params"]["reference_index"], sample_adata
        )
        init_params = sample_adata.uns["scCODA_params"]["mcmc"]["init_params"]
        hmc_kernel = HMC(self.model, *args, init_strategy=initialization.init_to_value(values=init_params), **kwargs)

        # Save important parameters in `sample_adata.uns`
        sample_adata.uns["scCODA_params"]["mcmc"]["num_samples"] = num_samples
        sample_adata.uns["scCODA_params"]["mcmc"]["num_warmup"] = num_warmup
        sample_adata.uns["scCODA_params"]["mcmc"]["algorithm"] = "HMC"

        return self.__run_mcmc(
            sample_adata, hmc_kernel, num_samples=num_samples, num_warmup=num_warmup, rng_key=rng_key, copy=copy
        )

    def summary_prepare(
        self, sample_adata: AnnData, est_fdr: float = 0.1, *args, **kwargs
    ) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generates summary dataframes for intercepts, effects and node-level effect (if using tree aggregation).
            This function builds on and supports all functionalities from ``az.summary``.

        Args:
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            est_fdr: Desired FDR value.
            args: Passed to ``az.summary``
            kwargs: Passed to ``az.summary``

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame] or Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Intercept, effect and node-level DataFrames

            intercept_df
                Summary of intercept parameters. Contains one row per cell type.

                - Final Parameter: Final intercept model parameter
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Expected sample: Expected cell counts for a sample with no present covariates. See the tutorial for more explanation

            effect_df
                Summary of effect (slope) parameters. Contains one row per covariate/cell type combination.

                - Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Expected sample: Expected cell counts for a sample with only the current covariate set to 1. See the tutorial for more explanation
                - log2-fold change: Log2-fold change between expected cell counts with no covariates and with only the current covariate
                - Inclusion probability: Share of MCMC samples, for which this effect was not set to 0 by the spike-and-slab prior.

            node_df
                Summary of effect (slope) parameters on the tree nodes (features or groups of features). Contains one row per covariate/cell type combination.

                - Final Parameter: Final effect model parameter. If this parameter is 0, the effect is not significant, else it is.
                - Median: Median of parameter over MCMC chain
                - HDI X%: Upper and lower boundaries of confidence interval (width specified via hdi_prob=)
                - SD: Standard deviation of MCMC samples
                - Delta: Decision boundary value - threshold of practical significance
                - Is credible: Boolean indicator whether effect is credible

         Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> intercept_df, effect_df = sccoda.summary_prepare(mdata["coda"])
        """
        select_type = sample_adata.uns["scCODA_params"]["select_type"]
        model_type = sample_adata.uns["scCODA_params"]["model_type"]

        # Create arviz summary for intercepts, effects and node effects
        var_names = ["alpha_per_region", "beta_full", "beta_region_specific"]
        
        # Run ArviZ summary
        summ = az.summary(
            data=self.make_arviz(sample_adata, num_prior_samples=0, use_posterior_predictive=False),
            var_names=var_names,
            kind="stats",
            stat_funcs={"median": np.median},
            *args,  
            **kwargs,
        )  

        # Separate shared and region-specific effects
        shared_effects_df = summ.loc[summ.index.str.match("|".join([r"beta_full\["]))].copy()
        region_effects_df = summ.loc[summ.index.str.match("|".join([r"beta_region_specific\["]))].copy()
        intercept_df = summ.loc[summ.index.str.match("|".join([r"alpha_per_region\["]))].copy()
        
        # Indexing
        cell_types = sample_adata.var.index.to_list()
        covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
        regions = sample_adata.uns["scCODA_params"]["region_names"]

        intercept_df.index = pd.MultiIndex.from_product([regions, cell_types], names=["Region", "Cell Type"])
        shared_effects_df.index = pd.MultiIndex.from_product([["Global"], covariates, cell_types], names=["Region", "Covariate", "Cell Type"])
        region_effects_df.index = pd.MultiIndex.from_product([regions, covariates, cell_types], names=["Region", "Covariate", "Cell Type"])

        intercept_df = intercept_df.rename(columns={"mean": "final_parameter"})

        # Complete effect df
        shared_effects_df = self.__complete_beta_df(
            sample_adata, intercept_df, shared_effects_df, "beta_full", target_fdr=est_fdr,
        )
        region_effects_df = self.__complete_beta_df(
            sample_adata, intercept_df, region_effects_df, "beta_region_specific", target_fdr=est_fdr,
        )

        # Give nice column names, remove unnecessary columns
        hdis = intercept_df.columns[intercept_df.columns.str.contains("hdi")]
        hdis_new = hdis.str.replace("hdi_", "HDI ")

        # Calculate credible intervals if using classical spike-and-slab
        ind_post_shared = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["ind_shared"])
        ind_post_shared[ind_post_shared < 1e-3] = np.nan

        b_raw_sel_shared = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["b_offset_shared"]) * ind_post_shared

        res_shared = az.convert_to_inference_data(np.array([b_raw_sel_shared]))

        summary_sel = az.summary(
            data=res_shared,
            kind="stats",
            var_names=["x"],
            skipna=True,
            *args,  # noqa: B026
            **kwargs,
        )
        
        ref_index = sample_adata.uns["scCODA_params"]["reference_index"]
        n_conditions = len(covariates)
        n_cell_types = len(cell_types)

        def insert_row(idx, df, df_insert):
            return pd.concat(
                [
                    df.iloc[:idx,],
                    df_insert,
                    df.iloc[idx:,],
                ]
            ).reset_index(drop=True)

        for i in range(n_conditions):
            summary_sel = insert_row(
                (i * n_cell_types) + ref_index,
                summary_sel,
                pd.DataFrame.from_dict(data={"mean": [0], "sd": [0], hdis[0]: [0], hdis[1]: [0]}),
            )

        shared_effects_df.loc[:, hdis[0]] = list(summary_sel[hdis[0]])
        shared_effects_df.loc[:, hdis[1]] = list(summary_sel.loc[:, hdis[1]])  # type: ignore

        # Calculate credible intervals if using classical spike-and-slab
        ind_post_region = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["ind_region"])
        ind_post_region[ind_post_region < 1e-3] = np.nan

        b_raw_sel_region = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"]["b_offset_region"]) * ind_post_region

        res_region = az.convert_to_inference_data(np.array([b_raw_sel_region]))

        summary_sel = az.summary(
            data=res_region,
            kind="stats",
            var_names=["x"],
            skipna=True,
            *args,  # noqa: B026
            **kwargs,
        )

        ref_index = sample_adata.uns["scCODA_params"]["reference_index"]
        n_regions = len(regions)
        n_conditions = len(covariates)
        n_cell_types = len(cell_types)

        for j in range(n_regions):
            for i in range(n_conditions):
                summary_sel = insert_row(
                    (j * n_conditions * n_cell_types) + (i * n_cell_types) + ref_index,
                    summary_sel,
                    pd.DataFrame.from_dict(data={"mean": [0], "sd": [0], hdis[0]: [0], hdis[1]: [0]}),
                )

        region_effects_df.loc[:, hdis[0]] = list(summary_sel[hdis[0]])
        region_effects_df.loc[:, hdis[1]] = list(summary_sel.loc[:, hdis[1]])  # type: ignore

        effect_df = pd.concat([shared_effects_df, region_effects_df], axis=0)

        # Select relevant columns and give nice column names for all result dfs, then return them
        intercept_df = intercept_df.loc[:, ["final_parameter", hdis[0], hdis[1], "sd"]].copy()
        intercept_df = intercept_df.rename(
            columns=dict(
                zip(
                    intercept_df.columns,
                    ["Final Parameter", hdis_new[0], hdis_new[1], "SD"],
                    strict=False,
                )
            )
        )
    
        effect_df = effect_df.loc[
            :, ["final_parameter", hdis[0], hdis[1], "sd", "inclusion_prob"]
        ].copy()
        effect_df = effect_df.rename(
            columns=dict(
                zip(
                    effect_df.columns,
                    [
                        "Final Parameter",
                        hdis_new[0],
                        hdis_new[1],
                        "SD",
                        "Inclusion probability",
                    ],
                    strict=False,
                )
            )
        )

        return intercept_df, effect_df

    def __complete_beta_df(
        self,
        sample_adata: AnnData,
        intercept_df: pd.DataFrame,
        effect_df: pd.DataFrame,
        mcmc_beta_key: str,
        target_fdr: float = 0.1,
    ) -> pd.DataFrame:
        """Evaluation of MCMC results for effect parameters. This function is only used within self.summary_prepare.

        This function also calculates the posterior inclusion probability for each effect and decides whether effects are significant.

        Args:
            sample_adata: Anndata object with cell counts as sample_adata.X and covariates saved in sample_adata.obs.
            intercept_df: Intercept summary, see ``summary_prepare``
            effect_df: Effect summary, see ``summary_prepare``
            model_type: String indicating the model type ("classic" or "tree_agg")
            target_fdr: Desired FDR value.
            node_df: If using tree aggregation, the node-level effect DataFrame must be passed.

        Returns:
            pd.DataFrame:  effect DataFrame with inclusion probability, final parameters, expected sample.
        """

        # Effect processing for different models
        # Classic scCODA (spike-and-slab + no tree aggregation)
        beta_inc_prob = []
        beta_nonzero_mean = []

        # Get MCMC samples for parameter "beta"
        beta_raw = np.array(sample_adata.uns["scCODA_params"]["mcmc"]["samples"][mcmc_beta_key])
        
        # Calculate inclusion prob, nonzero mean for every effect
        if len(beta_raw.shape) == 3:
            for j in range(beta_raw.shape[1]):
                for i in range(beta_raw.shape[2]):
                    beta_i_raw = beta_raw[:, j, i]
                    beta_i_raw_nonzero = np.where(np.abs(beta_i_raw) > 1e-3)[0]
                    prob = beta_i_raw_nonzero.shape[0] / beta_i_raw.shape[0]
                    beta_inc_prob.append(prob)
                    if len(beta_i_raw[beta_i_raw_nonzero]) > 0:
                        beta_nonzero_mean.append(beta_i_raw[beta_i_raw_nonzero].mean())
                    else:
                        beta_nonzero_mean.append(0)

        elif len(beta_raw.shape) == 4:
            for j in range(beta_raw.shape[1]):
                for i in range(beta_raw.shape[2]):
                    for k in range(beta_raw.shape[3]):
                        beta_i_raw = beta_raw[:, j, i, k]
                        beta_i_raw_nonzero = np.where(np.abs(beta_i_raw) > 1e-3)[0]
                        prob = beta_i_raw_nonzero.shape[0] / beta_i_raw.shape[0]
                        beta_inc_prob.append(prob)
                        if len(beta_i_raw[beta_i_raw_nonzero]) > 0:
                            beta_nonzero_mean.append(beta_i_raw[beta_i_raw_nonzero].mean())
                        else:
                            beta_nonzero_mean.append(0)
        else:
            raise ValueError("To many dimensions in the effect size table.")

        effect_df.loc[:, "inclusion_prob"] = beta_inc_prob
        effect_df.loc[:, "mean_nonzero"] = beta_nonzero_mean

        # Inclusion prob threshold value. Direct posterior probability approach cf. Newton et al. (2004)
        def opt_thresh(result, alpha):
            incs = np.array(result.loc[result["inclusion_prob"] > 0, "inclusion_prob"])
            incs[::-1].sort()

            for c in np.unique(incs):
                fdr = np.mean(1 - incs[incs >= c])

                if fdr < alpha:
                    # ceiling with 3 decimals precision
                    c = np.floor(c * 10**3) / 10**3
                    return c, fdr
            return 1.0, 0

        threshold, fdr_ = opt_thresh(effect_df, target_fdr)

        # Save cutoff inclusion probability to scCODA params in uns
        try:
            sample_adata.uns["scCODA_params"]["threshold_prob"][mcmc_beta_key] = threshold
        except KeyError:
            sample_adata.uns["scCODA_params"]["threshold_prob"] = {}
            sample_adata.uns["scCODA_params"]["threshold_prob"][mcmc_beta_key] = threshold

        effect_df.loc[:, "final_parameter"] = effect_df.loc[:, "mean_nonzero"]
        )
        
        return effect_df

    def summary(self, data: AnnData | MuData, extended: bool = False, modality_key: str = "coda", *args, **kwargs):
        """Printing method for the summary.

        Args:
            data: AnnData object or MuData object.
            extended: If True, return the extended summary with additional statistics.
            modality_key: If data is a MuData object, specify which modality to use.
            args: Passed to az.summary
            kwargs: Passed to az.summary

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> sccoda.summary(mdata)
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        select_type = sample_adata.uns["scCODA_params"]["select_type"]
        model_type = sample_adata.uns["scCODA_params"]["model_type"]

        # If other than default values for e.g. confidence interval are specified,
        # recalculate them for intercept and effect DataFrames
        if args or kwargs:
            intercept_df, effect_df = self.summary_prepare(sample_adata, *args, **kwargs)  # type: ignore
        # otherwise, get pre-calculated DataFrames. Effect DataFrame is stitched together from varm
        else:
            intercept_df = sample_adata.varm["intercept_df"]
            covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
            region_names = ["Global"]
            region_names.extend(sample_adata.uns["scCODA_params"]["region_names"])
            effect_dfs = []
            for region in region_names:
                for cov in covariates:
                    effect_dfs.append(sample_adata.varm[f"effect_df_{region}_{cov}"])
            effect_df = pd.concat(effect_dfs)
            effect_df.index = pd.MultiIndex.from_product(
                (region_names, covariates, sample_adata.var.index.tolist()), names=["Region", "Covariate", "Cell Type"]
            )
            effect_df.index = effect_df.index.set_levels(
                effect_df.index.levels[1].str.replace("Condition", "").str.replace("[", "").str.replace("]", ""),
                level=1,
            )
            
        # Get number of samples, cell types
        data_dims = sample_adata.X.shape

        console = Console()
        table = Table(title="Compositional Analysis summary", box=box.SQUARE, expand=True, highlight=True)
        table.add_column("Name", justify="left", style="cyan")
        table.add_column("Value", justify="left")
        table.add_row("Data", "Data: %d samples, %d cell types" % data_dims)
        table.add_row("Reference cell type", "{}".format(str(sample_adata.uns["scCODA_params"]["reference_cell_type"])))
        table.add_row("Formula", "{}".format(sample_adata.uns["scCODA_params"]["formula"]))
        if extended:
            table.add_row("Reference index", "{}".format(str(sample_adata.uns["scCODA_params"]["reference_index"])))
            if select_type == "spikeslab":
                table.add_row(
                    "Spike-and-slab threshold",
                    "{threshold:.3f}".format(threshold=sample_adata.uns["scCODA_params"]["threshold_prob"]),
                )
                table.add_row(
                    "Spike-and-slab threshold",
                    "{threshold:.3f}".format(threshold=sample_adata.uns["scCODA_params"]["threshold_prob"]),
                )
            num_results = sample_adata.uns["scCODA_params"]["mcmc"]["num_samples"]
            num_burnin = sample_adata.uns["scCODA_params"]["mcmc"]["num_warmup"]
            table.add_row("MCMC Sampling", f"Sampled {num_results} chain states ({num_burnin} burnin samples)")
            table.add_row(
                "Acceptance rate",
                "{ar:.1f}%".format(
                    ar=(100 * sample_adata.uns["scCODA_params"]["mcmc"]["acceptance_rate"]),
                ),
            )
        console.print(table)

        intercept_df_basic = intercept_df.loc[:, intercept_df.columns.isin(["Final Parameter"])]
        effect_df_basic = effect_df.loc[
            :, effect_df.columns.isin(["Final Parameter"])
        ]
        effect_df_extended = effect_df.loc[
            :, ~effect_df.columns.isin(["Final Parameter"])
        ]
        if extended:
            table = Table("Intercepts", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(intercept_df.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            table = Table("Effects", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(effect_df_basic.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            table = Table("Effects Extended", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(effect_df_extended.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

        else:
            table = Table("Intercepts", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(intercept_df_basic.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

            table = Table("Effects", box=box.SQUARE, expand=True, highlight=True)
            table.add_row(effect_df_basic.to_string(justify="center", float_format=lambda _: f"{_:.3f}"))
            console.print(table)

    def get_intercept_df(self, data: AnnData | MuData, modality_key: str = "coda"):
        """Get intercept dataframe as printed in the extended summary

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.

        Returns:
            pd.DataFrame: Intercept data frame.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> intercepts = sccoda.get_intercept_df(mdata)
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        return sample_adata.varm["intercept_df"]

    def get_effect_df(self, data: AnnData | MuData, modality_key: str = "coda"):
        """Get effect dataframe as printed in the extended summary

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.

        Returns:
            pd.DataFrame: Effect data frame.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> effects = sccoda.get_effect_df(mdata)
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
        region_names = ["Global"]
        region_names.extend(sample_adata.uns["scCODA_params"]["region_names"])
        effect_dfs = []
        for region in region_names:
            for cov in covariates:
                effect_dfs.append(sample_adata.varm[f"effect_df_{region}_{cov}"])
        effect_df = pd.concat(effect_dfs)
        effect_df.index = pd.MultiIndex.from_product(
            (covariates, sample_adata.var.index.tolist()), names=["Region", "Covariate", "Cell Type"]
        )
        effect_df.index = effect_df.index.set_levels(
            effect_df.index.levels[1].str.replace("Condition", "").str.replace("[", "").str.replace("]", ""),
            level=0,
        )

        return effect_df

    def set_fdr(self, data: AnnData | MuData, est_fdr: float, modality_key: str = "coda", *args, **kwargs):
        """Direct posterior probability approach to calculate credible effects while keeping the expected FDR at a certain level
            Note: Does not work for spike-and-slab LASSO selection method

        Args:
            data: AnnData object or MuData object.
            est_fdr: Desired FDR value.
            modality_key: If data is a MuData object, specify which modality to use.
            args: passed to self.summary_prepare
            kwargs: passed to self.summary_prepare

        Returns:
            Adjusts intercept_df and effect_df
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        if sample_adata.uns["scCODA_params"]["model_type"] == "classic":
            intercept_df, effect_df = self.summary_prepare(sample_adata, est_fdr, *args, **kwargs)  # type: ignore
        else:
            raise ValueError("No valid model type!")

        for region in intercept_df.index.get_level_values("Region").unique():
            sample_adata.varm[f"intercept_df_{region}"] = intercept_df.loc[region, :]
        
        for region in effect_df.index.get_level_values("Region").unique():
            for cov in effect_df.index.get_level_values("Covariate").unique():
                sample_adata.varm[f"effect_df_{region}_{cov}"] = effect_df.loc[(region, cov), :]

    def credible_effects(self, data: AnnData | MuData, modality_key: str = "coda", est_fdr: float = None) -> pd.Series:
        """Decides which effects of the scCODA model are credible based on an adjustable inclusion probability threshold.
            Note: Parameter est_fdr has no effect for spike-and-slab LASSO selection method

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.
            est_fdr: Estimated false discovery rate. Must be between 0 and 1.

        Returns:
            pd.Series: Credible effect decision series which includes boolean values indicate whether effects are credible under inc_prob_threshold.
        """
        if isinstance(data, MuData):
            try:
                sample_adata = data[modality_key]
            except IndexError:
                logger.error("When data is a MuData object, modality_key must be specified!")
                raise
        if isinstance(data, AnnData):
            sample_adata = data

        # Get model and effect selection types
        select_type = sample_adata.uns["scCODA_params"]["select_type"]
        model_type = sample_adata.uns["scCODA_params"]["model_type"]

        # If other than None for est_fdr is specified, recalculate intercept and effect DataFrames
        if isinstance(est_fdr, float):
            if est_fdr < 0 or est_fdr > 1:
                raise ValueError("est_fdr must be between 0 and 1!")
            else:
                _, eff_df = self.summary_prepare(sample_adata, est_fdr=est_fdr)  # type: ignore
        # otherwise, get pre-calculated DataFrames. Effect DataFrame is stitched together from varm
        else:
            covariates = sample_adata.uns["scCODA_params"]["covariate_names"]
            region_names = ["Global"]
            region_names.extend(sample_adata.uns["scCODA_params"]["region_names"])
            effect_dfs = []
            for region in region_names:
                for cov in covariates:
                    effect_dfs.append(sample_adata.varm[f"effect_df_{region}_{cov}"])
            effect_df = pd.concat(effect_dfs)
            effect_df.index = pd.MultiIndex.from_product(
                (covariates, sample_adata.var.index.tolist()), names=["Region", "Covariate", "Cell Type"]
            )

        out = eff_df["Final Parameter"] != 0
        out.rename("credible change")

        return out

    def _stackbar(  # pragma: no cover
        self,
        y: np.ndarray,
        type_names: list[str],
        title: str,
        level_names: list[str],
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        palette: ListedColormap | None = cm.tab20,
        show_legend: bool | None = True,
    ) -> plt.Axes:
        """Plots a stacked barplot for one (discrete) covariate.

        Typical use (only inside stacked_barplot): plot_one_stackbar(data.X, data.var.index, "xyz", data.obs.index)

        Args:
            y: The count data, collapsed onto the level of interest. i.e. a binary covariate has two rows,
               one for each group, containing the count mean of each cell type
            type_names: The names of all cell types
            title: Plot title, usually the covariate's name
            level_names: Names of the covariate's levels
            figsize: Figure size (matplotlib).
            dpi: Resolution in DPI (matplotlib).
            palette: The color map for the barplot.
            show_legend: If True, adds a legend.

        Returns:
            A :class:`~matplotlib.axes.Axes` object
        """
        n_bars, n_types = y.shape

        figsize = rcParams["figure.figsize"] if figsize is None else figsize

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)
        r = np.array(range(n_bars))
        sample_sums = np.sum(y, axis=1)

        barwidth = 0.85
        cum_bars = np.zeros(n_bars)

        for n in range(n_types):
            bars = [i / j * 100 for i, j in zip([y[k][n] for k in range(n_bars)], sample_sums, strict=False)]
            plt.bar(
                r,
                bars,
                bottom=cum_bars,
                color=palette(n % palette.N),
                width=barwidth,
                label=type_names[n],
                linewidth=0,
            )
            cum_bars += bars

        ax.set_title(title)
        if show_legend:
            ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1)
        ax.set_xticks(r)
        ax.set_xticklabels(level_names, rotation=45, ha="right")
        ax.set_ylabel("Proportion")

        return ax

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_stacked_barplot(  # pragma: no cover
        self,
        data: AnnData | MuData,
        feature_name: str,
        *,
        modality_key: str = "coda",
        palette: ListedColormap | None = cm.tab20,
        show_legend: bool | None = True,
        level_order: list[str] = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        show: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plots a stacked barplot for all levels of a covariate or all samples (if feature_name=="samples").

        Args:
            data: AnnData object or MuData object.
            feature_name: The name of the covariate to plot. If feature_name=="samples", one bar for every sample will be plotted
            modality_key: If data is a MuData object, specify which modality to use.
            figsize: Figure size.
            dpi: Dpi setting.
            palette: The matplotlib color map for the barplot.
            show_legend: If True, adds a legend.
            level_order: Custom ordering of bars on the x-axis.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> sccoda.plot_stacked_barplot(mdata, feature_name="samples")

        Preview:
            .. image:: /_static/docstring_previews/sccoda_stacked_barplot.png
        """
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data

        ct_names = data.var.index

        # option to plot one stacked barplot per sample
        if feature_name == "samples":
            if level_order:
                assert set(level_order) == set(data.obs.index), "level order is inconsistent with levels"
                data = data[level_order]
            self._stackbar(
                data.X,
                type_names=data.var.index,
                title="samples",
                level_names=data.obs.index,
                figsize=figsize,
                dpi=dpi,
                palette=palette,
                show_legend=show_legend,
            )
        else:
            # Order levels
            if level_order:
                assert set(level_order) == set(data.obs[feature_name]), "level order is inconsistent with levels"
                levels = level_order
            elif hasattr(data.obs[feature_name], "cat"):
                levels = data.obs[feature_name].cat.categories.to_list()
            else:
                levels = pd.unique(data.obs[feature_name])
            n_levels = len(levels)
            feature_totals = np.zeros([n_levels, data.X.shape[1]])

            for level in range(n_levels):
                l_indices = np.where(data.obs[feature_name] == levels[level])
                feature_totals[level] = np.sum(data.X[l_indices], axis=0)

            self._stackbar(
                feature_totals,
                type_names=ct_names,
                title=feature_name,
                level_names=levels,
                figsize=figsize,
                dpi=dpi,
                palette=palette,
                show_legend=show_legend,
            )

        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_effects_barplot(  # pragma: no cover
        self,
        data: AnnData | MuData,
        *,
        modality_key: str = "coda",
        covariates: str | list | None = None,
        parameter: Literal["Final Parameter"] = "Final Parameter",
        plot_facets: bool = True,
        plot_zero_covariate: bool = True,
        plot_zero_cell_type: bool = False,
        palette: str | ListedColormap | None = cm.tab20,
        level_order: list[str] = None,
        args_barplot: dict | None = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        show: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Barplot visualization for effects.

        The effect results for each covariate are shown as a group of barplots, with intra--group separation by cell types.
        The covariates groups can either be ordered along the x-axis of a single plot (plot_facets=False) or as plot facets (plot_facets=True).

        Args:
            data: AnnData object or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.
            covariates: The name of the covariates in data.obs to plot.
            parameter: The parameter in effect summary to plot.
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets.
            plot_zero_covariate: If True, plot covariate that have all zero effects. If False, do not plot.
            plot_zero_cell_type: If True, plot cell type that have zero effect. If False, do not plot.
            figsize: Figure size.
            dpi: Figure size.
            palette: The seaborn color map for the barplot.
            level_order: Custom ordering of bars on the x-axis.
            args_barplot: Arguments passed to sns.barplot.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> sccoda.plot_effects_barplot(mdata)

        Preview:
            .. image:: /_static/docstring_previews/sccoda_effects_barplot.png
        """
        if args_barplot is None:
            args_barplot = {}
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        # Get covariate names from adata, partition into those with nonzero effects for min. one cell type/no cell types
        covariate_names = data.uns["scCODA_params"]["covariate_names"]
        if covariates is not None:
            if isinstance(covariates, str):
                covariates = [covariates]
            partial_covariate_names = [
                covariate_name
                for covariate_name in covariate_names
                if any(covariate in covariate_name for covariate in covariates)
            ]
            covariate_names = partial_covariate_names
        covariate_names_non_zero = [
            covariate_name
            for covariate_name in covariate_names
            if data.varm[f"effect_df_{covariate_name}"][parameter].any()
        ]
        covariate_names_zero = list(set(covariate_names) - set(covariate_names_non_zero))
        if not plot_zero_covariate:
            covariate_names = covariate_names_non_zero

        # set up df for plotting
        plot_df = pd.concat(
            [data.varm[f"effect_df_{covariate_name}"][parameter] for covariate_name in covariate_names],
            axis=1,
        )
        plot_df.columns = covariate_names
        plot_df = pd.melt(plot_df, ignore_index=False, var_name="Covariate")

        plot_df = plot_df.reset_index()

        if len(covariate_names_zero) != 0:
            if plot_facets:
                if plot_zero_covariate and not plot_zero_cell_type:
                    for covariate_name_zero in covariate_names_zero:
                        new_row = {
                            "Covariate": covariate_name_zero,
                            "Cell Type": "zero",
                            "value": 0,
                        }
                        plot_df = pd.concat([plot_df, pd.DataFrame([new_row])], ignore_index=True)
                    plot_df["covariate_"] = pd.Categorical(plot_df["Covariate"], covariate_names)
                    plot_df = plot_df.sort_values(["covariate_"])
        if not plot_zero_cell_type:
            cell_type_names_zero = [
                name
                for name in plot_df["Cell Type"].unique()
                if (plot_df[plot_df["Cell Type"] == name]["value"] == 0).all()
            ]
            plot_df = plot_df[~plot_df["Cell Type"].isin(cell_type_names_zero)]

        # If plot as facets, create a FacetGrid and map barplot to it.
        if plot_facets:
            if isinstance(palette, ListedColormap):
                palette = np.array([palette(i % palette.N) for i in range(len(plot_df["Cell Type"].unique()))]).tolist()
            if figsize is not None:
                height = figsize[0]
                aspect = np.round(figsize[1] / figsize[0], 2)
            else:
                height = 3
                aspect = 2

            g = sns.FacetGrid(
                plot_df,
                col="Covariate",
                sharey=True,
                sharex=False,
                height=height,
                aspect=aspect,
            )

            g.map(
                sns.barplot,
                "Cell Type",
                "value",
                palette=palette,
                order=level_order,
                **args_barplot,
            )
            g.set_xticklabels(rotation=90)
            g.set(ylabel=parameter)
            axes = g.axes.flatten()
            for i, ax in enumerate(axes):
                ax.set_title(covariate_names[i])
                if len(ax.get_xticklabels()) < 5:
                    ax.set_aspect(10 / len(ax.get_xticklabels()))
                    if len(ax.get_xticklabels()) == 1:
                        if ax.get_xticklabels()[0]._text == "zero":
                            ax.set_xticks([])

        # If not plot as facets, call barplot to plot cell types on the x-axis.
        else:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)
            if len(covariate_names) == 1:
                if isinstance(palette, ListedColormap):
                    palette = np.array(
                        [palette(i % palette.N) for i in range(len(plot_df["Cell Type"].unique()))]
                    ).tolist()
                sns.barplot(
                    data=plot_df,
                    x="Cell Type",
                    y="value",
                    hue="x",
                    palette=palette,
                    ax=ax,
                )
                ax.set_title(covariate_names[0])
            else:
                if isinstance(palette, ListedColormap):
                    palette = np.array([palette(i % palette.N) for i in range(len(covariate_names))]).tolist()
                sns.barplot(
                    data=plot_df,
                    x="Cell Type",
                    y="value",
                    hue="Covariate",
                    palette=palette,
                    ax=ax,
                )
            cell_types = pd.unique(plot_df["Cell Type"])
            ax.set_xticklabels(cell_types, rotation=90)

        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_boxplots(  # pragma: no cover
        self,
        data: AnnData | MuData,
        feature_name: str,
        *,
        modality_key: str = "coda",
        y_scale: Literal["relative", "log", "log10", "count"] = "relative",
        plot_facets: bool = False,
        add_dots: bool = False,
        cell_types: list | None = None,
        args_boxplot: dict | None = None,
        args_swarmplot: dict | None = None,
        palette: str | None = "Blues",
        show_legend: bool | None = True,
        level_order: list[str] = None,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        show: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Grouped boxplot visualization.

         The cell counts for each cell type are shown as a group of boxplots
         with intra--group separation by a covariate from data.obs.

        Args:
            data: AnnData object or MuData object
            feature_name: The name of the feature in data.obs to plot
            modality_key: If data is a MuData object, specify which modality to use.
            y_scale: Transformation to of cell counts. Options: "relative" - Relative abundance, "log" - log(count),
                     "log10" - log10(count), "count" - absolute abundance (cell counts).
            plot_facets: If False, plot cell types on the x-axis. If True, plot as facets.
            add_dots: If True, overlay a scatterplot with one dot for each data point.
            cell_types: Subset of cell types that should be plotted.
            args_boxplot: Arguments passed to sns.boxplot.
            args_swarmplot: Arguments passed to sns.swarmplot.
            figsize: Figure size.
            dpi: Dpi setting.
            palette: The seaborn color map for the barplot.
            show_legend: If True, adds a legend.
            level_order: Custom ordering of bars on the x-axis.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> sccoda.plot_boxplots(mdata, feature_name="condition", add_dots=True)

        Preview:
            .. image:: /_static/docstring_previews/sccoda_boxplots.png
        """
        if args_boxplot is None:
            args_boxplot = {}
        if args_swarmplot is None:
            args_swarmplot = {}
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        # y scale transformations
        if y_scale == "relative":
            sample_sums = np.sum(data.X, axis=1, keepdims=True)
            X = data.X / sample_sums
            value_name = "Proportion"
        # add pseudocount 0.5 if using log scale
        elif y_scale == "log":
            X = data.X.copy()
            X[X == 0] = 0.5
            X = np.log(X)
            value_name = "log(count)"
        elif y_scale == "log10":
            X = data.X.copy()
            X[X == 0] = 0.5
            X = np.log(X)
            value_name = "log10(count)"
        elif y_scale == "count":
            X = data.X
            value_name = "count"
        else:
            raise ValueError("Invalid y_scale transformation")

        count_df = pd.DataFrame(X, columns=data.var.index, index=data.obs.index).merge(
            data.obs[feature_name], left_index=True, right_index=True
        )
        plot_df = pd.melt(count_df, id_vars=feature_name, var_name="Cell type", value_name=value_name)
        if cell_types is not None:
            plot_df = plot_df[plot_df["Cell type"].isin(cell_types)]

        # Currently disabled because the latest statsannotations does not support the latest seaborn.
        # We had to drop the dependency.
        # Get credible effects results from model
        # if draw_effects:
        #     if model is not None:
        #         credible_effects_df = model.credible_effects(data, modality_key).to_frame().reset_index()
        #     else:
        #         print("[bold yellow]Specify a tasCODA model to draw effects")
        #     credible_effects_df[feature_name] = credible_effects_df["Covariate"].str.removeprefix(f"{feature_name}[T.")
        #     credible_effects_df[feature_name] = credible_effects_df[feature_name].str.removesuffix("]")
        #     credible_effects_df = credible_effects_df[credible_effects_df["Final Parameter"]]

        # If plot as facets, create a FacetGrid and map boxplot to it.
        if plot_facets:
            if level_order is None:
                level_order = pd.unique(plot_df[feature_name])

            K = X.shape[1]

            if figsize is not None:
                height = figsize[0]
                aspect = np.round(figsize[1] / figsize[0], 2)
            else:
                height = 3
                aspect = 2

            g = sns.FacetGrid(
                plot_df,
                col="Cell type",
                sharey=False,
                col_wrap=int(np.floor(np.sqrt(K))),
                height=height,
                aspect=aspect,
            )
            g.map(
                sns.boxplot,
                feature_name,
                value_name,
                palette=palette,
                order=level_order,
                **args_boxplot,
            )

            if add_dots:
                if "hue" in args_swarmplot:
                    hue = args_swarmplot.pop("hue")
                else:
                    hue = None

                if hue is None:
                    g.map(
                        sns.swarmplot,
                        feature_name,
                        value_name,
                        color="black",
                        order=level_order,
                        **args_swarmplot,
                    ).set_titles("{col_name}")
                else:
                    g.map(
                        sns.swarmplot,
                        feature_name,
                        value_name,
                        hue,
                        order=level_order,
                        **args_swarmplot,
                    ).set_titles("{col_name}")

        # If not plot as facets, call boxplot to plot cell types on the x-axis.
        else:
            if level_order:
                args_boxplot["hue_order"] = level_order
                args_swarmplot["hue_order"] = level_order

            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

            ax = sns.boxplot(
                x="Cell type",
                y=value_name,
                hue=feature_name,
                data=plot_df,
                fliersize=1,
                palette=palette,
                ax=ax,
                **args_boxplot,
            )

            # Currently disabled because the latest statsannotations does not support the latest seaborn.
            # We had to drop the dependency.
            # if draw_effects:
            #     pairs = [
            #         [(row["Cell Type"], row[feature_name]), (row["Cell Type"], "Control")]
            #         for _, row in credible_effects_df.iterrows()
            #     ]
            #     annot = Annotator(ax, pairs, data=plot_df, x="Cell type", y=value_name, hue=feature_name)
            #     annot.configure(test=None, loc="outside", color="red", line_height=0, verbose=False)
            #     annot.set_custom_annotations([row[feature_name] for _, row in credible_effects_df.iterrows()])
            #     annot.annotate()

            if add_dots:
                sns.swarmplot(
                    x="Cell type",
                    y=value_name,
                    data=plot_df,
                    hue=feature_name,
                    ax=ax,
                    dodge=True,
                    palette="dark:black",
                    **args_swarmplot,
                )

            cell_types = pd.unique(plot_df["Cell type"])
            ax.set_xticklabels(cell_types, rotation=90)

            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                handout = []
                labelout = []
                for h, l in zip(handles, labels, strict=False):
                    if l not in labelout:
                        labelout.append(l)
                        handout.append(h)
                ax.legend(
                    handout,
                    labelout,
                    loc="upper left",
                    bbox_to_anchor=(1, 1),
                    ncol=1,
                    title=feature_name,
                )

        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_rel_abundance_dispersion_plot(  # pragma: no cover
        self,
        data: AnnData | MuData,
        *,
        modality_key: str = "coda",
        abundant_threshold: float | None = 0.9,
        default_color: str | None = "Grey",
        abundant_color: str | None = "Red",
        label_cell_types: bool = True,
        figsize: tuple[float, float] | None = None,
        dpi: int | None = 100,
        ax: plt.Axes | None = None,
        show: bool = True,
        return_fig: bool = False,
    ) -> Figure | None:
        """Plots total variance of relative abundance versus minimum relative abundance of all cell types for determination of a reference cell type.

        If the count of the cell type is larger than 0 in more than abundant_threshold percent of all samples, the cell type will be marked in a different color.

        Args:
            data: AnnData or MuData object.
            modality_key: If data is a MuData object, specify which modality to use.
            abundant_threshold: Presence threshold for abundant cell types.
            default_color: Bar color for all non-minimal cell types.
            abundant_color: Bar color for cell types with abundant percentage larger than abundant_threshold.
            label_cell_types: Label dots with cell type names.
            figsize: Figure size.
            dpi: Dpi setting.
            ax: A matplotlib axes object. Only works if plotting a single component.
            {common_plot_args}

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> haber_cells = pt.dt.haber_2017_regions()
            >>> sccoda = pt.tl.Sccoda()
            >>> mdata = sccoda.load(haber_cells, type="cell_level", generate_sample_level=True, cell_type_identifier="cell_label", \
                sample_identifier="batch", covariate_obs=["condition"])
            >>> mdata = sccoda.prepare(mdata, formula="condition", reference_cell_type="Endocrine")
            >>> sccoda.run_nuts(mdata, num_warmup=100, num_samples=1000, rng_key=42)
            >>> sccoda.plot_rel_abundance_dispersion_plot(mdata)

        Preview:
            .. image:: /_static/docstring_previews/sccoda_rel_abundance_dispersion_plot.png
        """
        if isinstance(data, MuData):
            data = data[modality_key]
        if isinstance(data, AnnData):
            data = data
        if ax is None:
            _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        rel_abun = data.X / np.sum(data.X, axis=1, keepdims=True)

        percent_zero = np.sum(data.X == 0, axis=0) / data.X.shape[0]
        nonrare_ct = np.where(percent_zero < 1 - abundant_threshold)[0]

        # select reference
        cell_type_disp = np.var(rel_abun, axis=0) / np.mean(rel_abun, axis=0)

        is_abundant = [x in nonrare_ct for x in range(data.X.shape[1])]

        # Scatterplot
        plot_df = pd.DataFrame(
            {
                "Total dispersion": cell_type_disp,
                "Cell type": data.var.index,
                "Presence": 1 - percent_zero,
                "Is abundant": is_abundant,
            }
        )

        if len(np.unique(plot_df["Is abundant"])) > 1:
            palette = [default_color, abundant_color]
        elif np.unique(plot_df["Is abundant"]) == [False]:
            palette = [default_color]
        else:
            palette = [abundant_color]

        ax = sns.scatterplot(
            data=plot_df,
            x="Presence",
            y="Total dispersion",
            hue="Is abundant",
            palette=palette,
            ax=ax,
        )

        # Text labels for abundant cell types

        abundant_df = plot_df.loc[plot_df["Is abundant"], :]

        def label_point(x, y, val, ax):
            a = pd.concat({"x": x, "y": y, "val": val}, axis=1)
            texts = [
                ax.text(
                    point["x"],
                    point["y"],
                    str(point["val"]),
                )
                for i, point in a.iterrows()
            ]
            adjust_text(texts)

        if label_cell_types:
            label_point(
                abundant_df["Presence"],
                abundant_df["Total dispersion"],
                abundant_df["Cell type"],
                plt.gca(),
            )

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), ncol=1, title="Is abundant")

        if show:
            plt.show()
        if return_fig:
            return plt.gcf()
        return None

    @_doc_params(common_plot_args=doc_common_plot_args)
    def plot_effects_umap(  # pragma: no cover
        self,
        mdata: MuData,
        effect_name: str | list | None,
        cluster_key: str,
        *,
        modality_key_1: str = "rna",
        modality_key_2: str = "coda",
        color_map: Colormap | str | None = None,
        palette: str | Sequence[str] | None = None,
        ax: Axes = None,
        show: bool = True,
        return_fig: bool = False,
        **kwargs,
    ) -> Figure | None:
        """Plot a UMAP visualization colored by effect strength.

        Effect results in .varm of aggregated sample-level AnnData (default is data['coda']) are assigned to cell-level AnnData
        (default is data['rna']) depending on the cluster they were assigned to.

        Args:
            mdata: MuData object.
            effect_name: The name of the effect results in .varm of aggregated sample-level AnnData to plot
            cluster_key: The cluster information in .obs of cell-level AnnData (default is data['rna']).
                         To assign cell types' effects to original cells.
            modality_key_1: Key to the cell-level AnnData in the MuData object.
            modality_key_2: Key to the aggregated sample-level AnnData object in the MuData object.
            color_map: The color map to use for plotting.
            palette: The color palette to use for plotting.
            ax: A matplotlib axes object. Only works if plotting a single component.
            {common_plot_args}
            **kwargs: All other keyword arguments are passed to `scanpy.plot.umap()`

        Returns:
            If `return_fig` is `True`, returns the figure, otherwise `None`.

        Examples:
            >>> import pertpy as pt
            >>> import scanpy as sc
            >>> import schist
            >>> adata = pt.dt.haber_2017_regions()
            >>> sc.pp.neighbors(adata)
            >>> schist.inference.nested_model(adata, n_init=100, random_seed=5678)
            >>> tasccoda_model = pt.tl.Tasccoda()
            >>> tasccoda_data = tasccoda_model.load(adata, type="cell_level",
            >>>                 cell_type_identifier="nsbm_level_1",
            >>>                 sample_identifier="batch", covariate_obs=["condition"],
            >>>                 levels_orig=["nsbm_level_4", "nsbm_level_3", "nsbm_level_2", "nsbm_level_1"],
            >>>                 add_level_name=True)
            >>> tasccoda_model.prepare(
            >>>     tasccoda_data,
            >>>     modality_key="coda",
            >>>     reference_cell_type="18",
            >>>     formula="condition",
            >>>     pen_args=dict(phi=0, lambda_1=3.5),
            >>>     tree_key="tree"
            >>> )
            >>> tasccoda_model.run_nuts(
            ...     tasccoda_data, modality_key="coda", rng_key=1234, num_samples=10000, num_warmup=1000
            ... )
            >>> tasccoda_model.run_nuts(
            ...     tasccoda_data, modality_key="coda", rng_key=1234, num_samples=10000, num_warmup=1000
            ... )
            >>> sc.tl.umap(tasccoda_data["rna"])
            >>> tasccoda_model.plot_effects_umap(tasccoda_data,
            >>>                         effect_name=["effect_df_condition[T.Salmonella]",
            >>>                                      "effect_df_condition[T.Hpoly.Day3]",
            >>>                                      "effect_df_condition[T.Hpoly.Day10]"],
            >>>                                       cluster_key="nsbm_level_1",
            >>>                         )

        Preview:
            .. image:: /_static/docstring_previews/tasccoda_effects_umap.png
        """
        # TODO: Add effect_name parameter and cluster_key and test the example
        data_rna = mdata[modality_key_1]
        data_coda = mdata[modality_key_2]
        if isinstance(effect_name, str):
            effect_name = [effect_name]
        for _, effect in enumerate(effect_name):
            data_rna.obs[effect] = [data_coda.varm[effect].loc[f"{c}", "Effect"] for c in data_rna.obs[cluster_key]]
        if kwargs.get("vmin"):
            vmin = kwargs["vmin"]
            kwargs.pop("vmin")
        else:
            vmin = min(data_rna.obs[effect].min() for _, effect in enumerate(effect_name))
        if kwargs.get("vmax"):
            vmax = kwargs["vmax"]
            kwargs.pop("vmax")
        else:
            vmax = max(data_rna.obs[effect].max() for _, effect in enumerate(effect_name))

        fig = sc.pl.umap(
            data_rna,
            color=effect_name,
            vmax=vmax,
            vmin=vmin,
            palette=palette,
            color_map=color_map,
            return_fig=return_fig,
            ax=ax,
            show=False,
            **kwargs,
        )

        if show:
            plt.show()
        if return_fig:
            return fig
        return None


def get_a(
    tree: tt.core.ToyTree,
) -> tuple[np.ndarray, int]:
    """Calculate ancestor matrix from a toytree tree

    Args:
        tree: A toytree tree object.

    Returns:
        Ancestor matrix and number of nodes without root node
        A
            Ancestor matrix (numpy array)
        T
            number of nodes in the tree, excluding the root node
    """
    # Builds ancestor matrix
    n_tips = tree.ntips
    n_nodes = tree.nnodes

    A_ = np.zeros((n_tips, n_nodes))

    for i in np.arange(n_nodes):
        leaves_i = list(set(tree.get_node_descendant_idxs(i)) & set(np.arange(n_tips)))
        A_[leaves_i, i] = 1

    # collapsed trees may have scrambled leaves.
    # Therefore, we permute the rows of A such that they are in the original order. Columns (nodes) stay permuted.
    scrambled_leaves = list(tree.get_node_values("idx_orig", True, True)[-n_tips:])
    scrambled_leaves.reverse()
    if scrambled_leaves[0] == "":
        scrambled_leaves = list(np.arange(0, n_tips, 1))

    A = np.zeros((n_tips, n_nodes))
    for r in range(n_tips):
        A[scrambled_leaves[r], :] = A_[r, :]
    A = A[:, :-1]

    return A, n_nodes - 1


def collapse_singularities(tree: tt.core.ToyTree) -> tt.core.ToyTree:
    """Collapses (deletes) nodes in a toytree tree that are singularities (have only one child).

    Args:
        tree: A toytree tree object

    Returns:
        A toytree tree without singularities

        tree_new
            A toytree tree
    """
    A, _ = get_a(tree)
    A_T = A.T
    unq, count = np.unique(A_T, axis=0, return_counts=True)

    repeated_idx = []
    for repeated_group in unq[count > 1]:
        repeated_idx.append(np.argwhere(np.all(A_T == repeated_group, axis=1)).ravel())

    nodes_to_delete = [i for idx in repeated_idx for i in idx[1:]]

    # _coords.update() scrambles the idx of leaves. Therefore, keep track of it here
    tree_new = tree.copy()
    for node in tree_new.treenode.traverse():
        node.add_feature("idx_orig", node.idx)

    for n in nodes_to_delete:
        node = tree_new.idx_dict[n]
        node.delete()

    tree_new._coords.update()

    # remove node artifacts
    for k in list(tree_new.idx_dict):
        if k >= tree_new.nnodes:
            tree_new.idx_dict.pop(k)

    return tree_new


def traverse(df_, a, i, innerl):
    """
    Helper function for df2newick
    Adapted from https://stackoverflow.com/questions/15343338/how-to-convert-a-data-frame-to-tree-structure-object-such-as-dendrogram
    """
    if i + 1 < df_.shape[1]:
        a_inner = pd.unique(df_.loc[np.where(df_.iloc[:, i] == a)].iloc[:, i + 1])

        desc = []
        for b in a_inner:
            desc.append(traverse(df_, b, i + 1, innerl))
        if innerl:
            il = a
        else:
            il = ""
        out = f"({','.join(desc)}){il}"
    else:
        out = a

    return out


def df2newick(df: pd.DataFrame, levels: list[str], inner_label: bool = True) -> str:
    """Converts a pandas DataFrame with hierarchical information into a newick string.

    Adapted from https://stackoverflow.com/questions/15343338/how-to-convert-a-data-frame-to-tree-structure-object-such-as-dendrogram

    Args:
        df: Pandas DataFrame that has one row for each leaf of the tree and columns that indicate a hierarchical ordering. See the tascCODA tutorial for an example.
        levels: List that indicates how the columns in df are ordered as tree levels. Begins with the root level, ends with the leaf level
        inner_label: Indicator whether labels for inner nodes should be included in the newick string

    Returns:
        Newick string describing the tree structure from df
    """
    df_tax = df.loc[:, [x for x in levels if x in df.columns]]

    alevel = pd.unique(df_tax.iloc[:, 0])
    strs = []
    for a in alevel:
        strs.append(traverse(df_tax, a, 0, inner_label))

    newick = f"({','.join(strs)});"
    return newick


def get_a_2(
    tree: Tree,
    leaf_order: list[str] = None,
    node_order: list[str] = None,
) -> tuple[np.ndarray, int]:
    """Calculate ancestor matrix from a ete3 tree.

    Args:
        tree: A ete3 tree object.
        leaf_order: List of leaf names how they should appear as the rows of the ancestor matrix.
                    If None, the ordering will be as in `tree.iter_leaves()`
        node_order: List of node names how they should appear as the columns of the ancestor matrix
                    If None, the ordering will be as in `tree.iter_descendants()`

    Returns:
            Ancestor matrix and number of nodes without root node

        A
            Ancestor matrix (numpy array)
        T
            number of nodes in the tree, excluding the root node
    """
    try:
        import ete3 as ete
    except ImportError:
        raise ImportError(
            "To use tasccoda please install additional dependencies as `pip install pertpy[coda]`"
        ) from None

    n_tips = len(tree.get_leaves())
    n_nodes = len(tree.get_descendants())

    node_names = [n.name for n in tree.iter_descendants()]
    duplicates = [x for x in node_names if node_names.count(x) > 1]
    if len(duplicates) > 0:
        raise ValueError(f"Tree nodes have duplicate names: {duplicates}. Make sure that node names are unique!")

    # Initialize ancestor matrix
    A_ = pd.DataFrame(np.zeros((n_tips, n_nodes)))
    A_.index = tree.get_leaf_names()
    A_.columns = [n.name for n in tree.iter_descendants()]

    # Fill in 1's for all connections
    for node in tree.iter_descendants():
        for leaf in tree.get_leaves():
            if leaf in node.get_leaves():
                A_.loc[leaf.name, node.name] = 1

    # Order rows and columns
    if leaf_order is not None:
        A_ = A_.loc[leaf_order]
    if node_order is not None:
        A_ = A_[node_order]
    A_ = np.array(A_)

    return A_, n_nodes


def collapse_singularities_2(tree: Tree) -> Tree:
    """Collapses (deletes) nodes in a ete3 tree that are singularities (have only one child).

    Args:
        tree: A ete3 tree object

    Returns:
        A ete3 tree without singularities.
    """
    for node in tree.iter_descendants():
        if len(node.get_children()) == 1:
            node.delete()

    return tree


def linkage_to_newick(
    Z: np.ndarray,
    labels: list[str],
) -> str:
    """Convert a linkage matrix to newick tree string.

    Adapted from https://stackoverflow.com/a/31878514/20299702.

    Args:
        Z: linkage matrix
        labels: leaf labels

    Returns:
        str: Newick string describing the tree structure
    """
    tree = sp_hierarchy.to_tree(Z, False)

    def build_newick(node, newick, parentdist, leaf_names):
        if node.is_leaf():
            return f"{leaf_names[node.id]}:{(parentdist - node.dist) / 2}{newick}"
        else:
            if len(newick) > 0:
                newick = f"):{(parentdist - node.dist) / 2}{newick}"
            else:
                newick = ");"
            newick = build_newick(node.get_left(), newick, node.dist, leaf_names)
            newick = build_newick(node.get_right(), f",{newick}", node.dist, leaf_names)
            newick = f"({newick}"
            return newick

    return build_newick(tree, "", tree.dist, labels)


def import_tree(
    data: AnnData | MuData,
    modality_1: str = None,
    modality_2: str = None,
    dendrogram_key: str = None,
    levels_orig: list[str] = None,
    levels_agg: list[str] = None,
    add_level_name: bool = True,
    key_added: str = "tree",
):
    """Generate ete tree for tascCODA models from dendrogram information or cell-level observations.

    Trees can either be generated from scipy dendrogram information e.g. from scanpy.tl.dendrogram,
    or from hierarchical information for each cell type - either saved in `.obs` of the cell-level data object, or in `.var` of the aggregated data.

    Notes:
    - Either `dendrogram_key`, `levels_orig` or `levels_agg` must be not None. Priority is `dendrogram_key` -> `levels_orig` -> `levels_agg`
    - If `data` is a MuData object, `modality_1` and `modality_2` must be specified
    - The node names of the generated tree must be unique. Often, setting `add_level_name=True` is enough to achieve that.

    Args:
        data: A tascCODA-compatible data object.
        modality_1: If `data` is MuData, specify the modality name to the original cell level anndata object.
        modality_2: If `data` is MuData, specify the modality name to the aggregated level anndata object.
        dendrogram_key: Key to the scanpy.tl.dendrogram result in `.uns` of original cell level anndata object.
        levels_orig: List that indicates which columns in `.obs` of the original data correspond to tree levels. The list must begin with the root level, and end with the leaf level.
        levels_agg: List that indicates which columns in `.var` of the aggregated data correspond to tree levels. The list must begin with the root level, and end with the leaf level.
        add_level_name: If True, internal nodes in the tree will be named as "{level_name}_{node_name}" instead of just {level_name}.
        key_added: If not specified, the tree is stored in .uns[‘tree’]. If `data` is AnnData, save tree in `data`.
                   If `data` is MuData, save tree in data[modality_2].

    Returns:
        Updates data with the following:

        See `key_added` parameter description for the storage path of tree.

        tree: A ete3 tree object.
    """
    try:
        import ete3 as ete
    except ImportError:
        raise ImportError(
            "To use tasccoda please install additional dependencies as `pip install pertpy[coda]`"
        ) from None

    if isinstance(data, MuData):
        try:
            data_1 = data[modality_1]
            data_2 = data[modality_2]
        except KeyError as name:
            logger.error(f"No {name} slot in MuData")
            raise
        except IndexError:
            logger.error("Please specify modality_1 and modality_2 to indicate modalities in MuData")
            raise
    else:
        data_1 = data
        data_2 = data

    if dendrogram_key is not None:
        newick = linkage_to_newick(
            data_1.uns["dendrogram_cell_label"]["linkage"],
            labels=data_1.uns["dendrogram_cell_label"]["categories_ordered"],
        )
        tree = ete.Tree(newick, format=1)
        node_id = 0
        for n in tree.iter_descendants():
            if not n.is_leaf():
                n.name = str(node_id)
                node_id += 1
    elif levels_orig is not None:
        newick = df2newick(data_1.obs.reset_index(), levels=levels_orig)
        tree = ete.Tree(newick, format=8)
        if add_level_name:
            for n in tree.iter_descendants():
                if not n.is_leaf():
                    dist = n.get_distance(n, tree)
                    n.name = f"{levels_orig[int(dist) - 1]}_{n.name}"
    elif levels_agg is not None:
        newick = df2newick(data_2.var.reset_index(), levels=levels_agg)
        tree = ete.Tree(newick, format=8)
        if add_level_name:
            for n in tree.iter_descendants():
                if not n.is_leaf():
                    dist = n.get_distance(n, tree)
                    n.name = f"{levels_agg[int(dist) - 1]}_{n.name}"
    else:
        raise ValueError("Either dendrogram_key, levels_orig or levels_agg must be specified!")

    node_names = [n.name for n in tree.iter_descendants()]
    duplicates = {x for x in node_names if node_names.count(x) > 1}
    if len(duplicates) > 0:
        raise ValueError(f"Tree nodes have duplicate names: {duplicates}. Make sure that node names are unique!")

    data_2.uns[key_added] = tree


def from_scanpy(
    adata: AnnData,
    cell_type_identifier: str,
    sample_identifier: str | list[str],
    covariate_uns: str | None = None,
    covariate_obs: list[str] | None = None,
    covariate_df: pd.DataFrame | None = None,
) -> AnnData:
    """Creates a compositional analysis dataset from a single AnnData object, as it is produced by e.g. scanpy.

    The anndata object needs to have a column in adata.obs that contains the cell type assignment.
    Further, it must contain one column or a set of columns (e.g. subject id, treatment, disease status) that uniquely identify each (statistical) sample.
    Further covariates (e.g. subject age) can either be specified via additional column names in adata.obs, a key in adata.uns, or as a separate DataFrame.

    NOTE: The order of samples in the returned dataset is determined by the first occurrence of cells from each sample in `adata`

    Args:
        adata: An anndata object from scanpy
        cell_type_identifier: column name in adata.obs that specifies the cell types
        sample_identifier: column name or list of column names in adata.obs that uniquely identify each sample
        covariate_uns: key for adata.uns, where covariate values are stored
        covariate_obs: list of column names in adata.obs, where covariate values are stored.
                       Note: If covariate values are not unique for a value of sample_identifier, this covariate will be skipped.
        covariate_df: DataFrame with covariates

    Returns:
        AnnData: A data set with cells aggregated to the (sample x cell type) level
    """
    sample_identifier = [sample_identifier] if isinstance(sample_identifier, str) else sample_identifier
    covariate_obs = list(set(covariate_obs or []) | set(sample_identifier))

    if isinstance(sample_identifier, list):
        adata.obs["scCODA_sample_id"] = adata.obs[sample_identifier].agg("-".join, axis=1)
        sample_identifier = "scCODA_sample_id"

    groups = adata.obs.value_counts([sample_identifier, cell_type_identifier])
    ct_count_data = groups.unstack(level=cell_type_identifier).fillna(0)
    covariate_df_ = pd.DataFrame(index=ct_count_data.index)

    if covariate_uns is not None:
        covariate_df_uns = pd.DataFrame(adata.uns[covariate_uns], index=ct_count_data.index)
        covariate_df_ = pd.concat([covariate_df_, covariate_df_uns], axis=1)

    if covariate_obs:
        unique_check = adata.obs.groupby(sample_identifier).nunique()
        for c in covariate_obs.copy():
            if unique_check[c].max() != 1:
                logger.warning(f"Covariate {c} has non-unique values for batch! Skipping...")
                covariate_obs.remove(c)
        if covariate_obs:
            covariate_df_obs = adata.obs.groupby(sample_identifier).first()[covariate_obs]
            covariate_df_ = pd.concat([covariate_df_, covariate_df_obs], axis=1)

    if covariate_df is not None:
        if set(covariate_df.index) != set(ct_count_data.index):
            raise ValueError("Mismatch between sample names in anndata and covariate_df!")
        covariate_df_ = pd.concat([covariate_df_, covariate_df.reindex(ct_count_data.index)], axis=1)

    var_dat = ct_count_data.sum().rename("n_cells").to_frame()
    var_dat.index = var_dat.index.astype(str)
    covariate_df_.index = covariate_df_.index.astype(str)

    return AnnData(X=ct_count_data.values, var=var_dat, obs=covariate_df_)
