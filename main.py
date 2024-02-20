"""Solving the example problem"""
import pathlib
import typing
import pulp
import tqdm
import pandas as pd

from mpisppy.opt.ef import ExtensiveForm
import mpisppy.utils.sputils as sputils
import pyomo.environ as pyo
from scipy.stats import burr12


# Constants.
THIS_DIR = pathlib.Path(__file__).parent.resolve()
DATA_DIR = THIS_DIR / "data"
DATA_FILE = DATA_DIR / "input_SUA.xlsx"


# %% Helper functions.
def build_robust_optimization_pulp_model(
    df_var: pd.DataFrame,
    df_p: pd.DataFrame,
    substitutability_limit: typing.List[float],
    mu: float = 0.3,
    epsilon: float = 0.1,
) -> pulp.LpProblem:
    """
    Define Pulp model from the given data.

    Args:
        df_p: dataframe that includes information about variables. This
            dataframe has the following columns: Demand, Margin, Capacity,
            Variance group, Substitutability group.
        df_var: Dataframe including distribution parameters for demand variance
            groups.
        substitutability_limit:
        mu: Adjustable macro target percentage.
        epsilon: Robust optimization's robustness parameter.

    Returns:
        Pulp model.
    """
    # checking inputs.
    if not df_p.shape[0] >= 1:
        raise ValueError('Wrong dimension for products table.')
    if not df_variance_group.shape[0] >= 1:
        raise ValueError('Wrong dimension for variance table.')

    products = df_p.index.tolist()
    variance_groups = df_var['Demand Var Group'].tolist()

    # Initialize the problem
    prob = pulp.LpProblem(
        name="Robust_Optimization_Problem",
        sense=pulp.LpMaximize,
    )

    # Define decision variables
    num_products = len(products)
    surpluses = [
        pulp.LpVariable(name=f"S_{p}", lowBound=0) for p in range(num_products)
    ]

    # Define objective function
    objective = pulp.lpSum(
        df_p['Margin'][p] * surpluses[p] for p in range(num_products)
    )
    prob += objective

    # Add constraints
    for p in range(num_products):
        # Find the variance_group for product p
        variance_group = df_p['Variance group'][p]

        # Retrieve parameters from variance_groups for Burr12 distribution
        c = df_var['c'][variance_group]
        loc = df_var['loc'][variance_group]
        d = df_var['d'][variance_group]
        scale = df_var['scale'][variance_group]
        variance = burr12(c, d, loc, scale).var()

        # Add the constraint
        prob += (df_p['Demand'][p] * (1 + surpluses[p]) >=
                    (1 + variance) * df_p['Demand'][p]
                ), f"distribution_constraint_{p}"

        # Capacity constraint
        prob += surpluses[p] <= df_p["Capacity"][p], f"Capacity_Constraint_{p}"
        prob += ((1 + surpluses[p]) * df_p["Demand"][p] <=
                    df_p["Demand"][p] * (1 + epsilon)
                ), f"Robust_Upper_Constraint_{p}"
        prob += ((1 + surpluses[p]) * df_p["Demand"][p] >=
                    df_p["Demand"][p] * (1 - epsilon)
                ), f"Robust_Lower_Constraint_{p}"

    # Global aggregate surplus constraint
    prob += (pulp.lpSum(surpluses) <=
                mu * pulp.lpSum(df_p["Demand"])
            ), "Aggregate_Surplus_Constraint"

    # Group-specific substitutability group constraints
    for g in variance_groups:
        group_products = [
            p for p in products if df_p['Substitutability group'][p] == g
        ]
        prob += pulp.lpSum(
                surpluses[p] * df_p["Demand"][p] for p in group_products
            ) >= substitutability_limit[g] * pulp.lpSum(
                df_p["Demand"][p] for p in group_products
            ), f"Substitutability_Group_Constraint_{g}"
    return prob


def build_robust_optimization_pyomo_model(
    df_p: pd.DataFrame,
    df_var: pd.DataFrame,
    substitutability_limit: typing.List[float],
    mu: float = 0.3,
) -> pyo.ConcreteModel:
    """
    Define Pyomo model from the given data.

    Args:
        df_p: dataframe that includes information about variables. This
            dataframe has the following columns: Demand, Margin, Capacity,
            Variance group, Substitutability group.
        df_var: Dataframe including distribution parameters for demand variance
            groups.
        substitutability_limit: Substitutability limit for the substitutaion
            constraint. The idea is that we need to have more than this limit
            from products in a group.
        mu: Adjustable macro target percentage.
    
    Returns:
        Pyomo model with objective function and constraints.

    """
    # checking inputs.
    if not df_p.shape[0] >= 1:
        raise ValueError('Wrong dimension for products table.')
    if not df_variance_group.shape[0] >= 1:
        raise ValueError('Wrong dimension for variance table.')

    products = df_p.index.tolist()
    variance_groups = df_var['Demand Var Group'].tolist()

    # Instantiate Pyomo model.
    pyomo_model = pyo.ConcreteModel()

    # Parameters
    pyomo_model.demand = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Demand'],
    )))
    pyomo_model.margin = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Margin'],
    )))
    pyomo_model.COGS = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['COGS'],
    )))
    pyomo_model.Capacity = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Capacity'],
    )))
    pyomo_model.Substitutability_group = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Substitutability group'],
    )))

    pyomo_model.c = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['c'],
    )))
    pyomo_model.d = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['d'],
    )))
    pyomo_model.loc = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['loc'],
    )))
    pyomo_model.scale = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['scale'],
    )))

    # Generate demand variability samples using Burr12 distribution.
    demand_var_samples = {}
    variances = {}
    for vg in variance_groups:
        demand_var_samples[vg] = burr12.rvs(
            c=pyomo_model.c[vg],
            d=pyomo_model.d[vg],
            loc=pyomo_model.loc[vg],
            scale=pyomo_model.scale[vg],
            size=1000,
        )
        variances[vg] = burr12(
            c=pyomo_model.c[vg],
            d=pyomo_model.d[vg],
            loc=pyomo_model.loc[vg],
            scale=pyomo_model.scale[vg],
        ).var()

    # Decision variables for demand variability samples
    pyomo_model.x = pyo.Var(
        products,
        domain=pyo.NonNegativeReals,
        initialize=0.0,
    )
    pyomo_model.y = pyo.Var(
        products,
        variance_groups,
        domain=pyo.NonNegativeReals,
        initialize=0.0,
    )

    # Objective function
    def objective_function(self):
        """Objective function."""
        return sum(self.margin[p] * pyomo_model.x[p]  for p in products)
    pyomo_model.obj = pyo.Objective(rule=objective_function, sense=pyo.maximize)

    # Constraints
    def capacity_constraint_rule(self,
        p,
    ):
        """
        capacity constraint, stating that the surplus quantity adde to each
        product's demand must not surpass its designated capacity limit.

        Args:
            p: product.

        Returns:
            capacity constraint.
        """
        return self.x[p] <= self.Capacity[p]
    pyomo_model.capacity_constraint = pyo.Constraint(
        products,
        rule=capacity_constraint_rule,
    )

    def aggregate_surplus_constraint_rule(self):
        """
        The aggregate surplus quanityt across all products should not exceeed the
        total demand for all products, adjusted by a macro target percentage.
        """
        return sum(
                self.x[p] * self.demand[p] for p in products
            ) <= mu * sum(self.demand[p] for p in products)
    pyomo_model.aggregate_surplus_constraint = pyo.Constraint(
        rule=aggregate_surplus_constraint_rule,
    )

    def substitutability_group_constraint_rule(self,
        g,
    ):
        """
        Substitutability rule: For products classified within the same
        substitutability group, it is importatn to maintain adequate total surplus
        quantitites.

        Args:
            g: group number.

        Returns:
            Substitutability constraint.
        """
        group_products = [p for p in products if self.Substitutability_group[p] == g]
        return sum(
                self.x[p] * self.demand[p] for p in group_products
            ) >= substitutability_limit[g-1] * sum(
                self.demand[p] for p in group_products
            )
    pyomo_model.substitutability_group_constraint = pyo.Constraint(
        variance_groups,
        rule=substitutability_group_constraint_rule,
    )

    # Constraints for demand variability samples
    def demand_variability_constraint_rule(self,
        p,
    ):
        """
        Ensuring that all the samples from Burr distribution add up to the 
        surplus.
        """
        return sum(self.y[p, vg] for vg in variance_groups) == self.x[p]
    pyomo_model.demand_variability_constraint = pyo.Constraint(
        products,
        rule=demand_variability_constraint_rule,
    )

    # Target demand relationship constraint
    def target_demand_constraint_rule(self,
        p,
    ):
        """
        Target demand constraint, ensuring:
        Target_demand_p >= demand_p * variance_p
        """
        variance_group_p = df_products.loc[
            df_products.index==p,
            'Variance group',
        ].iloc[0]
        return (
            self.demand[p] * variances[variance_group_p]
        ) <= (1 + self.x[p]) * self.demand[p]
    pyomo_model.target_demand_constraint = pyo.Constraint(
        products,
        rule=target_demand_constraint_rule,
    )
    return pyomo_model


def build_two_stage_stochastic_model(
    variances: float,
    df_p: pd.DataFrame,
    df_var: pd.DataFrame,
    substitutability_limit: typing.List[float],
    mu: float = 0.3,
):
    """
    Use 'mpi-sppy' package from Pyomo to solve the problem as a two-stage
    stochastic programming problem.
    Args:
        variances: variance for the given distribution.
        df_p: dataframe that includes information about variables. This
            dataframe has the following columns: Demand, Margin, Capacity,
            Variance group, Substitutability group.
        df_var: Dataframe including distribution parameters for demand variance
            groups.
        substitutability_limit: Substitutability limit for the substitutaion
            constraint. The idea is that we need to have more than this limit
            from products in a group.
        mu: Adjustable macro target percentage.


    Returns:
        mpi-sppy model.
    """
    # checking inputs.
    if not df_p.shape[0] >= 1:
        raise ValueError('Wrong dimension for products table.')
    if not df_variance_group.shape[0] >= 1:
        raise ValueError('Wrong dimension for variance table.')

    products = df_p.index.tolist()
    variance_groups = df_var['Demand Var Group'].tolist()

    # Instantiate Pyomo model.
    deterministic_model = pyo.ConcreteModel()

    # Parameters
    deterministic_model.demand = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Demand'],
    )))
    deterministic_model.margin = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Margin'],
    )))
    deterministic_model.COGS = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['COGS'],
    )))
    deterministic_model.Capacity = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Capacity'],
    )))
    deterministic_model.Substitutability_group = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Substitutability group'],
    )))

    deterministic_model.c = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['c'],
    )))
    deterministic_model.d = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['d'],
    )))
    deterministic_model.loc = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['loc'],
    )))
    deterministic_model.scale = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['scale'],
    )))

    # Decision variables for demand variability samples
    deterministic_model.x = pyo.Var(
        products,
        domain=pyo.NonNegativeReals,
        initialize=0.0,
    )
    deterministic_model.y = pyo.Var(
        products,
        variance_groups,
        domain=pyo.NonNegativeReals,
        initialize=0.0,
    )

    # Objective function
    def objective_function(self):
        """Objective function."""
        return sum(self.margin[p] * deterministic_model.x[p]  for p in products)
    deterministic_model.obj_expr = pyo.Expression(rule=objective_function)
    deterministic_model.obj = pyo.Objective(
        expr=deterministic_model.obj_expr,
        sense=pyo.maximize,
    )

    # Constraints
    def capacity_constraint_rule(self,
        p,
    ):
        """
        capacity constraint, stating that the surplus quantity adde to each
        product's demand must not surpass its designated capacity limit.

        Args:
            p: product.

        Returns:
            capacity constraint.
        """
        return self.x[p] <= self.Capacity[p]
    deterministic_model.capacity_constraint = pyo.Constraint(
        products,
        rule=capacity_constraint_rule,
    )

    def aggregate_surplus_constraint_rule(self):
        """
        The aggregate surplus quanityt across all products should not exceeed the
        total demand for all products, adjusted by a macro target percentage.
        """
        return sum(
                self.x[p] * self.demand[p] for p in products
            ) <= mu * sum(self.demand[p] for p in products)
    deterministic_model.aggregate_surplus_constraint = pyo.Constraint(
        rule=aggregate_surplus_constraint_rule,
    )

    def substitutability_group_constraint_rule(self,
        g,
    ):
        """
        Substitutability rule: For products classified within the same
        substitutability group, it is importatn to maintain adequate total surplus
        quantitites.

        Args:
            g: group number.

        Returns:
            Substitutability constraint.
        """
        group_products = [
            p for p in products if self.Substitutability_group[p] == g
        ]
        return sum(
                self.x[p] * self.demand[p] for p in group_products
            ) >= substitutability_limit[g-1] * sum(
                self.demand[p] for p in group_products
            )
    sub_groups = df_p["Substitutability group"].unique().tolist()
    sub_groups = [int(j) for j in sub_groups]
    deterministic_model.substitutability_group_constraint = pyo.Constraint(
        sub_groups,
        rule=substitutability_group_constraint_rule,
    )

    # Constraints for demand variability samples
    def demand_variability_constraint_rule(self,
        p,
    ):
        """
        Ensuring that all the samples from Burr distribution add up to the 
        surplus.
        """
        return sum(self.y[p, vg] for vg in variance_groups) == self.x[p]
    deterministic_model.demand_variability_constraint = pyo.Constraint(
        products,
        rule=demand_variability_constraint_rule,
    )

    # Target demand relationship constraint
    def target_demand_constraint_rule(self,
        p,
    ):
        """
        Target demand constraint, ensuring:
        Target_demand_p >= demand_p * variance_p
        """
        return (
            self.demand[p] * variances
        ) <= (1 + self.x[p]) * self.demand[p]
    deterministic_model.target_demand_constraint = pyo.Constraint(
        products,
        rule=target_demand_constraint_rule,
    )
    return deterministic_model


def build_two_stage_stochastic_model_with_substitution(
    variances: float,
    df_p: pd.DataFrame,
    df_var: pd.DataFrame,
    mu: float = 0.3,
):
    """
    Formulate the problem as a two-stage stochastic programming with recourse
    where production is stage one's variable, and in stage two, demand is the
    random variable.

    variances: variance for the given distribution (random variable).
        df_p: dataframe that includes information about variables. This
            dataframe has the following columns: Demand, Margin, Capacity,
            Variance group, Substitutability group.
        df_var: Dataframe including distribution parameters for demand variance
            groups.
        mu: Adjustable macro target percentage.

    Returns:
        mpi-sppy model.
    """
    # checking inputs.
    if not df_p.shape[0] >= 1:
        raise ValueError('Wrong dimension for products table.')
    if not df_variance_group.shape[0] >= 1:
        raise ValueError('Wrong dimension for variance table.')

    products = df_p.index.tolist()
    variance_groups = df_var['Demand Var Group'].tolist()

    # Instantiate Pyomo model.
    deterministic_model_subs = pyo.ConcreteModel()

    # Parameters
    deterministic_model_subs.demand = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Demand'],
    )))
    deterministic_model_subs.margin = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Margin'],
    )))
    deterministic_model_subs.COGS = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['COGS'],
    )))
    deterministic_model_subs.Capacity = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Capacity'],
    )))
    deterministic_model_subs.Substitutability_group = pyo.Param(
        products,
        initialize=dict(zip(
            df_p.index,
            df_p['Substitutability group'],
    )))

    deterministic_model_subs.c = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['c'],
    )))
    deterministic_model_subs.d = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['d'],
    )))
    deterministic_model_subs.loc = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['loc'],
    )))
    deterministic_model_subs.scale = pyo.Param(
        variance_groups,
        initialize=dict(zip(
            df_var['Demand Var Group'],
            df_var['scale'],
    )))

    # Decision variable p = production, s = sales.
    deterministic_model_subs.P = pyo.Var(
        products,
        domain=pyo.NonNegativeReals,
    )
    deterministic_model_subs.S = pyo.Var(
        products,
        domain=pyo.NonNegativeReals,
    )
    # Objective function
    def sales_profit(self):
        """sales_profit."""
        return sum(deterministic_model_subs.S[i] * self.margin[i] for i in products)
    def production_cost(self):
        """Production cost."""
        return sum(self.COGS[i] * deterministic_model_subs.P[i] for i in products)
    deterministic_model_subs.SALES_PROFIT = pyo.Expression(rule=sales_profit)
    deterministic_model_subs.PRODUCTION_COST = pyo.Expression(rule=production_cost)
    deterministic_model_subs.obj = pyo.Objective(
        expr=deterministic_model_subs.PRODUCTION_COST - deterministic_model_subs.SALES_PROFIT,
        sense=pyo.minimize,
    )

    # Constraints
    def capacity_constraint_rule(self,
        i,
    ):
        """
        capacity constraint, stating that the surplus quantity adde to each
        product's demand must not surpass its designated capacity limit.

        Args:
            i: product index.

        Returns:
            capacity constraint.
        """
        return self.P[i] <= (1 + self.Capacity[i]) * self.demand[i]
    deterministic_model_subs.capacity_constraint = pyo.Constraint(
        products,
        rule=capacity_constraint_rule,
    )

    def aggregate_surplus_constraint_rule(self):
        """
        The aggregate surplus quanityt across all products should not exceeed the
        total demand for all products, adjusted by a macro target percentage.
        """
        return sum(self.P[i] for i in products) <= (1 + mu) * sum(self.demand[i] * variances for i in products)
    deterministic_model_subs.aggregate_surplus_constraint = pyo.Constraint(
        rule=aggregate_surplus_constraint_rule,
    )

    def substitutability_group_constraint_rule(self,
        g,
    ):
        """
        Substitutability rule: For products classified within the same
        substitutability group, it is important to maintain adequate total surplus
        quantitites.

        Args:
            g: group number.

        Returns:
            Substitutability constraint.
        """
        group_products = [
            i for i in products if self.Substitutability_group[i] == g
        ]
        return sum(self.demand[i] * variances for i in group_products) >= sum(self.S[i] for i in group_products)
    sub_groups = df_p["Substitutability group"].unique().tolist()
    sub_groups = [int(j) for j in sub_groups]
    deterministic_model_subs.substitutability_group_constraint = pyo.Constraint(
        sub_groups,
        rule=substitutability_group_constraint_rule,
    )

    # Target demand relationship constraint
    def sell_product_constraint_rule(self,
        i,
    ):
        """
        p_i >= s_i >= 0
        """
        return self.S[i] <= self.P[i]
    deterministic_model_subs.sell_product_constraint = pyo.Constraint(
        products,
        rule=sell_product_constraint_rule,
    )
    return deterministic_model_subs


def scenario_creator(
    scenario_name: str = "good",
    **kwargs,
) -> pyo.ConcreteModel:
    """
    Scenario creator for `mpisppy` package.
    """
    if scenario_name == "good":
        variances = 0.15
    elif scenario_name == "average":
        variances = 0.2
    elif scenario_name == "bad":
        variances = 0.45
    else:
        raise ValueError("Unrecognized scenario name")

    mpisppy_model = build_two_stage_stochastic_model(
        variances=variances,
        substitutability_limit=sub_lim,
        **kwargs,
    )
    sputils.attach_root_node(
        mpisppy_model,
        mpisppy_model.obj_expr,
        [mpisppy_model.x, mpisppy_model.y],
    )
    mpisppy_model._mpisppy_probability = 1.0 / 3
    return mpisppy_model


def scenario_creator_with_substitution(
    scenario_name: str = "good",
    **kwargs,
) -> pyo.ConcreteModel:
    """
    Scenario creator for `mpisppy` package. Substitution considered.
    """
    if scenario_name == "good":
        variances = 0.15
    elif scenario_name == "average":
        variances = 0.2
    elif scenario_name == "bad":
        variances = 0.45
    else:
        raise ValueError("Unrecognized scenario name")
    mpisppy_model_subs = build_two_stage_stochastic_model_with_substitution(
        variances=variances,
        **kwargs,
    )
    sputils.attach_root_node(
        mpisppy_model_subs,
        mpisppy_model_subs.PRODUCTION_COST,
        [mpisppy_model_subs.P],
    )
    mpisppy_model_subs._mpisppy_probability = 1.0 / 3
    return mpisppy_model_subs


# %% Main
# Load and preprocess data.
dff = pd.read_excel(DATA_FILE, index_col=0)
df_products = dff.transpose()
df_variance_group = pd.read_excel(DATA_FILE, sheet_name=1)

# parameters that the use can modify.
MAX_CAPACITY = [0.15, 0.5, 1] # what to subsitutute "NaN" values in "Capacity"
                              # column with.
MACRO_TARGET_PERCENTAGE = [0.3, 0.4, 0.5]  # Adjustable macro target percentage
SUB_LIMIT_BOUND = [0.01, 0.05, 0.1] # Substitutability limit.
EPSILON = 0.1  # Robustness parameter
mpisppy_options = {"solver": "glpk"}
mpisppy_all_scenario_names = ["good", "average", "bad"]

pulp_sol = []
pyomo_sol = []
mpisppy_sol = []
mpisppy_sol_subs = []
VERBOSE = 0

for max_capacity in tqdm.tqdm(MAX_CAPACITY, disable=False):
    for macro in MACRO_TARGET_PERCENTAGE:
        for sub in SUB_LIMIT_BOUND:
            df_products['Capacity'] = df_products['Capacity'].fillna(
                value=max_capacity,
            )
            sub_lim = [sub] * len(
                df_products["Substitutability group"].unique()
            )

            # use Pyomo to solve robust optimization problem.
            model = build_robust_optimization_pyomo_model(
                df_p=df_products,
                df_var=df_variance_group,
                mu=macro,
                substitutability_limit=sub_lim,
            )
            solver = pyo.SolverFactory('glpk')
            pyomo_results = solver.solve(model)

            # Use Pulp.
            pulp_model = build_robust_optimization_pulp_model(
                df_p=df_products,
                df_var=df_variance_group,
                epsilon=EPSILON,
                mu=macro,
                substitutability_limit=sub_lim,
            )
            pulp_model.solve(pulp.PULP_CBC_CMD(msg=False))

            # Use mpi-sppy to solve the problem as a two-stage stochastic
            # problem.
            scenario_kwargs = {
                'df_p': df_products,
                'df_var':df_variance_group,
                'mu':macro,
            }
            ef = ExtensiveForm(
                mpisppy_options,
                mpisppy_all_scenario_names,
                scenario_creator,
                scenario_creator_kwargs=scenario_kwargs,
            )
            results = ef.solve_extensive_form()
            objval = ef.get_objective_value()

            # Use mpi-sppy to solve two-stage stochastic program with
            # substitution.
            ef_subs = ExtensiveForm(
                mpisppy_options,
                mpisppy_all_scenario_names,
                scenario_creator_with_substitution,
                scenario_creator_kwargs=scenario_kwargs,
            )
            results_subs = ef_subs.solve_extensive_form()
            objval_subs = ef_subs.get_objective_value()

            pulp_sol.append(int(pulp_model.objective.value()))
            pyomo_sol.append(int(pyo.value(model.obj)))
            mpisppy_sol.append(int(objval))
            mpisppy_sol_subs.append(-int(objval_subs))

            if VERBOSE:
                print("*** ------------------------------------- ***")
                print("max capacity = ", max_capacity)
                print("MACRO TARGET PERCENTAGE = ", macro)
                print("SUBSTITUTION LIMIT = ", sub)
                print(
                    "Robust Optimization objective value using Pyomo:",
                    int(pyo.value(model.obj)),
                )
                print(
                    "Robust optimization 0bjective Value using Pulp:",
                    int(pulp_model.objective.value()),
                )
                print(
                    "Two-stage stochastic programming usin mpisppy:",
                    int(objval),
                )
                print(
                    "Two-stage stochastic programming with substitution:",
                    -int(objval_subs),
                )


out = pd.DataFrame(data={
    'pulp_sol': pulp_sol,
    'pyomo_sol': pyomo_sol,
    'mpisppy_sol': mpisppy_sol,
    'mpisppy_sol_subs': mpisppy_sol_subs,
})
