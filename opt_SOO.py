# optimization - 17 february 2025
def ss_data_crosscheck(path,output_dict,path1):
    from pymoo.problems.functional import FunctionalProblem
    import pandas as pd
    import pickle
    from pymoo.algorithms.soo.nonconvex.de import DE
    from pymoo.termination.default import DefaultSingleObjectiveTermination
    from pymoo.operators.sampling.lhs import LHS
    from pymoo.optimize import minimize
    from itertools import product
    import numpy as np
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.variable import Real, Integer, Choice
    from pymoo.core.mixed import MixedVariableGA
    import json
    
    with open(path1, 'r') as file:
        data = json.load(file)


    objective = data['objective']
    target = data['target']
    
    df_ss = pd.read_excel(path)
    def transform(df):
        def transformation(df):
            return df.set_index('Variable Name').T.reset_index()
        def discrete_levels(df):
            a = []
            b = []
            discrete_list = df[(df['Variable Type'].str.contains('Discrete')) & (df['Feature']=='Input')]['index'].to_list()
            for i in discrete_list:
                vt = df[df['index'] == i]['Variable Type'].to_list()[0]
                vt_2 = vt.split('=')[1].strip().split(',')
                a.append(i)
                b.append(vt_2)
            return pd.DataFrame({'cv':a,'levels':b})
        def cv_step(df):
            a = []
            b = []
            c = []
            d = []
            cv_step_list = df[(df['Variable Type'].str.contains('C with I')) & (df['Feature']=='Input')]['index'].to_list()
            for i in cv_step_list:
                a.append(float(df[df['index'] == i]['Variable Type'].to_list()[0].split('=')[1].strip()))
                b.append(float(df[df['index'] == i]['Min']))
                c.append(float(df[df['index'] == i]['Max']))
                d.append(i)
            return pd.DataFrame({'variable':d,'step size':a,'min':b,'max':c})
        def cv(df):
            a = []
            b = []
            c = []
            cv_list = df[(df['Variable Type'] == 'C') & (df['Feature']=='Input')]['index'].to_list()
            for i in cv_list:
                a.append(i)
                b.append(float(df[df['index'] == i]['Min']))
                c.append(float(df[df['index'] == i]['Max']))
            return pd.DataFrame({'variable':a, 'min':b,'max':c})

        discrete_df = discrete_levels(transformation(df))
        continuous_step_df = cv_step(transformation(df))
        continuous_df = cv(transformation(df))
        discrete_df['levels'] = discrete_df['levels'].apply(lambda x:[float(i) for i in x])
        discrete_df_dict = dict(zip(discrete_df['cv'], discrete_df['levels']))
        
        continuous_step_df_dict = dict()
        continuous_df_dict = dict()
        for i in range(len(continuous_step_df)):
            continuous_step_df_dict[continuous_step_df['variable'][i]] = (continuous_step_df['min'][i],continuous_step_df['max'][i],continuous_step_df['step size'][i])
        for i in range(len(continuous_df)):
            continuous_df_dict[continuous_df['variable'][i]] = (continuous_df['min'][i],continuous_df['max'][i])


        return continuous_df_dict,discrete_df_dict,continuous_step_df_dict
    
    continuous_df_dict,discrete_df_dict,continuous_step_df_dict = transform(df=df_ss)
    model_dict = {}
    for out in list(output_dict.keys()):
        with open(output_dict[out]['Model_object'], 'rb') as file:
            best_model = pickle.load(file)
        model_dict[out] = best_model
    features = output_dict[list(output_dict.keys())[0]]['features']
    features_optimization = list(continuous_df_dict.keys()) + list(discrete_df_dict.keys()) + list(continuous_step_df_dict.keys()) 
    
    final_dict = {}
    if len(set(features)) > len(set(features_optimization)):
        missing_features = list(set(features) - set(features_optimization))
        final_dict['solution'] = (', ').join([str(i) for i in missing_features]) + ' ' + 'missing from optimization search space'
        return final_dict
    # elif len(set(features_optimization)) > len(set(features)):
    #     missing_features = list(set(features_optimization) - set(features))  
    #     final_dict['solution'] = (', ').join([str(i) for i in missing_features]) + ' ' + 'has not been used in modelling'
    #     return final_dict 
    else:
        # if list(continuous_df_dict.keys())+list(discrete_df_dict.keys())+list(continuous_step_df_dict.keys()) == features:
        cdf = continuous_df_dict.copy()
        ddf = discrete_df_dict.copy()
        csdf = continuous_step_df_dict.copy()

        def create_dict_subset(original_dict, keys_list):
            subset = {key: original_dict[key] for key in keys_list if key in original_dict}
            return subset
        
        continuous_df_dict = create_dict_subset(cdf, features)
        discrete_df_dict = create_dict_subset(ddf,features)
        continuous_step_df_dict = create_dict_subset(csdf, features)

        var_order = list(continuous_df_dict.keys())+list(discrete_df_dict.keys())+list(continuous_step_df_dict.keys())
        # var_order = [i for i in continuous_df_dict.keys() if i in features]+[i for i in discrete_df_dict.keys() if i in features]+[i for i in continuous_step_df_dict.keys() if i in features]
        if (len(discrete_df_dict)==0) & (len(continuous_step_df_dict)==0):
            def maximize_objective_function(x,model_dict=model_dict,target=target):
                return -1*model_dict[target].predict([x])[0]

            def minimize_objective_function(x,model_dict=model_dict,target=target):
                return model_dict[target].predict([x])[0]

            def constraint_generate_functions(data,model_dict=model_dict,var_order=var_order):
                value_based_constraints = {}
                for key,value in data['constraints'].items():
                    value_based_constraints[key] = (value['lower_constraint'],value['upper_constraint'])

                function_list = []
                for key,value in value_based_constraints.items():
                    def new_function(x,key=key,value=value):
                        y2_pred = model_dict[key].predict([x])[0]

                        lower_bound = value[0]
                        upper_bound = value[1]

                        if (lower_bound!=None) & (upper_bound==None):
                            if y2_pred < lower_bound:
                                return lower_bound - y2_pred
                            else:
                                return 0
                        elif (lower_bound==None) & (upper_bound!=None):
                            if y2_pred > upper_bound:
                                return y2_pred - upper_bound
                            else:
                                return 0
                        else:
                            if y2_pred < lower_bound:
                                return lower_bound - y2_pred  # Return positive value if below lower bound
                            elif y2_pred > upper_bound:
                                return y2_pred - upper_bound  # Return positive value if above upper bound
                            else:
                                return 0

                    function_list.append(new_function)

                for eq in data['equation_based_constraints']:
                    if eq['LC']!=None:
                        def lc_function(x,eq=eq,var_order=var_order):
                            var_dict = {var_order[ind]: x[ind] for ind in range(len(var_order))}
                            eq_negated = f"-({eq['equation']})" + f"+{eq['LC']}"
                            return eval(eq_negated, {}, var_dict)
                        function_list.append(lc_function)
                    if eq['UC']!=None:
                        def uc_function(x,eq=eq,var_order=var_order):
                            var_dict = {var_order[ind]: x[ind] for ind in range(len(var_order))}
                            uc = -1 * eq['UC']
                            eq_norm = eq['equation'] + f"+{uc}"
                            return eval(eq_norm, {}, var_dict)
                        function_list.append(uc_function)

                return function_list
            Constraint_functions = constraint_generate_functions(data)
            n_var = len(continuous_df_dict)
            bounds = list(continuous_df_dict.values())
            if objective == 'minimize':
                objective_function = minimize_objective_function
            elif objective == 'maximize':
                objective_function = maximize_objective_function
            problem = FunctionalProblem(n_var,
                        objective_function,
                        xl=np.array([v[0] for v in list(bounds)]),
                        xu=np.array([v[1] for v in list(bounds)]),
                        constr_ieq=Constraint_functions
                        )
            algorithm = DE(

                )

            termination = DefaultSingleObjectiveTermination(
                    xtol=1e-3,
                    ftol=1e-5,
                    period=100,
                    n_max_gen=1000,
                    n_max_evals=100000
                )

            result = minimize(problem,
                    algorithm,
                    termination,
                    save_history=True, 
                    seed=42,
                    verbose=False)

            history = {}
            for i, entry in enumerate(result.history):
                for ind in entry.pop:
                    if np.all(ind.G <= 0):
                        history[tuple(ind.X)] = ind.F[0]

            history_all = [e.opt[0].F[0] for e in result.history]

            if len(history)!=0:
                if objective == 'minimize':
                    final_solution_dict = dict(sorted(history.items(), key=lambda item: item[1])[:5])
                    final_mapped_solution_dict = {}
                    for count,(key,value) in enumerate(final_solution_dict.items()):
                        temp_li=[]
                        temp_li.append(dict(zip(list(continuous_df_dict.keys()),key)))
                        temp_li.append(value)
                        final_mapped_solution_dict[count+1]=temp_li
                    final_dict = {}
                    final_dict['solution'] = final_mapped_solution_dict
                    iteration_count = list(range(1,len(history_all)+1))

                    final_dict['convergence_plot'] = [iteration_count,history_all]
                else:
                    final_solution_dict = dict(sorted(history.items(), key=lambda item: item[1])[:5])
                    final_mapped_solution_dict = {}
                    for count,(key,value) in enumerate(final_solution_dict.items()):
                        temp_li=[]
                        temp_li.append(dict(zip(list(continuous_df_dict.keys()),key)))
                        temp_li.append(-1*value)
                        final_mapped_solution_dict[count+1]=temp_li
                    final_dict = {}
                    final_dict['solution'] = final_mapped_solution_dict
                    iteration_count = list(range(1,len(history_all)+1))
                    fitness = [-c for c in history_all]
                    final_dict['convergence_plot'] = [iteration_count,fitness]
                return final_dict                       
            else:
                final_dict = {}
                final_dict['solution'] = 'Infeasible Solution'
                return final_dict
        else:
            def constraint_generate_functions(data,model_dict=model_dict,var_order=var_order):
                value_based_constraints = {}
                for key,value in data['constraints'].items():
                    value_based_constraints[key] = (value['lower_constraint'],value['upper_constraint'])

                function_list = []
                for key,value in value_based_constraints.items():
                    def new_function(x,key=key,value=value):
                        y2_pred = model_dict[key].predict([x])[0]

                        lower_bound = value[0]
                        upper_bound = value[1]

                        if (lower_bound!=None) & (upper_bound==None):
                            if y2_pred < lower_bound:
                                return lower_bound - y2_pred
                            else:
                                return 0
                        elif (lower_bound==None) & (upper_bound!=None):
                            if y2_pred > upper_bound:
                                return y2_pred - upper_bound
                            else:
                                return 0
                        else:
                            if y2_pred < lower_bound:
                                return lower_bound - y2_pred  # Return positive value if below lower bound
                            elif y2_pred > upper_bound:
                                return y2_pred - upper_bound  # Return positive value if above upper bound
                            else:
                                return 0

                    function_list.append(new_function)

                for eq in data['equation_based_constraints']:
                    if eq['LC']!=None:
                        def lc_function(x,eq=eq,var_order=var_order):
                            var_dict = {var_order[ind]: x[0][ind] for ind in range(len(var_order))}
                            eq_negated = f"-({eq['equation']})" + f"+{eq['LC']}"
                            return eval(eq_negated, {}, var_dict)
                        function_list.append(lc_function)
                    if eq['UC']!=None:
                        def uc_function(x,eq=eq,var_order=var_order):
                            var_dict = {var_order[ind]: x[0][ind] for ind in range(len(var_order))}
                            uc = -1 * eq['UC']
                            eq_norm = eq['equation'] + f"+{uc}"
                            return eval(eq_norm, {}, var_dict)
                        function_list.append(uc_function)

                return function_list

            Constraint_functions = constraint_generate_functions(data)

            real_variables = {}
            choice_variables={}
            for key,value in discrete_df_dict.items():
                choice_variables[key] = Choice(options=value)
            for key,value in continuous_step_df_dict.items():
                choice_variables[key] = Choice(options=list(np.arange(value[0],value[1],value[2])))
            for key,value in continuous_df_dict.items():
                real_variables[key] =  Real(bounds=(value[0],value[1]))

            variables = real_variables | choice_variables


            class MixedVariableOptimizationProblem(ElementwiseProblem):

                def __init__(self, model, constraints, variables,objective,**kwargs):
                    self.model = model
                    self.constraints = constraints
                    self.variables = variables
                    self.objective = objective
                    n_constr = len(constraints)  # Number of constraints
                    super().__init__(vars=self.variables, n_obj=1, n_constr=n_constr,**kwargs)

                def _evaluate(self, X, out, *args, **kwargs):
                    X_values = np.array(list(X.values())).reshape(1, -1)
                    if self.objective == 'maximize':
                        y_pred = -1*(self.model.predict(X_values))
                    else:
                        y_pred = self.model.predict(X_values)

                    out["F"] = y_pred[0]

                    # Initialize constraints
                    out["G"] = np.zeros(len(self.constraints))

                    for i, constraint in enumerate(self.constraints):
                        out["G"][i] = constraint(X_values)

            optimization_problem = MixedVariableOptimizationProblem(
                    model_dict[target],
                    Constraint_functions,
                    variables,objective
                )
            algorithm = MixedVariableGA()
            termination = DefaultSingleObjectiveTermination(
                    xtol=1e-3,
                    ftol=1e-5,
                    period=100,
                    n_max_gen=1000,
                    n_max_evals=100000
                )


            result = minimize(optimization_problem,
                                algorithm,
                                termination=termination,
                                seed=42,
                                verbose=False,
                                save_history=True)

            history = {}
            for i, entry in enumerate(result.history):
                for ind in entry.pop:
                    if np.all(ind.G <= 0):
                        key = str(ind.X) 
                        history[key] = ind.F[0]
            sorted_history = sorted(history.items(), key=lambda item: item[1])
            final_solution_dict = {
                i + 1: [eval(key), value] for i, (key, value) in enumerate(sorted_history[:5])
            }
            
            history_all = [e.opt[0].F[0] for e in result.history]
            
            if len(history)!=0:
                if objective == 'minimize':
                    final_dict = {}
                    final_dict['solution'] = final_solution_dict
                    iteration_count = list(range(1,len(history_all)+1))
                    final_dict['convergence_plot'] = [iteration_count,history_all]
                else:
                    for key, value in final_solution_dict.items():
                        value[1] *= -1
                    final_dict = {}
                    final_dict['solution'] = final_solution_dict
                    iteration_count = list(range(1,len(history_all)+1))
                    fitness = [-c for c in history_all]
                    final_dict['convergence_plot'] = [iteration_count,fitness]
                    
                return final_dict                       
            else:
                final_dict = {}
                final_dict['solution'] = 'Infeasible Solution'
                return final_dict