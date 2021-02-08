#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 10:47:10 2020
ROI afhankelijk van prijs toevoegen
Log lin?
@author: simonteggelaar
"""
import numpy as np
from itertools import product
import pandas as pd
import itertools
import statsmodels.formula.api as smf
from tqdm import tqdm
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from statsmodels.stats.outliers_influence import variance_inflation_factor


class MultipleRegressions:
    def __init__(self, data, sales='kpi', used_variables=[], optional_variables=[], or_variables=[], media=[],
                 conditions={'P_value_min': 1, 'AdjR2_min': 0.0, 'DW_min': 0, 'DW_max': 4, 'JB_min': 0.0},
                 positive=[], negative=[], media_positive=[]):
        self.data = data
        self.sales = sales
        self.used_variables = used_variables
        self.optional_variables = optional_variables
        self.or_variables = or_variables
        self.media = media
        self.conditions = conditions
        self.output1 = []
        self.positive = positive
        self.negative = negative
        self.media_positive = media_positive

        if len(self.media) > 0:
            df_media_decays = df_media(self.media, self.data)
            df_merged = pd.concat([self.data, df_media_decays], axis=1).reset_index()
            model_data = df_merged
        else:
            model_data = self.data

        self.data_models = model_data
        self.get_roi = use_roi(self.media)

    def __str__(self):
        return "These regressions will have as a kpi: {} and variables that have the enter the model: {}".format(
            self.sales, self.used_variables)

    def plot_curves(self, media_var, alphas_scurve=[], alphas_diminishing_curve=[]):
        '''
        This function allows you to plot the response curves you are going to test in the model
        you can test different alpha's for the Scurve and a diminishing curve

        input:
            -media_var: a variable you want to see as response curve (like TV GRP's or Radio spend)
                        should be a panda series

            -alphas_scurve: a list of alpha's you wanna test for the s-curve
                            should be a list like: [1,3,10]

            -alphas_diminishing_curve: a list of alpha's you wanna test for the diminshing-curve
                            should be a list like: [1,3,10]
         '''
        pio.renderers.default = 'browser'
        x = np.arange(max(media_var) + 0.1 * max(media_var))
        # to plot virtical lines min/mean/max
        l = [i for i in list(media_var.array) if i > 0]  # remove all 0 values
        min_ = min(l)
        max_ = max(l)
        mean_ = sum(l) / len(l)

        fig = go.Figure()
        for aplha_value in alphas_scurve:
            fig.add_trace(go.Scatter(x=x, y=s_curve(x, aplha_value),
                                     name='s_curve' + ' alpha = ' + str(aplha_value)))
        for aplha_value in alphas_diminishing_curve:
            fig.add_trace(go.Scatter(x=x, y=diminishing_curve(x, aplha_value),
                                     name='diminishing_curve' + ' alpha = ' + str(aplha_value)))
        fig.update_layout(title='Response Curves to be tested',
                          xaxis_title=media_var.name,
                          yaxis_title='Impact (will be multiplied with your estimated beta)')

        fig.add_shape(
            # Line Vertical min
            dict(
                type="line",
                x0=min_,
                y0=0,
                x1=min_,
                y1=1,
                line=dict(
                    color="RoyalBlue",
                    width=4,
                    dash="dot",
                )
            ))
        fig.add_shape(
            # Line Vertical mean
            dict(
                type="line",
                x0=mean_,
                y0=0,
                x1=mean_,
                y1=1,
                line=dict(
                    color="Red",
                    width=4,
                    dash="dot",
                )
            ))
        fig.add_shape(
            # Line Vertical min
            dict(
                type="line",
                x0=max_,
                y0=0,
                x1=max_,
                y1=1,
                line=dict(
                    color="Orange",
                    width=4,
                    dash="dot",
                )
            ))
        # Create scatter trace of text labels
        fig.add_trace(go.Scatter(
            x=[min_, mean_, max_],
            y=[1.01, 1.01, 1.01],
            text=["Min GRPs or $",
                  "Average GRPs or $",
                  "Max GRPs or $"],
            mode="text",
            name="Inzet values",
        ))
        fig.show()

    def plot_decay(self, date=[]):
        """
        This functions show you the media variables in your model and how they are transformed by the decay function and chosen values

        -input:
            optional date: pandas series to show the date on the x axis
                for example: date = data.datum

        """
        pio.renderers.default = 'browser'
        df_decays = df_media(self.media, self.data)
        if len(date) > 0:
            x = date
        else:
            x = np.arange(len(self.data))

        variables = list(df_decays.keys())

        fig = go.Figure()

        for var in variables:
            fig.add_trace(go.Scatter(x=x, y=df_decays[var],
                                     name=var))

        actual_variables = list(self.media.keys())
        for ac_var in actual_variables:
            fig.add_trace(go.Bar(x=x, y=self.data[ac_var],
                                 name=ac_var))
        fig.show()

    def plot_decay_and_curves(self, date=[]):
        """
        This functions show you the media variables in your model and how they are transformed by the decay and response curves function and chosen values

        -input:
            optional date: pandas series to show the date on the x axis
                for example: date = data.datum

        """

        pio.renderers.default = 'browser'
        df = self.data
        media = self.media

        if len(date) > 0:
            x = date
        else:
            x = np.arange(len(df))

        actual_variables = list(media.keys())

        fig = go.Figure()

        for ac_var in actual_variables:
            alpha_ = media[ac_var]['alpha']
            decay_ = media[ac_var]['decay']
            var_ = df[ac_var]
            if media[ac_var]['curve'] == 's_curve':
                for al_ in alpha_:
                    for dec_ in decay_:
                        y = s_curve(decay(var_, dec_), al_)
                        fig.add_trace(go.Scatter(x=x, y=y,
                                                 name='s_curve ' + ac_var + ' ' + 'dec' + str(dec_) + ' alpha ' + str(
                                                     al_)))
        if media[ac_var]['curve'] == 'diminishing_curve':
            for al_ in alpha_:
                for dec_ in decay_:
                    y = diminishing_curve(decay(var_, dec_), al_)
                    fig.add_trace(go.Scatter(x=x, y=y, name='diminishing_curve ' + ac_var + ' ' + 'dec' + str(
                        dec_) + ' alpha ' + str(al_)))
        fig.show()

    def models_to_be_tested(self):
        '''
        This function shows you how many models will be tested depning on your input for the variables and parameters

        no input is needed
        '''
        if len(self.media) > 0:
            or_variables_media = variables_decay_rc_names(self.media)[2]
            self.or_variables.update(or_variables_media)
        if len(self.or_variables) > 0:
            possibilities_ = []
            for key in self.or_variables:
                possibilities_.append(len(self.or_variables[key]))
            amount_or_variables = np.multiply.reduce(possibilities_)
            models_testing = pow(2, len(self.optional_variables)) * amount_or_variables
        else:
            models_testing = pow(2, len(self.optional_variables))
        return print('Models to be tested: ', models_testing, " Estimated time in sec: ", models_testing / 20)

    def create_formulas(self):  # sales,used_variables,optional_variables,or_variables):
        '''
        Function to create formulas based on the 3 input variable lists
        Variables can be used as power with 'power(var,n)' or something like 'np.log(var)'.
        '''
        # !#!#! AANPASSEN als or_variables leeg is
        # optional_variables
        if len(self.media) > 0:
            or_variables_media = variables_decay_rc_names(self.media)[2]
            self.or_variables.update(or_variables_media)
        optional_variables_list = []
        for i in range(len(self.optional_variables) + 1):
            optional_variables_list += list(itertools.combinations(self.optional_variables, i))
        # or_variables
        if len(self.or_variables) > 0:
            or_variables_list = pd.DataFrame([row for row in product(*self.or_variables.values())],
                                             columns=self.or_variables.keys())
        else:
            or_variables_list = []
        # used_variables nothing to do with
        # samenvoegen tot formules
        formules = []
        for j in range(len(optional_variables_list)):
            opt_var = optional_variables_list[j]

            if len(or_variables_list) == 0:
                f = self.sales + ' ~ '
                if len(self.used_variables) > 0:
                    f += " + ".join(self.used_variables)
                if len(opt_var) > 0:
                    f += " + " + " + ".join(opt_var)
                if len(opt_var) + len(self.used_variables) == 0:
                    f += "1"
                formules.append(f)
            else:

                for z in range(len(or_variables_list)):
                    of_var = list(or_variables_list.loc[z,])
                    f = self.sales + ' ~ '
                    if len(self.used_variables) > 0:
                        f += " + ".join(self.used_variables) + " + "
                    if len(of_var) > 0:
                        f += " + ".join(of_var)
                    if len(opt_var) > 0:
                        f += " + " + " + ".join(opt_var)
                    if len(opt_var) + len(of_var) + len(self.used_variables) == 0:
                        f += "1"
                    formules.append(f)
        return formules

    def all_variables_used(self):
        '''
        This function shows all the variables that are used in the models
        '''
        or_variables_list = []
        if len(self.or_variables) > 0:
            for sublist in list(self.or_variables.values()):
                for item in sublist:
                    or_variables_list.append(item)
        variabelen_totaal = or_variables_list + self.optional_variables + self.used_variables
        return variabelen_totaal

    def model_characteristics(self, model):
        '''
        Input: een res.summary() from an ols model
        Output: AdjR2, DW, JB  
        '''
        results = model.fit()
        AdjR2 = round(results.rsquared_adj, 3)
        results_summary = results.summary()
        results_as_html2 = results_summary.tables[2].as_html()
        dwjb = pd.read_html(results_as_html2, index_col=0)[0]
        DW = round(dwjb.iloc[0, 2], 3)
        JB = round(dwjb.iloc[2, 2], 3)
        return {'AdjR2': AdjR2, 'DW': DW, 'JB': JB}

    def decomposition_sum(self, model):
        '''
        for a given model, it returns the % contribution of a variable
        '''
        res = model.fit()
        X = pd.DataFrame(model.exog, columns=model.exog_names)
        X_beta = X * res.params
        decomp_totaal = X_beta.sum() / res.fittedvalues.sum()
        return decomp_totaal

    def roi(self, model): #TOEVOEGEN ALS ER GEEN KOSTEN ZIJN OVERSLAAN
        '''
        determines the roi for each media variable and for the total media
        it includes all the cost for total media roi
        '''

        #If cost is used:
        if use_roi(self.media):
            decomp = self.decomposition_sum(model)
            media_roi = {}
            contribution_total = 0
            cost_total = 0

            if len(self.media.keys())>0:
                for media_var in list(self.media.keys()):
                    try:
                        contribution = decomp.filter(like=media_var, axis=0)[0]
                        cost = self.media[media_var]['cost'][0]
                        roi = (self.data[self.sales].sum() * contribution) / cost
                        media_roi[media_var] = round(roi, 3)
                        # for total ROI
                        contribution_total += contribution
                        cost_total += cost
                    except:
                        contribution_total += 0
                        cost_total += self.media[media_var]['cost'][0]
                        media_roi[media_var] = ""
                if cost_total == 0:
                    cost_total = 1
                roi_totaal = round((self.data[self.sales].sum() * contribution_total) / cost_total, 3)
            return roi_totaal, media_roi

    def estimates(self, model):
        res = model.fit()
        return res.params, res.pvalues

    def reject_model(self, model):
        '''
        Determines if the model gets rejected by the conditions given and stores why it is rejectes
        '''
        P_value_rejection = AdjR2_min_rejection = DW_rejection = JB_rejection = 0

        P_value_max = self.estimates(model)[1].max()
        AdjR2, DW, JB = self.model_characteristics(model).values()

        if P_value_max > self.conditions['P_value_min']:
            P_value_rejection = 1

        if AdjR2 < self.conditions['AdjR2_min']:
            AdjR2_min_rejection = 1

        if DW < self.conditions['DW_min'] or DW > self.conditions['DW_max']:
            DW_rejection = 1

        if JB < self.conditions['JB_min']:
            JB_rejection = 1

        return P_value_rejection, AdjR2_min_rejection, DW_rejection, JB_rejection

    def reject_sign_variables(self, model):
        '''
        Determines if the model gets rejected by the sign of an estimate
        '''
        rejection = 0
        variables_estimates = self.estimates(model)[0]
        variables_in_model = self.estimates(model)[0].index

        positive_rejection = []
        for positive_var in self.positive:
            if positive_var in variables_in_model:
                if variables_estimates[positive_var] < 0:
                    positive_rejection.append({positive_var})
                    rejection = 1
        negative_rejection = []
        for negative_var in self.negative:
            if negative_var in variables_in_model:
                if variables_estimates[negative_var] > 0:
                    negative_rejection.append({negative_var})
                    rejection = 1

        if any(self.media_positive):
            if self.reject_media_negative(model) == 1:
                rejection = 1

        return positive_rejection, negative_rejection, rejection

    def reject_media_negative(self, model):
        media_sign_rejection = 0
        estimates_ = model.fit().params
        for media_var in list(self.media.keys()):
            if estimates_.filter(like=media_var)[0] < 0:
                media_sign_rejection = 1
        return media_sign_rejection

    def run_all_models(self):
        '''
        This function runs all the models for you, and shows you how long it expects it to take.
        '''
        model_data = self.data_models

        rejected_model_statistics = []
        rejected_sign_betas = []  # Deze kijken we pas naar als de statistics voldoen
        # List met uitkomsten voor het model
        model_statistics = []
        formules_used = []
        estimates_model = []
        decomposition = []
        roi_models = []
        formules_ = self.create_formulas()
        for i in tqdm(range(len(formules_))):
            model = smf.ols(formula=formules_[i], data=model_data)
            # Restrictions model:
            if sum(self.reject_model(model)) == 0:
                # Restrictions estimates:
                if self.reject_sign_variables(model)[2] == 0:
                    # Model not rejected:
                    # Formula used appending
                    formules_used.append(formules_[i])
                    # Model statistics appending
                    model_statistics.append(self.model_characteristics(model))
                    # Model estimates appending
                    estimates_model.append(self.estimates(model))
                    # Decomposition appending
                    decomposition.append(self.decomposition_sum(model))
                    if self.get_roi == 1:
                        roi_models.append(self.roi(model))
                # rejected because of estimate conditions
                else:
                    rejected_sign_betas.append(self.reject_sign_variables(model)[0:2])
                    # rejected because of model conditions
            else:
                rejected_model_statistics.append(self.reject_model(model))

        betas = []
        p_values = []
        for i in range(len(estimates_model)):
            betas.append(estimates_model[i][0])
            p_values.append(estimates_model[i][1])

        self.betas = pd.DataFrame(betas)
        self.p_values = pd.DataFrame(p_values)

        if self.get_roi == 1 and any(roi_models):
            self.model_roi = create_roi_df(pd.DataFrame(roi_models, columns=["Total ROI", "Media type ROI"]))
        else:
            self.model_roi = []

        self.model_statistics = pd.DataFrame(model_statistics).join(pd.DataFrame(formules_used))
        self.formules_used = pd.DataFrame(formules_used)
        self.decomposition = pd.DataFrame(decomposition)
        self.rejected_model_statistics = pd.DataFrame(rejected_model_statistics,
                                                      columns=['P_value_rejection', 'AdjR2_min_rejection',
                                                               'DW_rejection', 'JB_rejection'])
        self.rejected_sign_betas = pd.DataFrame(rejected_sign_betas)
        self.estimates_model = estimates_model

    def get_model_values_media(self):
        '''
        A function that returns the estimates around media and the ROIs
        '''
        if len(self.media) > 0:
            media_vars = list(self.media.keys())
            final_results = {}
            for media_var in media_vars:

                df_b = self.betas.filter(like=media_var, axis=1)
                df_d = self.decomposition.filter(like=media_var, axis=1)
                df_p = self.p_values.filter(like=media_var, axis=1)

                if use_roi(self.media):
                    df_roi = self.model_roi[media_var]
                else:
                    df_roi = 'NA'
                betas_ = []
                decay_ = []
                alpha_ = []

                decomp_ = []
                p_values_ = []

                for i in range(0, len(df_b)):
                    estimate = df_b.iloc[i, :].dropna()
                    betas_.append(estimate.item())
                    name_ = list(estimate.keys())[0]
                    dec, alp = get_alp_dec_from_name(name_)
                    decay_.append(dec)
                    alpha_.append(alp)

                    decomp_.append(df_d.iloc[i, :].dropna().item())
                    p_values_.append(df_p.iloc[i, :].dropna().item())

                results = pd.DataFrame(
                    {'betas': betas_,
                     'decomp_values': decomp_,
                     'p_values': p_values_,
                     'decay': decay_,
                     'alpha': alpha_,
                     'roi': df_roi
                     })
                results['curve'] = self.media[media_var]['curve']
                final_results[media_var] = results
            self.media_estimes = final_results
            return final_results

        else:
            print("No media in this model")

    def best_media_table(self, N=5):
        '''
        Returns the N most significant media estimates

        Input:
            -N can be set to the amount, and without input is set on 5
        '''
        best_media = {}
        media_estimes_ = self.get_model_values_media()
        for media_var in self.media:
            best_media[media_var] = media_estimes_[media_var].sort_values('p_values').iloc[0:N, :]
        return best_media

    def plot_media_curves_results(self, N=5, media_variables=[]):
        '''
        Function that plots the founded results for media from the models

        Input:
            -N = integer, without input set on 5
                Gives the N most significant results
            -media_variables = a list of variables that will be plotted,
                without input it will plot all the variables
        '''

        pio.renderers.default = 'browser'

        if len(media_variables) > 0:
            media_vars = media_variables
        else:
            media_vars = list(self.media.keys())

        media_best_table = self.best_media_table(N=N)

        max_range = 0
        for media_var in media_vars:
            max_range = max(max_range, max(self.data[media_var]))
        x = np.arange(max_range + 0.1 * max_range)

        fig = go.Figure()

        for media_var in media_vars:
            curve = self.media[media_var]['curve']
            betas = list(media_best_table[media_var]['betas'])
            alphas = list(media_best_table[media_var]['alpha'])

            for i in range(0, len(betas)):
                if curve == 's_curve':
                    fig.add_trace(go.Scatter(x=x, y=betas[i] * s_curve(x, float(alphas[i])),
                                             name=media_var + ' s_curve' + ' alpha = ' + str(alphas[i])))
                if curve == 'diminishing_curve':
                    fig.add_trace(go.Scatter(x=x, y=betas[i] * diminishing_curve(x, float(alphas[i])),
                                             name=media_var + ' diminishing_curve' + ' alpha = ' + str(alphas[i])))

        # Axis
        fig.update_layout(title='Most significant Response Curves', xaxis_title="Amount of media",
                          yaxis_title='Impact on the KPI')

        fig.show()


    def histogram_extra_info(self, hist_variables, what_to_show='d', variables_to_show=[], marginal='rug'):
        '''

            This is function to illustate the results founds, it show a histogram of the results
            and in the top an option to see what variables were in the model and their estimates

            Input variables

            -hist_variables:  ['var1,var2',..]
                the variables you want to show in the histogram,
                you can either show one or more variables

            -what_to_show: is standard set as 'd' for decomposition, other options are:
                    * 'd' for decomposition
                    * 'p' for p values
                    * 'b' for betas
                    * 'r' for roi's

            -variables_to_show: ['var1,var2',..]
                the variables that you see when hovering over the graph

            -marginal: is standard set as 'rug, but can also be: `box`, `violin`
                The way the graph in the top is visualised (box and violin are
                statistical presentations)

            '''

        pio.renderers.default = 'browser'

        if what_to_show == 'd':
            dataset = self.decomposition

        if what_to_show == 'p':
            dataset = self.p_values

        if what_to_show == 'b':
            dataset = self.betas

        if what_to_show == 'r':
            dataset = self.model_roi

        color_list = []
        for var in hist_variables:
            color_list.append([var] * len(dataset))
        color = np.concatenate(color_list)

        variable_value = pd.Series(dataset[hist_variables].values.ravel('F'))

        dataset_total = pd.DataFrame()
        for i in range(len(hist_variables)):
            dataset_total = dataset_total.append(dataset)

        dataset_total.index = range(len(dataset_total))
        dataset_total['variable_value'] = variable_value
        dataset_total['color'] = color

        if len(variables_to_show) == 0:
            hover_data = dataset_total.columns
        else:
            hover_data = variables_to_show
        fig = px.histogram(dataset_total, x="variable_value", color="color", marginal='rug', barmode="overlay",
                     hover_data=hover_data)
        fig.show()

    def x_beta_fitted_actual(self, model):
        beta = model.fit().params
        X = pd.DataFrame(model.exog, columns=beta.index)
        x_beta = X * 0
        model_fit = model.fit()
        for i in range(len(X)):
            for j in range(len(model_fit.params)):
                x_beta.iloc[i, j] = model.exog[i, j] * model_fit.params[j]
                # negatieve waardes van intercept halen
        sum_negatives = []
        for i in range(x_beta.shape[0]):
            negative_value = 0
            for j in range(x_beta.shape[1]):
                if x_beta.iloc[i, j] < 0:
                    negative_value += x_beta.iloc[i, j]
            sum_negatives.append(negative_value)
        sum_negatives_pd = pd.DataFrame(sum_negatives)
        x_beta_intercept_correctie = x_beta.copy()
        x_beta_intercept_correctie.Intercept = x_beta_intercept_correctie.Intercept + sum_negatives_pd

        return x_beta, x_beta_intercept_correctie

    def decomposition_graph(self, var_date, model, color_kpi='deepskyblue'):
        '''
        This function has as output a bar plot with all the variables and
        the actual y used

        Input variables

            -model: model
                The model that you want to use
        '''
        pio.renderers.default = 'browser'

        data_decomp = self.x_beta_fitted_actual(model)[1]

        # x = list(range(1,len(data_decomp)+1))
        x = var_date
        variabelen = data_decomp
        names = list(data_decomp)
        y_model = model.fit().fittedvalues

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_model, name='y_model'))

        fig.add_trace(go.Scatter(x=var_date, y=model.endog, name=model.endog_names,
                                 line_color=color_kpi))

        for i in range(0, variabelen.shape[1]):
            fig.add_trace(go.Bar(x=x, y=variabelen[names[i]], name=names[i]))

        fig.update_layout(barmode='relative', title_text='Decompositie', bargap=0)

        # AXIS

        fig.show()
        # return fig

    def actual_vs_fit_graph(self, var_date, model, color_kpi='deepskyblue', color_fit='dimgray'):
        '''

        This function has as output a graph of 2 lines, the actual kpi values
        and the fitted values from the model

        Input variables

            -var_date:  [df.date]
               this variable contains the dates you want to show in the graph

            -model: model
                The model that you want to use

            -color_kpi: 'red'
                here you can give a color to the kpi like 'yellow' or 'grey'...
                as default it is put on deepskyblue'
            -color_fit: 'red'
                here you can give a color to the fit like 'yellow' or 'grey'...
                as default it is put on dimgray'
        '''
        pio.renderers.default = 'browser'

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=var_date, y=model.endog, name=model.endog_names,
                                 line_color=color_kpi))

        fig.add_trace(go.Scatter(x=var_date, y=model.fit().fittedvalues,
                                 name=model.endog_names + "_fitted",
                                 line_color=color_fit))

        fig.update_layout(title_text='Actual vs model',
                          xaxis_rangeslider_visible=True)
        fig.show()
        # return fig

    def VIF(self, model):
        X = pd.DataFrame(model.exog)
        vif = pd.Series([variance_inflation_factor(X.values, i)
                         for i in range(X.shape[1])], index=model.exog_names)
        return vif

    def check_variables_to_add(self, model):
        '''

        A function that uses all variables in de data used to model (incl. decays if added by media),
        to see what the effect is of adding one variable to the current chosen model.

        input:
            -a model

        returns:
            -AdjR2 change: the adjusted R2 from the model - the adjusted R2 from the model with the variable added.
                Keep in mind that this can be negative if the new variable doesn't add enough to the explained variance
                If you find variables that have a high number, they might be worth adding to the model

            -DW change: the DW from the model - the DW from the model with the variable added.
                Keep in mind that we want this variable to be as close as possible to 2.
                However, we expect it usually when it is between 1.6 and 2.4

            -'JB change': the JB from the model - the JB from the model with the variable added.
                Keep in mind that we want this variable (p_value) to be as big as possible (or usually bigger than 0.05).
                It is a H0 test to see if the error terms are not normally distributed

        '''

        # only use the variables that are not yet in the model and have no na's
        used = list(model.exog_names)
        used.append(self.sales)
        used.append(list(model.endog_names))
        all_variables = list(self.data_models.dropna(axis=1, how='all')._get_numeric_data().columns)

        var_to_test = [x for x in all_variables if x not in used]

        df_results = pd.DataFrame(columns=['AdjR2 change', 'DW change', 'JB change'], index=var_to_test)

        old_model = self.model_characteristics(model)
        old_model_adjr2 = old_model['AdjR2']
        old_model_dw = old_model['DW']
        old_model_jb = old_model['JB']

        formule = model.formula
        # new model
        for var in var_to_test:
            new_formule = formule + ' +' + var
            try:
                new_model_results = smf.ols(formula=new_formule, data=self.data_models)

                new_model = self.model_characteristics(new_model_results)
                new_model_adjr2 = new_model['AdjR2']
                new_model_dw = new_model['DW']
                new_model_jb = new_model['JB']
                results = [new_model_adjr2 - old_model_adjr2, new_model_dw - old_model_dw, new_model_jb - old_model_jb]
                df_results.loc[var, :] = results
            except:
                print('Something goes wrong with variable: ', var)

        return df_results.sort_values(by=['AdjR2 change'], ascending=False)

    def roi_curve_media_channels_plot(self, model, steps=10000, max_budget=1000000,
                                      index_kosten=[]):  # ,steps=5000,max_budget=3000000,index_kosten=[]
        '''
            Function that plots the ROI curve of the media used in the model. It uses the beta/decay/resonse curve from the model.
            It uses the media variable that is used in the model, but then step by step uses more budget ONLY in weeks that media was used

            Returns: a plot with ROI and turnover over different investments

            Input:
                -Model, the model you want to look at
                -steps: how much do you want to try in each new step to calculate the ROI and turnover
                        the smaller this number the more simulations and also takes longer but maybe more accurate
                        it is set at 10000
                -max_budget: how far you want the steps to go, so how much budget you want to do the calculations over.
                             it is set at 1000000
                -index_kosten: this variable can contain the weekly indexes for the price of media, if not given, each week is treated the same.

        '''

        results_ = self.roi_curve_media_channel(model, steps, max_budget, index_kosten=[])
        pio.renderers.default = 'browser'

        fig = go.Figure()

        for media_var_name in list(self.media.keys()):
            results = results_[media_var_name]

            x = results[2]
            roi_ = results[0]
            turnover_ = results[1]

            fig.add_trace(go.Scatter(x=x, y=roi_,
                                     name='ROI ' + media_var_name))

            fig.add_trace(go.Scatter(x=x, y=turnover_,
                                     name='Turnover ' + media_var_name))

            b = results_[media_var_name][2]
            costs_ = self.media[media_var_name]['cost'][0]
            index_ = min(range(len(b)), key=lambda i: abs(b[i] - costs_))
            rois_ = results_[media_var_name][0][index_]

            fig.add_shape(
                # Line Vertical min
                dict(
                    type="line",
                    x0=costs_,
                    y0=0,
                    x1=costs_,
                    y1=rois_,
                    line=dict(
                        color="Orange",
                        width=4,
                        dash="dot",
                    )
                ))
            #        # Create scatter trace of text labels
            fig.add_trace(go.Scatter(
                x=[costs_],
                y=[rois_ * 0.9],
                text=["Current investment"],
                mode="text",
                name="Current values",
            ))

        fig.update_layout(title='Impact amount of investment',
                          xaxis_title=' Investment',
                          yaxis_title='ROI / turnover')

        fig.show()

    def roi_curve_media_channel(self, model, steps=5000, max_budget=30000, index_kosten=[]):
        '''
            Function that returns the ROI curve of the media used in the model. It uses the beta/decay/resonse curve from the model.
            It uses the media variable that is used in the model, but then step by step uses more budget ONLY in weeks that media was used

            Returns: a dataframe with ROI and turnover over different investments

            Input:
                -Model, the model you want to look at
                -steps: how much do you want to try in each new step to calculate the ROI and turnover
                        the smaller this number the more simulations and also takes longer but maybe more accurate
                        it is set at 10000
                -max_budget: how far you want the steps to go, so how much budget you want to do the calculations over.
                             it is set at 1000000
                -index_kosten: this variable can contain the weekly indexes for the price of media, if not given, each week is treated the same.

        '''
        final_results = {}

        for media_var_name in list(self.media.keys()):

            #    media_var_name = list(media.keys())[1]
            media_var = self.data[media_var_name]
            cost_media_var = self.media[list(self.media.keys())[0]]['cost'][0]

            beta_, decay_, alpha_ = get_beta_dec_alp(model, media_var_name)

            total_media = media_var.sum()
            cost_per_media_grp = cost_media_var / total_media
            inzet = steps
            curve_ = self.media[media_var_name]['curve']

            # weken met inzet:
            weeks = list(media_var[media_var > 0].index)
            # kosten per week
            if not any(index_kosten):
                index_kosten = pd.Series([1] * len(media_var), index=media_var.index)

            cost_per_grp = cost_per_media_grp * index_kosten

            dec_var = decay_fun(media_var, decay_)
            max_value_decy = dec_var.max()

            media_df = pd.DataFrame([0] * len(media_var), index=media_var.index)
            testing_media_df = media_df.copy()
            inzet_totaal = 0
            inzet = steps
            # Tot een bepaald budget blijven inzetten
            turnover_list = []
            inzet_totaal_list = []
            best_media_list = []
            runs = int(max_budget / steps) + (max_budget % steps > 0)

            for i in tqdm(range(runs)):

                #    while inzet_totaal < max_budget:

                inzet_totaal += inzet
                turnover = 0

                # Alle mogelijke weken proberen
                for week in weeks:

                    testing_media_df = media_df.copy()  # begin zoals in het week - 1 was
                    testing_media_df.iloc[week, :] += (
                                inzet / cost_per_grp[week])  # toevoegen van budget in week uit weeks

                    dec_var = decay_fun(testing_media_df[0], decay_)  # Var om in de curve te stoppen
                    #        turnover = beta_*s_curve(dec_var,alpha_).sum()

                    # als er hogere turnver is, dan deze als max zetten en ook nieuwe nieuwe best_media
                    if curve_ == 's_curve':
                        if turnover < beta_ * s_curve_after_model(dec_var, alpha_, max_value_decy).sum():
                            turnover = beta_ * s_curve_after_model(dec_var, alpha_, max_value_decy).sum()
                            best_media = testing_media_df.copy()
                    if curve_ == 'diminishing_curve':
                        if turnover < beta_ * diminishing_curve_after_model(dec_var, alpha_, max_value_decy).sum():
                            turnover = beta_ * diminishing_curve_after_model(dec_var, alpha_, max_value_decy).sum()
                            best_media = testing_media_df.copy()

                turnover_list.append(turnover)
                inzet_totaal_list.append(inzet_totaal)
                media_df = best_media.copy()
                best_media_list.append(media_df)

            turnover_np = np.array(turnover_list, dtype=np.float)
            inzet_np = np.array(inzet_totaal_list, dtype=np.float)

            roi_list = turnover_np / inzet_np

            final_results[media_var_name] = roi_list, turnover_list, inzet_totaal_list

        return final_results


def s_curve(var, alpha_):
    s_curve_values = alpha_ * (var / max(var)) * (1 - np.exp(-alpha_ * (var / max(var)))) / (
                1 + alpha_ * (var / max(var)))
    return s_curve_values


def decay(var, labda_):
    decayVariable = []
    decayVariable.append(var.values[0])
    for i in range(1, len(var)):
        decayVariable.append(var.values[i] + labda_ * decayVariable[i - 1])
    return pd.Series(data=decayVariable, dtype=float)


def diminishing_curve(var, alpha_):
    diminishing_curve_values = 1 - np.exp(-var * alpha_ / max(var))
    return diminishing_curve_values


def df_media(media, df):
    df_totaal = pd.DataFrame()
    media_variables = list(media.keys())
    for media_var in media_variables:
        for decay_value in media[media_var]['decay']:
            dec_var = media_var + '__' + str(decay_value).replace('.', '_')
            dec_values = decay(df[media_var], decay_value)
            df_totaal[dec_var] = dec_values
    return df_totaal.set_index(df.index)


def variables_decay_rc_names(media):  # per media var een list
    variables_name_decay = {}
    media_variables = list(media.keys())
    decays = []
    curves = []
    for media_var in media_variables:
        curves2 = []
        type_curve = media[media_var]['curve']
        for decay_value in media[media_var]['decay']:
            dec_var = media_var + '__' + str(decay_value).replace('.', '_')
            decays.append(dec_var)
            for alpha_ in media[media_var]['alpha']:
                curves_var = str(type_curve) + '(' + dec_var + ',' + str(alpha_) + ')'
                curves.append(curves_var)
                curves2.append(curves_var)
        variables_name_decay[media_var] = curves2
    return decays, curves, variables_name_decay


def get_alp_dec_from_name(name_: str):
    t2 = name_[name_.index("__") + 2:]
    decay_ = t2[0:t2.index(',')]
    alpha_ = t2[t2.index(',') + 1: t2.index(')')]
    return decay_, alpha_


def use_roi(media):
    '''check if we should use ROI (media cost are available'''
    roi_yes = 0
    if len(media) == 0:
        return roi_yes
    else:
        roi_yes = 0
        count_ = 0
        for media_var in list(media.keys()):
            try:
                count_ += len(media[media_var]['cost'])
            except:
                count_ += 0

        if len(list(media.keys())) == count_:
            roi_yes = 1

        return roi_yes


def create_roi_df(df):
    results = []
    for i in range(len(df)):
        results.append(list(df['Media type ROI'][i].values()))
    df_roi = pd.DataFrame(results, columns=list(df['Media type ROI'][0].keys()))
    df_roi['Total ROI'] = df['Total ROI']
    return df_roi


def s_curve_after_model(var, alpha_, max_value):
    s_curve_values = alpha_ * (var / max_value) * (1 - np.exp(-alpha_ * (var / max_value))) / (
                1 + alpha_ * (var / max_value))
    return s_curve_values


def diminishing_curve_after_model(var, alpha_, max_value):
    diminishing_curve_values = 1 - np.exp(-var * alpha_ / max_value)
    return diminishing_curve_values


def decay_fun(var, labda_):
    decayVariable = []
    decayVariable.append(var.values[0])
    for i in range(1, len(var)):
        decayVariable.append(var.values[i] + labda_ * decayVariable[i - 1])
    return pd.Series(data=decayVariable, dtype=float)


def get_beta_dec_alp(model, med_var):
    '''returns the beta / dec / alp of a media variable in a certrain model'''
    estimates_ = model.fit().params
    beta_ = estimates_.filter(like=med_var, axis=0)[0]
    var_name = estimates_.filter(like=med_var, axis=0).index[0]
    dec_alp = get_alp_dec_from_name(var_name)
    dec_ = float('0.' + dec_alp[0][dec_alp[0].index("_") + 1:])
    alp_ = float(dec_alp[1])
    return beta_, dec_, alp_
