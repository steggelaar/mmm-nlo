from multiple_regressions import *
import pandas as pd
##
'''read the data'''
df = pd.read_csv('MMM_data.csv', delimiter = ',', decimal = '.', encoding = 'utf-8', index_col=False)
#or:
# url = 'https://raw.githubusercontent.com/SimonTeg/multiple_regressions/main/MMM_data.csv'
# df = pd.read_csv(url, error_bad_lines=False)

'''The variable you want to explain:'''
kpi = 'sales'

'''Variales that are in every model:'''
used_variables = ["holiday_summer",'holiday_winter',"promo"]

'''Variales that are tested:'''
optional_variables = ["trend","promo_competitor_1","promo_competitor_2"]

'''One variale is each time tested from the list:'''
or_variables ={'eco': [ 'consumer_trust', 'consumer_buying_intention']}

'''Media variales that are tested in each form'''
media = {'tv':
            {'curve':'s_curve',
              'alpha':[5,10,15],
              'decay':[0.1,0.4]
                ,
              'cost':[2000]
              },
        'radio':
            {'curve':'s_curve',
              'alpha':[1,5,10],
              'decay':[0.2,0.4]
                ,
              'cost':[1000]
              }
            }

'''signs of the estimates, if not right, the model will be left out'''
positive_signs = []
negative_signs = []
media_positive = 'Yes'

conditions = {'P_value_min':0.99,'AdjR2_min': 0.0, 'DW_min': 0.0, 'DW_max': 4.0, 'JB_min' : 0.0}

'''Setting up the models:'''
mmm_models = MultipleRegressions(data = df
                            ,sales = kpi
                            ,used_variables = used_variables
                           ,optional_variables = optional_variables
                           ,or_variables = or_variables
                           ,media = media
                           ,conditions = conditions
                           ,positive = positive_signs, negative = negative_signs, media_positive = media_positive
                            )

mmm_models.run_all_models()
'''Restrictions on the model:'''
##
'''shows the response curves'''
#help(mmm_models.plot_curves)
mmm_models.plot_curves(media_var = df.tv, alphas_scurve = [1,5,20,100], alphas_diminishing_curve = [20])
##
'''shows the decays used in media'''
mmm_models.plot_decay(date = df.date)
##
'''shows the decays and response curves used in media'''
mmm_models.plot_decay_and_curves(date = df.date)
##
'''how many models to be tested:'''
mmm_models.models_to_be_tested()
##
'''The formulas that are tested:'''
mmm_models.create_formulas()
##
'''The variables that are used:'''
mmm_models.all_variables_used()
##
'''A function to start running the models:'''
mmm_models.run_all_models()
##
'''models rejected'''
mmm_models.rejected_model_statistics
##
'''Get the results from media:'''
media_resuts = mmm_models.get_model_values_media()
media_resuts['tv']
##
'''Get the best N results from media:'''
best_media_resuts = mmm_models.best_media_table(N=3)
best_media_resuts['tv']
##
'''Shows the N best results from media or a media variable:'''
mmm_models.plot_media_curves_results(N=1,media_variables = ['radio','tv'])
##
'''Shows the histogram of the results'''
fig = mmm_models.histogram_extra_info(hist_variables = ['tv'], what_to_show = 'r', marginal = 'rug')
fig.show()
##
'''Shows the histogram of the results'''
fig2 = mmm_models.histogram_extra_info(hist_variables = ["promo"], what_to_show = 'd', variables_to_show = ['holiday_winter'], marginal = 'rug')
fig2.show()
##
'''All model statistics'''
modellen = mmm_models.model_statistics.sort_values(by=['AdjR2'], ascending=False) #ROI TOEVOEGEN?
modellen
##
'''Best formula'''
formule = mmm_models.model_statistics.sort_values(by=['AdjR2'], ascending=False).iloc[0,3]
formule

##
'''Going to one model'''
model = smf.ols(formula=formule, data=mmm_models.data_models)
model_fit = model.fit()
model_fit.summary()
##
'''Decomposition of a model'''
mmm_models.decomposition_graph(df.date,model)
##
'''Fit vs actual'''
mmm_models.actual_vs_fit_graph(df.date,model)
##
'''ROI curve depending on investment''' #aanpassen current investment bij voorbeeld
mmm_models.roi_curve_media_channels_plot(model,steps=10000,max_budget=500000)
##
'''VIF'''
mmm_models.VIF(model)
##
'''Bekijken welke variabelen misschien toevoegen''' #KPI eruit halen
#mmm_models.model_characteristics(model)
add_var_df = mmm_models.check_variables_to_add(model)
add_var_df
