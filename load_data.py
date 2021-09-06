import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from collections import namedtuple
from sklearn.preprocessing import StandardScaler

'''
all four data loading function below recycle much of the code found within https://github.com/jmikko/fair_ERM/blob/master/load_data.py


@inproceedings{donini2018empirical,
  title={Empirical risk minimization under fairness constraints},
  author={Donini, Michele and Oneto, Luca and Ben-David, Shai and Shawe-Taylor, John S and Pontil, Massimiliano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={2791--2801},
  year={2018}
}


the 'load_mortgage' function uses a single line code snippet from https://github.com/askoshiyama/audit_mortgage/blob/master/Modelling.ipynb
'''

def load_adult(smaller=False, scaler=True, protectedAttributes=None):
    
    # the protectedAttributes argument is for the passing in of a dictionary with keys 'race', gender' say, and values 'White', 'Male'
    # where the values are the non-protected groups for each key
    
    '''
    :param smaller: selecting this flag it is possible to generate a smaller version of the training and test sets.
    :param scaler: if True it applies a StandardScaler() (from sklearn.preprocessing) to the data.
    :return: train and test data.

    Features of the Adult dataset:
    0. age: continuous.
    1. workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked.
    2. fnlwgt: continuous.
    3. education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th,
    Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
    4. education-num: continuous.
    5. marital-status: Married-civ-spouse, Divorced, Never-married, Separated, Widowed,
    Married-spouse-absent, Married-AF-spouse.
    6. occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty,
    Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,
    Protective-serv, Armed-Forces.
    7. relationship: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried.
    8. race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.
    9. sex: Female, Male.
    10. capital-gain: continuous.
    11. capital-loss: continuous.
    12. hours-per-week: continuous.
    13. native-country: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc),
    India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,
    Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala,
    Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.
    (14. label: <=50K, >50K)
    '''
    
    dataPath = '../datasets/adult/'
    
    data = pd.read_csv(
        dataPath+"/adult.data",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        sep=', ',engine='python'
            )
    
    print('training: {}'.format(len(data.values[:,-1])))
    
    # note that in the adult.test file downloaded, there was an initial line containing "|1x3 Cross validator"
    # this line was removed by hand prior to processing (note that other, older downloads of this file did
    # not contain this line)
    
    data_test = pd.read_csv(
        dataPath+"adult.test",
        names=[
            "Age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "gender", "capital gain", "capital loss",
            "hours per week", "native-country", "income"],
        sep=', ',engine='python'
    )
    
    print('testing: {}'.format(len(data_test.values[:,-1])))
    
    data = pd.concat([data, data_test])
    
    len_train = len(data.values[:, -1])
    
    # Considering the relative low portion of missing data, we discard rows with missing data
    domanda = data["workclass"][4].values[1]
    data = data[data["workclass"] != domanda]
    data = data[data["occupation"] != domanda]
    data = data[data["native-country"] != domanda]
    # Here we apply discretisation on column marital_status
    data.replace(['Divorced', 'Married-AF-spouse',
                  'Married-civ-spouse', 'Married-spouse-absent',
                  'Never-married', 'Separated', 'Widowed'],
                 ['not married', 'married', 'married', 'married',
                  'not married', 'not married', 'not married'], inplace=True)
    
    # Here we remove the superfluous full-stops from the income column that is in the test data
    # [it tooks hours to find out that this was causing incorrect model results!!!]
    data.replace(['<=50K.', '>50K.'],
                 ['<=50K', '>50K'], inplace=True)
    
    # categorical fields
    category_col = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']
    
    dataOld = data.copy()
    
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    print(datamat.shape)
    target = np.array([-1.0 if val == 0 else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    print(datamat.shape)
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    
    # for each protected attribute, replaces the non-protected group with 1, 0 o/w
    if protectedAttributes is not None:
        for protectedAttribute in protectedAttributes:
            dfProtectedAtrributeValues = dataOld[protectedAttribute].values
            arrayProtectedAttributeValues = np.where(dfProtectedAtrributeValues==protectedAttributes[protectedAttribute],1,0)
            print('{}: {}'.format(protectedAttribute, dataOld.columns.get_loc(protectedAttribute)))        
            datamat[:,dataOld.columns.get_loc(protectedAttribute)]=arrayProtectedAttributeValues
    
    if smaller:
        print('A smaller version of the dataset is loaded...')
        X = datamat[:len_train // 20, :]
        y = target[:len_train // 20]
    else:
        print('The dataset is loaded...')
        X = datamat[:len_train, :]
        y = target[:len_train]
    
    return X,y


def load_bank(smaller=False, scaler=True):
    
    # the protectedAttributes argument is for the passing in of a dictionary with keys 'race', gender' say, and values 'White', 'Male'
    # where the values are the non-protected groups for each key
    
    '''
    Input variables:
    # bank client data:
    1 - age (numeric)
    2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
    3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
    4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
    5 - default: has credit in default? (categorical: 'no','yes','unknown')
    6 - housing: has housing loan? (categorical: 'no','yes','unknown')
    7 - loan: has personal loan? (categorical: 'no','yes','unknown')
    # related with the last contact of the current campaign:
    8 - contact: contact communication type (categorical: 'cellular','telephone') 
    9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
    10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
    11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
    # other attributes:
    12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
    13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
    14 - previous: number of contacts performed before this campaign and for this client (numeric)
    15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')
    # social and economic context attributes
    16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
    17 - cons.price.idx: consumer price index - monthly indicator (numeric) 
    18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric) 
    19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
    20 - nr.employed: number of employees - quarterly indicator (numeric)

    Output variable (desired target):
    21 - y - has the client subscribed a term deposit? (binary: 'yes','no')
    '''
    
    dataPath = '../datasets/bank/'
    
    data = pd.read_csv(dataPath+"/bank-additional-full.csv", sep=';',engine='python')
    len_train = len(data.values[:, -1])
    
    # Here we apply discretisation on column marital_status
#     data.replace(['divorced', 'single'], ['not married', 'not married'], inplace=True)
    
    # categorical fields
    category_col = ['job', 'marital', 'education', 'default', 'housing', 'loan',
       'contact', 'month', 'day_of_week', 'poutcome']    
    
    dataOld = data.copy()
    
    for col in category_col:
        b, c = np.unique(data[col], return_inverse=True)
        data[col] = c
    datamat = data.values
    print(datamat.shape)
    target = np.array([-1.0 if val == 'no' else 1.0 for val in np.array(datamat)[:, -1]])
    datamat = datamat[:, :-1]
    print(datamat.shape)
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    
    # for the protected attribute 'age', treat those for which 25<=age<=60 as the protected group, and non-protected otherwise (cf. Zafar et al.)
    protectedAttribute = 'age'
    
    dfProtectedAtrributeValues = dataOld[protectedAttribute].values
    arrayProtectedAttributeValues = np.array([0 if val >= 25 and val <= 60 else 1 for val in np.array(dfProtectedAtrributeValues)])
    print('{}: {}'.format(protectedAttribute, dataOld.columns.get_loc(protectedAttribute)))
    datamat[:,dataOld.columns.get_loc(protectedAttribute)]=arrayProtectedAttributeValues
        
    if smaller:
        print('A smaller version of the dataset is loaded...')
#         data = namedtuple('_', 'data, target')(datamat[:len_train // 20, :], target[:len_train // 20])
#         data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])
        X = datamat[:len_train // 20, :]
        y = target[:len_train // 20]
    
    else:
        print('The dataset is loaded...')
#         data = namedtuple('_', 'data, target')(datamat[:len_train, :], target[:len_train])
#         data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])
        X = datamat[:len_train, :]
        y = target[:len_train]
    
#     return data
    return X,y


def load_german(smaller=False, scaler=True):

    dataPath = '../datasets/german/'

    data = pd.read_csv(dataPath+"/proc_german_num_02 withheader-2.csv", sep=',',engine='python')
    len_train = len(data.values[:, -1])
    
    # convert PersonStatusSex column into binary gender so that the non-protected group (male) is 1, and 0 o/w
    data['PersonStatusSex'].replace([1,2,3,4,5], [1,0,1,1,0], inplace=True)

    # swap flags around on ForeignWorker column so that the non-protected group (ForeignWorker=false) is 1, and 0 o/w
    data['ForeignWorker'].replace([2,1], [1,0], inplace=True)

    dataOld = data.copy()
    
    datamat = data.values
    print(datamat.shape)
    target = np.array([1.0 if val == 1 else -1.0 for val in np.array(datamat)[:, 0]])
    datamat = datamat[:, 1:]
    print(datamat.shape)

    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
        
    # put 0/1 protected attributes flags back in after scaling
    protectedAttributes = ['PersonStatusSex','ForeignWorker']
    for protectedAttribute in protectedAttributes:
        
        dfProtectedAtrributeValues = dataOld[protectedAttribute].values
        print('{}: {}'.format(protectedAttribute, dataOld.columns.get_loc(protectedAttribute)-1))
        datamat[:,dataOld.columns.get_loc(protectedAttribute)-1]=np.array(dfProtectedAtrributeValues)
        
    protectedAttribute = 'AgeInYears'
    dfProtectedAtrributeValues = dataOld[protectedAttribute].values
    arrayProtectedAttributeValues = np.array([0 if val >= 25 and val <= 60 else 1 for val in np.array(dfProtectedAtrributeValues)])
    print('{}: {}'.format(protectedAttribute, dataOld.columns.get_loc(protectedAttribute)-1))
    datamat[:,dataOld.columns.get_loc(protectedAttribute)-1]=arrayProtectedAttributeValues
    
    
    if smaller:
        print('A smaller version of the dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train // 20, :], target[:len_train // 20])
#         data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])
        X = datamat[:len_train // 20, :]
        y = target[:len_train // 20]
    else:
        print('The dataset is loaded...')
        data = namedtuple('_', 'data, target')(datamat[:len_train, :], target[:len_train])
#         data_test = namedtuple('_', 'data, target')(datamat[len_train:, :], target[len_train:])
        X = datamat[:len_train, :]
        y = target[:len_train]
    
#     return data
    return X,y


def load_mortgage(smaller=False, scaler=True, protectedCategoricalAttributes=None, protectedNumericalAttributes=None):

    dataPath = '../datasets/mortgage/'
    
    data = pd.read_pickle(dataPath+"mortgage_data_balanced.pkl.gz")
    
    len_train = len(data.values[:, -1])
    
    keep_vars = ['respondent_id', 'as_of_year', 'agency_abbr', 'loan_type_name', 'loan_amount_000s', 'owner_occupancy_name',
             'loan_purpose_name', 'property_type_name', 'preapproval_name', 'msamd_name', 'state_abbr', 'county_name',
             'applicant_ethnicity_name', 'co_applicant_ethnicity_name', 'applicant_race_name_1', 'co_applicant_race_name_1',
             'applicant_sex_name', 'co_applicant_sex_name', 'applicant_income_000s', 'purchaser_type_name', 
             'denial_reason_name_1', 'hoepa_status_name', 'lien_status_name', 'population', 'minority_population',
             'hud_median_family_income', 'tract_to_msamd_income', 'number_of_owner_occupied_units', 
             'number_of_1_to_4_family_units', 'action_taken_name']
    
    data = data[keep_vars].copy()
    
    dataOld = data.copy()
    
    numericalCols = data.describe().columns
    
    # convert non-numerical columns into numerical columns
    for col in data.columns:
        if col not in numericalCols:
            b, c = np.unique(np.array(data[col],dtype='str'), return_inverse=True)
            data[col] = c
    
    datamat = data.values
    print(datamat.shape)
    target = np.array([1.0 if val == 'Loan originated' else -1.0 for val in np.array(dataOld)[:, -1]])
    
    datamat = datamat[:, :-1]
    print(datamat.shape)
    if scaler:
        scaler = StandardScaler()
        scaler.fit(datamat)
        datamat = scaler.transform(datamat)
    
    # for each protected attribute, replaces the non-protected group with 1, 0 o/w
    if protectedCategoricalAttributes is not None:
        print('Protected Categorical Attributes:')
        for protectedAttribute in protectedCategoricalAttributes:
            dfProtectedAtrributeValues = dataOld[protectedAttribute].values
            arrayProtectedAttributeValues = np.where(dfProtectedAtrributeValues==protectedCategoricalAttributes[protectedAttribute],1,0)
            print('{}: {}'.format(protectedAttribute, dataOld.columns.get_loc(protectedAttribute)))        
            datamat[:,dataOld.columns.get_loc(protectedAttribute)]=arrayProtectedAttributeValues
        
    if protectedNumericalAttributes is not None:
        print('Protected Numerical Attributes:')
        for protectedAttribute in protectedNumericalAttributes:
            print('{}: {}'.format(protectedAttribute, dataOld.columns.get_loc(protectedAttribute)))    
    
    if smaller:
        print('A smaller version of the dataset is loaded...')
#         data = namedtuple('_', 'data, target')(datamat[:len_train // 20, :], target[:len_train // 20])
        X = datamat[:len_train // 20, :]
        y = target[:len_train // 20]
        
    else:
        print('The dataset is loaded...')
#         data = namedtuple('_', 'data, target')(datamat[:len_train, :], target[:len_train])
        X = datamat[:len_train, :]
        y = target[:len_train]
        
#     return data
    return X,y