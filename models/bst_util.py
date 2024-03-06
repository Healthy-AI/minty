import numpy as np
import pandas as pd
import xgboost as xgb
import sys

def df_missingness_reliance(D_tree, X):
    """
    Computes the fraction of rows in a dataframe or numpy array X for which the ensemble model specified by 
    the dataframe df makes use of a "missing" or "default" edge in prediction. 
    
    args: 
        df (dataframe specifying tree ensemble)      : The model to evaluate
        X (ndarray or dataframe) : The data for which to compute the missingness reliance
    
    """

    # Get estimators in dataframe representation 
    n_trees = D_tree['Tree'].max()+1
    
    if isinstance(X, np.ndarray):
        feat_id = dict([(i,i) for i in range(X.shape[1])] + [('f%d'%i,i) for i in range(X.shape[1])])
    else:
        feat_id = dict([(str(X.columns[i]), i) for i in range(X.shape[1])] + [('f%d'%i,i) for i in range(X.shape[1])])
        X = X.values
        
    # Construct dictionaries for looking up tree components
    children = dict([(r[0], (r[1], r[2])) for r in D_tree[['ID', 'Yes', 'No']].values])
    features = dict([(r[0], r[1]) for r in D_tree[['ID', 'Feature']].values])
    thresholds = dict([(r[0], r[1]) for r in D_tree[['ID', 'Split']].values])

    # Iterate over trees keeping track of which rows have missing features used by the trees
    miss_rows = []
    for tree in range(n_trees):

        # Compute (recursively) the rows with missing features
        def check_missing_(node, I, X, children, features, thresholds):
            f = features[node]

            # Return empty list if leaf or no observations left
            if f == 'Leaf' or len(I)<1:
                return []

            fid = feat_id[f]
            t = float(thresholds[node])
            
            # Left and right children
            left = children[node][0]
            right = children[node][1]

            # Rows with missing and observed feature for this node
            X = X.astype(float) #added to transform all values to float, especially NaNs and onehotencoded columns
            I_na = np.where(np.isnan(X[:,fid]))[0]
            I_o = np.where(~np.isnan(X[:,fid]))[0]
            
            # Compute the observations for left and right children
            X_o = X[I_o,]
            I_r = I_o[np.where(X_o[:,fid]<t)[0]]
            I_l = I_o[np.where(X_o[:,fid]>=t)[0]]
            
            # Maintain list of pairs of observations and missing feature
            miss_feat = [(I[i], fid) for i in I_na]
            miss_l = check_missing_(left, I[I_l], X[I_l,], children, features, thresholds)
            miss_r = check_missing_(right, I[I_r], X[I_r,], children, features, thresholds)

            return miss_feat + miss_l + miss_r
        
        # Get the root of the current tree
        node = '%d-0' % tree
        
        miss_feat = check_missing_(node, np.arange(X.shape[0]), X, children, features, thresholds)
        
        # Add to a list of rows with missing features
        miss_rows += [r[0] for r in miss_feat]

    # Compute the reliance on missing features
    imp_rel = len(np.unique(miss_rows))/X.shape[0]

    return imp_rel


def bst_missingness_reliance(bst, X):
    """
    Computes the fraction of rows in a dataframe or numpy array X for which the xgboost model in 
    bst makes use of a "missing" or "default" edge in prediction. 
    
    args: 
        bst (xgboost model)      : The model to evaluate
        X (ndarray or dataframe) : The data for which to compute the missingness reliance
    
    """

    # Get estimators in dataframe representation 
    D_tree = bst._Booster.trees_to_dataframe()

    return df_missingness_reliance(D_tree, X)


def dt_missingness_reliance(dt, X):
    """
    Computes the fraction of rows in a dataframe or numpy array X for which the decision tree model in 
    bst makes use of a "missing" or "default" edge in prediction. 
    
    args: 
        dt (decision tree model)      : The model to evaluate
        X (ndarray or dataframe) : The data for which to compute the missingness reliance
    
    """

    n_nodes = dt.tree_.node_count

    children_left  = ['0-%d'%c if c>-1 else np.nan for c in dt.tree_.children_left]
    children_right = ['0-%d'%c if c>-1 else np.nan for c in dt.tree_.children_right]
    
    if isinstance(X, np.ndarray):
        feature = ['Leaf' if i==-2 else 'f%d'%i for i in dt.tree_.feature]
    else:
        feature = ['Leaf' if i==-2 else X.columns[i] for i in dt.tree_.feature]
        
    threshold = dt.tree_.threshold
    values = dt.tree_.value
    trees = np.zeros(n_nodes).astype(int)
    node_ids = ['0-%d' % i for i in range(n_nodes)]


    df = pd.DataFrame({'Tree': trees, 'Node': range(n_nodes), 'ID': node_ids, 
                       'Feature': feature, 'Split': threshold, 'Yes': children_left, 'No': children_right})

    return df_missingness_reliance(df, X)


def rulefit_missingness_reliance(rulefit_model, X):
    """
    Computes the fraction of rows in a dataframe or numpy array X for which the RuleFit model
    makes use of a "missing" or "default" edge in prediction.
    
    Args:
        rulefit_model: The trained RuleFit model to evaluate.
        X (DataFrame or ndarray): The data for which to compute the missingness reliance.
    
    Returns:
        float: The fraction of rows relying on missing or default edges in prediction.
    """
    #TODO: differentiate with Data set is used 
    # Convert input to DataFrame if it is an ndarray
    '''columns_names = ['AGE', 'PTEDUCAT', 'FDG', 'ABETA_bl', 'TAU_bl', 'PTAU_bl', 'LDELTOTAL',
    'MMSE', 'Hippocampus_bl', 'WholeBrain_bl', 'Entorhinal', 'Fusiform',
    'ICV', 'APOE4_0.0', 'APOE4_1.0', 'APOE4_2.0', 'DX_bl_AD', 'DX_bl_CN',
    'DX_bl_EMCI', 'DX_bl_LMCI', 'DX_bl_SMC', 'PTGENDER_Female',
    'PTGENDER_Male', 'PTETHCAT_Hisp/Latino', 'PTETHCAT_Not Hisp/Latino',
    'PTETHCAT_Unknown', 'PTRACCAT_Am Indian/Alaskan', 'PTRACCAT_Asian',
    'PTRACCAT_Black', 'PTRACCAT_Hawaiian/Other PI',
    'PTRACCAT_More than one', 'PTRACCAT_Unknown', 'PTRACCAT_White',
    'PTMARRY_Divorced', 'PTMARRY_Married', 'PTMARRY_Never married',
    'PTMARRY_Unknown', 'PTMARRY_Widowed']
    
    columns_names = ['Year', 'Infant_deaths', 'Under_five_deaths', 'Adult_mortality', 'Alcohol_consumption', 'Hepatitis_B', 'Measles', 'BMI', 'Polio', 'Diphtheria', 'Incidents_HIV', 'GDP_per_capita', 'Population_mln', 'Thinness_ten_nineteen_years', 'Thinness_five_nine_years', 'Schooling', 'Economy_status_Developed', 'Economy_status_Developing', 'Region_Asia', 'Region_Central America and Caribbean', 'Region_European Union', 'Region_Middle East', 'Region_North America', 'Region_Oceania', 'Region_Rest of Europe', 'Region_South America', 'Country_Albania', 'Country_Algeria', 'Country_Angola', 'Country_Antigua and Barbuda', 'Country_Argentina', 'Country_Armenia', 'Country_Australia', 'Country_Austria', 'Country_Azerbaijan', 'Country_Bahamas, The', 'Country_Bahrain', 'Country_Bangladesh', 'Country_Barbados', 'Country_Belarus', 'Country_Belgium', 'Country_Belize', 'Country_Benin', 'Country_Bhutan', 'Country_Bolivia', 'Country_Bosnia and Herzegovina', 'Country_Botswana', 'Country_Brazil', 'Country_Brunei Darussalam', 'Country_Bulgaria', 'Country_Burkina Faso', 'Country_Burundi', 'Country_Cabo Verde', 'Country_Cambodia', 'Country_Cameroon', 'Country_Canada', 'Country_Central African Republic', 'Country_Chad', 'Country_Chile', 'Country_China', 'Country_Colombia', 'Country_Comoros', 'Country_Congo, Dem. Rep.', 'Country_Congo, Rep.', 'Country_Costa Rica', "Country_Cote d'Ivoire", 'Country_Croatia', 'Country_Cuba', 'Country_Cyprus', 'Country_Czechia', 'Country_Denmark', 'Country_Djibouti', 'Country_Dominican Republic', 'Country_Ecuador', 'Country_Egypt, Arab Rep.', 'Country_El Salvador', 'Country_Equatorial Guinea', 'Country_Eritrea', 'Country_Estonia', 'Country_Eswatini', 'Country_Ethiopia', 'Country_Fiji', 'Country_Finland', 'Country_France', 'Country_Gabon', 'Country_Gambia, The', 'Country_Georgia', 'Country_Germany', 'Country_Ghana', 'Country_Greece', 'Country_Grenada', 'Country_Guatemala', 'Country_Guinea', 'Country_Guinea-Bissau', 'Country_Guyana', 'Country_Haiti', 'Country_Honduras', 'Country_Hungary', 'Country_Iceland', 'Country_India', 'Country_Indonesia', 'Country_Iran, Islamic Rep.', 'Country_Iraq', 'Country_Ireland', 'Country_Israel', 'Country_Italy', 'Country_Jamaica', 'Country_Japan', 'Country_Jordan', 'Country_Kazakhstan', 'Country_Kenya', 'Country_Kiribati', 'Country_Kuwait', 'Country_Kyrgyz Republic', 'Country_Lao PDR', 'Country_Latvia', 'Country_Lebanon', 'Country_Lesotho', 'Country_Liberia', 'Country_Libya', 'Country_Lithuania', 'Country_Luxembourg', 'Country_Madagascar', 'Country_Malawi', 'Country_Malaysia', 'Country_Maldives', 'Country_Mali', 'Country_Malta', 'Country_Mauritania', 'Country_Mauritius', 'Country_Mexico', 'Country_Micronesia, Fed. Sts.', 'Country_Moldova', 'Country_Mongolia', 'Country_Montenegro', 'Country_Morocco', 'Country_Mozambique', 'Country_Myanmar', 'Country_Namibia', 'Country_Nepal', 'Country_Netherlands', 'Country_New Zealand', 'Country_Nicaragua', 'Country_Niger', 'Country_Nigeria', 'Country_North Macedonia', 'Country_Norway', 'Country_Oman', 'Country_Pakistan', 'Country_Panama', 'Country_Papua New Guinea', 'Country_Paraguay', 'Country_Peru', 'Country_Philippines', 'Country_Poland', 'Country_Portugal', 'Country_Qatar', 'Country_Romania', 'Country_Russian Federation', 'Country_Rwanda', 'Country_Samoa', 'Country_Sao Tome and Principe', 'Country_Saudi Arabia', 'Country_Senegal', 'Country_Serbia', 'Country_Seychelles', 'Country_Sierra Leone', 'Country_Singapore', 'Country_Slovak Republic', 'Country_Slovenia', 'Country_Solomon Islands', 'Country_Somalia', 'Country_South Africa', 'Country_Spain', 'Country_Sri Lanka', 'Country_St. Lucia', 'Country_St. Vincent and the Grenadines', 'Country_Suriname', 'Country_Sweden', 'Country_Switzerland', 'Country_Syrian Arab Republic', 'Country_Tajikistan', 'Country_Tanzania', 'Country_Thailand', 'Country_Timor-Leste', 'Country_Togo', 'Country_Tonga', 'Country_Trinidad and Tobago', 'Country_Tunisia', 'Country_Turkiye', 'Country_Turkmenistan', 'Country_Uganda', 'Country_Ukraine', 'Country_United Arab Emirates', 'Country_United Kingdom', 'Country_United States', 'Country_Uruguay', 'Country_Uzbekistan', 'Country_Vanuatu', 'Country_Venezuela, RB', 'Country_Vietnam', 'Country_Yemen, Rep.', 'Country_Zambia', 'Country_Zimbabwe']
    '''
    
    columns_names = ['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea', 'FullBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea', 'OpenPorchSF', 'MoSold', 'YrSold', 'GarageType_Attchd', 'GarageType_Basment', 'GarageType_BuiltIn', 'GarageType_CarPort', 'GarageType_Detchd', 'BldgType_2fmCon', 'BldgType_Duplex', 'BldgType_Twnhs', 'BldgType_TwnhsE', 'MasVnrType_BrkFace', 'MasVnrType_Stone', 'ExterQual_Fa', 'ExterQual_Gd', 'ExterQual_TA', 'LandContour_HLS', 'LandContour_Low', 'LandContour_Lvl', 'HeatingQC_Fa', 'HeatingQC_Gd', 'HeatingQC_Po', 'HeatingQC_TA', 'Condition2_Feedr', 'Condition2_Norm', 'Condition2_PosN', 'Condition2_RRAe', 'Condition2_RRAn', 'Condition2_RRNn', 'Exterior1st_AsphShn', 'Exterior1st_BrkComm', 'Exterior1st_BrkFace', 'Exterior1st_CBlock', 'Exterior1st_CemntBd', 'Exterior1st_HdBoard', 'Exterior1st_ImStucc', 'Exterior1st_MetalSd', 'Exterior1st_Plywood', 'Exterior1st_Stone', 'Exterior1st_Stucco', 'Exterior1st_VinylSd', 'Exterior1st_Wd Sdng', 'Exterior1st_WdShing', 'RoofMatl_CompShg', 'RoofMatl_Metal', 'RoofMatl_Roll', 'RoofMatl_Tar&Grv', 'RoofMatl_WdShake', 'RoofMatl_WdShngl', 'SaleType_CWD', 'SaleType_Con', 'SaleType_ConLD', 'SaleType_ConLI', 'SaleType_ConLw', 'SaleType_New', 'SaleType_Oth', 'SaleType_WD', 'CentralAir_Y', 'Heating_GasA', 'Heating_GasW', 'Heating_Grav', 'Heating_OthW', 'Heating_Wall', 'GarageQual_Fa', 'GarageQual_Gd', 'GarageQual_Po', 'GarageQual_TA', 'Electrical_FuseF', 'Electrical_FuseP', 'Electrical_Mix', 'Electrical_SBrkr', 'GarageCond_Fa', 'GarageCond_Gd', 'GarageCond_Po', 'GarageCond_TA', 'LotShape_IR2', 'LotShape_IR3', 'LotShape_Reg', 'Foundation_CBlock', 'Foundation_PConc', 'Foundation_Slab', 'Foundation_Stone', 'Foundation_Wood', 'SaleCondition_AdjLand', 'SaleCondition_Alloca', 'SaleCondition_Family', 'SaleCondition_Normal', 'SaleCondition_Partial', 'LandSlope_Mod', 'LandSlope_Sev', 'Functional_Maj2', 'Functional_Min1', 'Functional_Min2', 'Functional_Mod', 'Functional_Sev', 'Functional_Typ', 'ExterCond_Fa', 'ExterCond_Gd', 'ExterCond_Po', 'ExterCond_TA', 'BsmtExposure_Gd', 'BsmtExposure_Mn', 'BsmtExposure_No', 'Neighborhood_Blueste', 'Neighborhood_BrDale', 'Neighborhood_BrkSide', 'Neighborhood_ClearCr', 'Neighborhood_CollgCr', 'Neighborhood_Crawfor', 'Neighborhood_Edwards', 'Neighborhood_Gilbert', 'Neighborhood_IDOTRR', 'Neighborhood_MeadowV', 'Neighborhood_Mitchel', 'Neighborhood_NAmes', 'Neighborhood_NPkVill', 'Neighborhood_NWAmes', 'Neighborhood_NoRidge', 'Neighborhood_NridgHt', 'Neighborhood_OldTown', 'Neighborhood_SWISU', 'Neighborhood_Sawyer', 'Neighborhood_SawyerW', 'Neighborhood_Somerst', 'Neighborhood_StoneBr', 'Neighborhood_Timber', 'Neighborhood_Veenker', 'Utilities_NoSeWa', 'KitchenQual_Fa', 'KitchenQual_Gd', 'KitchenQual_TA', 'MSZoning_FV', 'MSZoning_RH', 'MSZoning_RL', 'MSZoning_RM', 'Condition1_Feedr', 'Condition1_Norm', 'Condition1_PosA', 'Condition1_PosN', 'Condition1_RRAe', 'Condition1_RRAn', 'Condition1_RRNe', 'Condition1_RRNn', 'HouseStyle_1.5Unf', 'HouseStyle_1Story', 'HouseStyle_2.5Fin', 'HouseStyle_2.5Unf', 'HouseStyle_2Story', 'HouseStyle_SFoyer', 'HouseStyle_SLvl', 'BsmtCond_Gd', 'BsmtCond_Po', 'BsmtCond_TA', 'RoofStyle_Gable', 'RoofStyle_Gambrel', 'RoofStyle_Hip', 'RoofStyle_Mansard', 'RoofStyle_Shed', 'LotConfig_CulDSac', 'LotConfig_FR2', 'LotConfig_FR3', 'LotConfig_Inside', 'PavedDrive_P', 'PavedDrive_Y', 'BsmtQual_Fa', 'BsmtQual_Gd', 'BsmtQual_TA']
    
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns =columns_names)
    
    #print('X data', X)
    # Ensure missingness in X is properly accounted for
    X = X.replace([np.inf, -np.inf], np.nan)
    
    # Extract rules from the RuleFit model
    rules = rulefit_model._get_rules()
    #print("These are the rules", rules)

    
    # Filter out the rules that are not conditions (i.e., keep only decision rules)
    decision_rules = rules[rules['type'] == 'rule']
    
    # Track rows relying on missing values based on original column names used in rules
    missing_reliance_rows = set()
    
    for index, rule in decision_rules.iterrows():
        # Parse each rule to identify the original column names used
        # Assuming rules are in a format like 'X1 > 5' or 'age < 30'
        conditions = rule['rule'].split(' and ')
        for condition in conditions:
            parts = condition.strip().split()
            # Check if the feature part of the condition is in the dataframe columns
            if len(parts) >= 1:  # Ensures there is at least a feature name present
                feature = parts[0]
                if feature in X.columns and X[feature].isnull().any():
                    # If the feature is present in X and has missing values
                    missing_reliance_rows.update(X.index[X[feature].isnull()])
    
    # Calculate the fraction of rows relying on missing values
    fraction_missing_reliance = len(missing_reliance_rows) / len(X)
    
    return fraction_missing_reliance