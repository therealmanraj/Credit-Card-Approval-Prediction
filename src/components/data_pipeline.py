import pickle
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
from app_logger.logger import logging
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import FunctionTransformer, Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline

from utils.common import CustomException
from utils.common import save_object, load_object

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataTransformationConfig:
        transformation_obj_file_path=os.path.join('artifacts','preprocessor', 'preprocessor.pkl')
        transformation_obj_dir_path=os.path.join('artifacts','preprocessor')

class DataPipeline(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        self.feature_to_drop = ['ID','Has a mobile phone','Children count','Job title','Account age']
        self.feat_with_outliers = ['Family member count','Income', 'Employment length']
        self.feat_with_days = ['Employment length', 'Age']
        self.one_hot_enc_ft = ['Gender', 'Marital status', 'Dwelling', 'Employment status', 'Has a car', 'Has a property', 'Has a work phone', 'Has a phone', 'Has an email']
        self.feat_with_skewness = ['Income','Age']
        self.feat_with_num_enc = ['Has a work phone','Has a phone','Has an email']
        self.ordinal_enc_ft = ['Education level']
        self.min_max_scaler_ft = ['Age', 'Income', 'Employment length']
        
    def outliers(self, df):
        if (set(self.feat_with_outliers).issubset(df.columns)):
            # 25% quantile
            Q1 = df[self.feat_with_outliers].quantile(0.25)
            # 75% quantile
            Q3 = df[self.feat_with_outliers].quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df[self.feat_with_outliers]<(Q1 - 3*IQR)) | (df[self.feat_with_outliers]>(Q3 + 3*IQR))).any(axis=1)]
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
    def drop_features(self, df):
        if (set(self.feature_to_drop).issubset(df.columns)):
            df.drop(self.feature_to_drop, axis=1, inplace = True)
            return df
        else:
            print('One or more features are not in dataframe')
            return df
        
    def time_conversion(self, X, y=None):
        if (set(self.feat_with_days).issubset(X.columns)):
            # convert days to absolute value
            X[self.feat_with_days] = np.abs(X[self.feat_with_days])
            return X
        else:
            print('One or more features are not in dataframe')
            return X
        
    def retiree_handling(self, df):
        if 'Employment length' in df.columns:
            # select rows with employment length is 365243 which corresponds to retirees
            df_ret_idx = df['Employment length'][df['Employment length']==365243].index
            # change 365243 to 0
            df.loc[df_ret_idx,'Employment length'] = 0
            return df
        else:
            print('Employment length feature is not in dataframe')
            return df
        
    def skewness_handling(self, df):
        if (set(self.feat_with_skewness).issubset(df.columns)):
            # Handle skewness with cubic root transformation
            df[self.feat_with_skewness] = np.cbrt(df[self.feat_with_skewness])
            return df
        else:
            print('One or more features are not in dataframe')
            return df
        
    def binning(self,df):
        if (set(self.feat_with_num_enc).issubset(df.columns)):
            # Change 0 to N and 1 to Y for all the features in feat_with_num_enc
            for ft in self.feat_with_num_enc:
                df[ft] = df[ft].map({1:'Y',0:'N'})
            return df
        else:
            print("One or more features are not in the dataframe")
            return df
        
    def one_hot_encoding(self, df):
        if (set(self.one_hot_enc_ft).issubset(df.columns)):
            
            def one_hot_encoding(df, one_hot_enc_ft):
                one_hot_enc = OneHotEncoder()
                one_hot_enc.fit(df[one_hot_enc_ft])
                feat_names_one_hot_enc = one_hot_enc.get_feature_names_out(one_hot_enc_ft)
                df = pd.DataFrame(one_hot_enc.transform(df[self.one_hot_enc_ft]).toarray(), columns=feat_names_one_hot_enc, index=df.index)
                return df
            
            def concat_with_rest(df, one_hot_enc_df, one_hot_enc_ft):
                rest_of_feats = [ft for ft in df.columns if ft not in one_hot_enc_ft]
                df_concat = pd.concat([one_hot_enc_df,df[rest_of_feats]],axis=1)
                return df_concat
            
            one_hot_enc_df = one_hot_encoding(df, self.one_hot_enc_ft)
            full_one_hot_enc_df = concat_with_rest(df, one_hot_enc_df, self.one_hot_enc_ft)
            
            return full_one_hot_enc_df
        else:
            print("One or more features are not in the dataframe")
            return df
        
    def ordinal_encoding(self, df):
        if 'Education level' in df.columns:
            oridinal_enc = OrdinalEncoder()
            df[self.ordinal_enc_ft] = oridinal_enc.fit_transform(df[self.ordinal_enc_ft])
            return df
        else:
            print('Education level is not in the dataframe')
            return df
        
    def min_max_scaling(self, df):
        if (set(self.min_max_scaler_ft).issubset(df.columns)):
            min_max_enc = MinMaxScaler()
            df[self.min_max_scaler_ft] = min_max_enc.fit_transform(df[self.min_max_scaler_ft])
            return df
        else:
            print('One or more features are not in the dataframe')
            return df
        
    def change_to_num(self, df):
        if 'Is high risk' in df.columns:
            df['Is high risk'] = pd.to_numeric(df['Is high risk'])
            return df
        else:
            print('Is high risk is not in the dataframe')
            return df
        
    def oversampling(self, df):
        if 'Is high risk' in df.columns:
            oversample = SMOTE(sampling_strategy='minority')
            X_bal, y_bal = oversample.fit_resample(df.loc[:,df.columns != 'Is high risk'], df['Is high risk'])
            df_bal = pd.concat([pd.DataFrame(X_bal), pd.DataFrame(y_bal)], axis=1)
            return df_bal
        else:
            print('Is high risk is not in the dataframe')
            return df
    
    def initiate_pipeline(self):
        """
        Build and return a sklearn Pipeline that applies each of your
        DataPipeline methods in sequence.
        """
        try:
            steps = [
                ("outliers",           FunctionTransformer(self.outliers,      validate=False)),
                ("drop_features",      FunctionTransformer(self.drop_features, validate=False)),
                ("time_conversion",    FunctionTransformer(self.time_conversion, validate=False)),
                ("retiree_handling",   FunctionTransformer(self.retiree_handling, validate=False)),
                ("skewness_handling",  FunctionTransformer(self.skewness_handling, validate=False)),
                ("binning",            FunctionTransformer(self.binning,         validate=False)),
                ("one_hot_encoding",   FunctionTransformer(self.one_hot_encoding, validate=False)),
                ("ordinal_encoding",   FunctionTransformer(self.ordinal_encoding, validate=False)),
                ("min_max_scaling",    FunctionTransformer(self.min_max_scaling, validate=False)),
                ("change_to_num",      FunctionTransformer(self.change_to_num,   validate=False)),
            ]
            pipeline = Pipeline(steps=steps)
            logging.info("Pipeline object created.")
            return pipeline

        except Exception as e:
            logging.error(f"Error building pipeline: {e}")
            raise CustomException(e, sys)
    
    def data_transformation(self, train_df, test_df):
        try:
            pipeline_obj = self.initiate_pipeline()
            logging.info("Obtained pipeline object.")
            
            df = pd.concat([train_df, test_df])
            model = pipeline_obj.fit(df)
            
            input_feature_train = pipeline_obj.transform(train_df)
            input_feature_train = self.oversampling(input_feature_train)
            
            input_feature_test  = pipeline_obj.transform(test_df)
            input_feature_test = self.oversampling(input_feature_test)
            
            
            os.makedirs(self.data_transformation_config.transformation_obj_dir_path, exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.transformation_obj_file_path,
                obj=model
            )
            logging.info(f"Saved preprocessing object to {self.data_transformation_config.transformation_obj_file_path}")
            
            return input_feature_train, input_feature_test

        except Exception as e:
            raise CustomException(e, sys)