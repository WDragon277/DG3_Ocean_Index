import pandas as pd
from sklearn.metrics import mean_absolute_error
from common.utils import df_date
import numpy as np
import sys
from sklearn.ensemble import GradientBoostingRegressor

sys.path.append('C:\\Users\\0614_\\PycharmProjects\\DG3_Ocean_Index\\model\\hrdi_p')
sys.path.append('c:\\users\\0614_\\anaconda3\\envs\\dg3_ocean_index\\lib\\site-packages')
from repository.repository import redifined_data

def pred_hrci_model():
    data, rgst_date = redifined_data()
    non_nan_indices = data[data['hrci_cach_expo_shifted'].notna()].index
    # first_non_nan_index = non_nan_indices[0]
    last_non_nan_index = non_nan_indices[-1]

    data['hrci_cach_expo_pred'] = data['hrci_cach_expo']
    # data['hrci_cach_expo_pred'][last_non_nan_index-1:] = np.nan

    X = np.array(data[['ccfi_cach_expo','scfi_cach_expo']])
    y = np.array(data['hrci_cach_expo_shifted'].dropna())

    X_train = X[:last_non_nan_index-1]
    X_pred = X[last_non_nan_index-1:]
    y_train = y[:last_non_nan_index]

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)
    data['hrci_cach_expo_pred'][last_non_nan_index-1:] = y_pred


    # 실제 인덱스 값
    result_real = pd.DataFrame({'data_cd':'hrci', 'rgsr_dt':data['rgst_dt'][:last_non_nan_index-1], 'cach_expo': data['hrci_cach_expo'],'pred': 'N'})
    # 예측 인덱스 값
    result_pred = pd.DataFrame({'data_cd':'hrci', 'rgsr_dt':rgst_date[last_non_nan_index-1:], 'cach_expo': y_pred, 'pred': 'Y'})
    # 전체 인덱스 값
    result_total = result_real.merge(result_pred)

    return result_total


# R2_score = model.score(X_pred, y_pred)
# mae_score = mean_absolute_error(X_pred, y_pred)
# print("MAE score:", mae_score)
# print("R^2 score:", R2_score)
if __name__=='__main__':
    result = pred_hrci_model()
    print(result)