from matplotlib import font_manager, rc, pyplot as plt
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np
import sys

sys.path.append('C:\\Users\\0614_\\PycharmProjects\\DG3_Ocean_Index\\model\\hrdi_p')
sys.path.append('c:\\users\\0614_\\anaconda3\\envs\\dg3_ocean_index\\lib\\site-packages')
from repository.repository import redifined_data

data = redifined_data()
# hrdi_data = data[['rgsr_dt','hrci_cach_expo','hrci_cach_expo_shifted']]
# ccfi_data = data['ccfi_cach_expo']
# scfi_data = data['scfi_cach_expo']

X = np.array(data[['ccfi_cach_expo','scfi_cach_expo']])
y = np.array(data['hrci_cach_expo'])

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
R2_score = model.score(X_test, y_pred)
mae_score = mean_absolute_error(y_test, y_pred)
print("MAE score:", mae_score)
print("R^2 score:", R2_score)


def pred_hrci_model():
    data, rgst_date = redifined_data()
    non_nan_indices = data[data['hrci_cach_expo_shifted'].notna()].index
    # first_non_nan_index = non_nan_indices[0]
    last_non_nan_index = non_nan_indices[-1]

    data['hrci_cach_expo_pred'] = data['hrci_cach_expo']
    data['hrci_cach_expo_pred'][last_non_nan_index-1:] = np.nan

    X = np.array(data[['ccfi_cach_expo','scfi_cach_expo']])
    y = np.array(data['hrci_cach_expo_shifted'].dropna())

    X_train = X[:last_non_nan_index-1]
    X_pred = X[last_non_nan_index-1:]
    y_train = y[:last_non_nan_index]

    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_pred)
    data['hrci_cach_expo_pred'][last_non_nan_index-1:] = y_pred
    # tmp = data['hrci_cach_expo_pred']
    result = pd.DataFrame({'data_cd':'hrci', 'rgsr_dt':rgst_date, 'cach_expo': data['hrci_cach_expo_pred']})
    return result