import pandas as pd

from common.utils.utils import searchAPI, switch_idx_data, \
                    interpolation,draw_graph,df_corr_hrci, logger
import seaborn as sns
from matplotlib import font_manager, rc, pyplot as plt
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def redifined_data():
    # 데이터 호출
    df = searchAPI("dgl_idx_expo_lst")
    logger.info('데이터 베이스에서 데이터를 불러오는데 성공했습니다.')
    # 코드 데이터 칼럼화
    df_total = switch_idx_data(df)
    df_total = df_total.drop(['bdi_cach_expo','kcci_cach_expo'],axis = 1)

    rgst_date = df_total['rgsr_dt']
    # df_total['bdi_cach_expo'] = pd.to_numeric(df_total['bdi_cach_expo'])
    df_total['ccfi_cach_expo'] = pd.to_numeric(df_total['ccfi_cach_expo'])
    df_total['hrci_cach_expo'] = pd.to_numeric(df_total['hrci_cach_expo'])
    df_total['scfi_cach_expo'] = pd.to_numeric(df_total['scfi_cach_expo'])
    # df_total['kcci_cach_expo'] = pd.to_numeric(df_total['kcci_cach_expo'])


    non_nan_indices = df_total[df_total['hrci_cach_expo'].notna()].index
    first_non_nan_index = non_nan_indices[0]
    last_non_nan_index = non_nan_indices[-1]

    sliced_df = df_total.loc[first_non_nan_index:last_non_nan_index]

    # 보간법 적용
    df_interpolated = interpolation(sliced_df)
    df_interpolated_filled = df_interpolated.fillna(method='bfill')
    # 가장 상관도가 높은 위치로 데이터 이동
    df_interpolated_filled['hrci_cach_expo_shifted'] = df_interpolated_filled['hrci_cach_expo'].shift(df_corr_hrci(df_interpolated_filled)[0])
    return df_interpolated_filled, rgst_date

if __name__=='__main__':
    df_interpolated = redifined_data()
    #그래프 그리기
    draw_graph(df_interpolated,'containner')
    #히트맵 그리기
    heatmap = sns.heatmap(df_interpolated.corr(), annot=True, cmap='coolwarm')
    heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation =45, ha='right')
    plt.show()

    #이동형 상관도 비교결과 가장 큰 상관도를 가지는 이동일 수 및  상관계수
    print(df_corr_hrci(df_interpolated)[0])