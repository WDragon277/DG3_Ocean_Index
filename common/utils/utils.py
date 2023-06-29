import pandas as pd
from elasticsearch import Elasticsearch
import seaborn as sns
from matplotlib import pyplot as plt
import logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def searchAPI(index_name):
  es = Elasticsearch('http://121.138.113.16:19202')
  index = index_name
  body = {
    'size': 10000,
    'query': {
      'match_all': {}
    }
  }
  result = es.search(index=index, body = body, size=10000)
  res = pd.DataFrame([hit['_source'] for hit in result['hits']['hits']])

  return res

#행렬 변환
def switch_idx_data(df):

    #데이터 정렬
    df = df.sort_values('data_cd')

    #인덱스 재설정
    df = df.reset_index().drop('index',axis=1)

    #운송지수별 데이터프레임 만들기
    df_bdi = pd.DataFrame(df[df['data_cd']=='bdi'])
    df_ccfi = pd.DataFrame(df[df['data_cd']=='ccfi'])
    df_scfi = pd.DataFrame(df[df['data_cd']=='scfi'])
    df_hrci = pd.DataFrame(df[df['data_cd']=='hrci'])
    df_kcci = pd.DataFrame(df[df['data_cd']=='kcci'])

    #칼럼명 변경
    df_bdi  = df_bdi .rename(columns = {'cach_expo' :'bdi_cach_expo'})
    df_ccfi = df_ccfi.rename(columns = {'cach_expo' :'ccfi_cach_expo'})
    df_scfi = df_scfi.rename(columns = {'cach_expo' :'scfi_cach_expo'})
    df_hrci = df_hrci.rename(columns = {'cach_expo' :'hrci_cach_expo'})
    df_kcci = df_kcci.rename(columns={'cach_expo': 'kcci_cach_expo'})

    #코드 칼럼 삭제
    df_bdi  = df_bdi .drop('data_cd',axis=1)
    df_ccfi = df_ccfi.drop('data_cd',axis=1)
    df_scfi = df_scfi.drop('data_cd',axis=1)
    df_hrci = df_hrci.drop('data_cd',axis=1)
    df_kcci = df_kcci.drop('data_cd', axis=1)

    #날짜인덱스 만들기
    max_date = df['rgsr_dt'].max()
    min_date = df['rgsr_dt'].min()
    dates = pd.date_range(min_date, max_date)
    df_dates = pd.DataFrame(dates)

    #날짜 칼럼명 일치를 위한 변경
    df_dates = df_dates.rename(columns = {0:'rgsr_dt'})

    #날짜 부분의 데이터 타입 변경
    df_dates['rgsr_dt'] = df['rgsr_dt'].astype(str).str.replace('-', '')

    #일자순으로 통합하기
    df_total = pd.merge(df_dates,df_bdi, how = 'outer',on='rgsr_dt')
    df_total = pd.merge(df_total,df_ccfi, how = 'outer',on='rgsr_dt')
    df_total = pd.merge(df_total,df_hrci, how = 'outer',on='rgsr_dt')
    df_total = pd.merge(df_total,df_scfi, how = 'outer',on='rgsr_dt')
    df_total = pd.merge(df_total, df_kcci, how='outer', on='rgsr_dt')

    #날짜 순으로 정렬 및 인덱스 설정
    df_total = df_total.sort_values(by = 'rgsr_dt')
    df_total = df_total.reindex()

    #중복 제거
    df_total = df_total.drop_duplicates(keep='first', inplace=False, ignore_index= True)

    return df_total

def interpolation(df):
    result = df.interpolate()
    return result

def df_corr_hrci(df):
    indx_corr = []
    rang = range(30)
    for i in rang:
        j = -i
        df['hrci_cach_expo_shifted'] = df['hrci_cach_expo'].shift(j)
        indx_corr.append([j, df[['hrci_cach_expo_shifted', 'scfi_cach_expo', 'ccfi_cach_expo']].corr() \
                                 ['hrci_cach_expo_shifted'][1:3].mean()])
        sorted_indx_corr = sorted(indx_corr, key=lambda x: x[1], reverse=True)
        result = sorted_indx_corr[0]
        logger.info('적절한 예측 기간 : ', result)

    return result

def draw_graph(df_total,index):
    #하나의 그래프(아티스트)에 모두 그리기
    plt.figure(figsize=(10,10))
    if  index == 'bulk':
        plt.plot(df_total['rgsr_dt']
                 ,df_total['bdi_cach_expo'],'ro',linestyle='solid')
        plt.title('벌크 해상운임지수 그래프')
        plt.legend(['BDI'
                    ])
    if  index == 'containner':
        plt.plot(df_total['rgsr_dt']
                 ,df_total['ccfi_cach_expo'],color='forestgreen', marker='^', markersize=6)
        plt.plot(df_total['rgsr_dt']
                 ,df_total['hrci_cach_expo'],'bo')
        plt.plot(df_total['rgsr_dt']
                 ,df_total['scfi_cach_expo'],'mo')
        plt.title('컨테이너 해상운임지수 3종 그래프')
        plt.legend(['CCFI'
                   , 'HRCI'
                   , 'SCFI'])
    if  index == 'containner2':
        plt.plot(df_total['rgsr_dt']
                 ,df_total['ccfi_cach_expo'],color='forestgreen', marker='^', markersize=6)
        plt.plot(df_total['rgsr_dt']
                 ,df_total['scfi_cach_expo'],'mo')
        plt.plot(df_total['rgsr_dt']
                 , df_total['hrci_cach_expo_shifted'], 'yo')
        plt.title('컨테이너 해상운임지수 3종 그래프_(이동된)')
        plt.legend(['CCFI'
                   , 'SCFI'
                   , 'HRCI_Shifted30'])
    if index == 'total':
        plt.plot(df_total['rgsr_dt']
                 , df_total['bdi_cach_expo'], 'ro', linestyle='solid')
        plt.plot(df_total['rgsr_dt']
                 , df_total['ccfi_cach_expo'], color='forestgreen', marker='^', markersize=6, linestyle='solid')
        plt.plot(df_total['rgsr_dt']
                 , df_total['hrci_cach_expo'], 'bo', linestyle='solid')
        plt.plot(df_total['rgsr_dt']
                 , df_total['scfi_cach_expo'], 'mo', linestyle='solid')
        plt.plot(df_total['rgsr_dt']
                 , df_total['hrci_cach_expo_shifted'], 'yo')
        plt.title('해상운임지수 4종 그래프')
        plt.legend(['BDI'
                       , 'CCFI'
                       , 'HRCI'
                       , 'SCFI'
                       , 'HRCI_Shifted30'])
    plt.xlabel('Time')
    plt.xticks([0,110,220,330,440,536])
    plt.ylabel('Cach_expo')
    plt.show()

def draw_heatmap(df):
    heatmap = sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    heatmap.set_xticklabels(heatmap.get_xticklabels(),rotation =45, ha='right')
    plt.show()

## 데이터 불러오기
# from common.utils.utils import searchAPI, switch_idx_data, \
#                     interpolation, df_corr
from elasticsearch import Elasticsearch as es
from matplotlib import font_manager, rc
font_path = "C:/Windows/Fonts/NGULIM.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

def raw_data():

    df = searchAPI("dgl_idx_expo_lst")
    # 코드 데이터 칼럼화
    df_total = switch_idx_data(df)
    # 보간법 적용
    df_interpolated = interpolation(df_total)

    return df_interpolated

def defined_data():

    df = searchAPI("dgl_idx_expo_lst")
    # 코드 데이터 칼럼화
    df_total = switch_idx_data(df)
    # 보간법 적용
    df_interpolated = interpolation(df_total)
    # 가장 상관도가 높은 위치로 데이터 이동
    df_interpolated['hrci_cach_expo_shifted'] = df_interpolated['hrci_cach_expo'].shift(df_corr(df_interpolated)[0])

    return df_interpolated

def pred_data():
    df = searchAPI("dgl_idx_expo_lst")
    # 코드 데이터 칼럼화
    df_total = switch_idx_data(df)
    # 보간법 적용
    df_interpolated = interpolation(df_total)
    pred_data_tmp = df_interpolated[df_corr(df_interpolated)[0]:]
    pred_data = pred_data_tmp[['rgsr_dt','scfi_cach_expo','ccfi_cach_expo']]
    return pred_data

## 데이터 입력시 활용
def doc_type_setting(index):
    if index == 'dgl_idx_expo_pred':
        document = {
            "mappings": {
                "dynamic": False,
                "properties": {
                    "data_cd": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "rgsr_dt": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    },
                    "cach_expo": {
                        "type": "integer"
                    }
                }
            }
        }
    return document

def input_database(df,index_name,doc_type):
    csv_file = 'tmp_input_csv_file'
    df.to_csv(csv_file, index=False)
    with open(csv_file, 'r') as f:
        es.indices.create(index = index_name, ignore=400)

# if __name__=='__main__':
#     a = defined_data()
#     print(a)

