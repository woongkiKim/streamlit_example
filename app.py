import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle
import time
import joblib
from modules.charts import MakeChart

## 1. 페이지 설정 (set_page_config는 맨 위에 위치)
st.set_page_config(
    page_title="LS빅데이터스쿨 2기 Streamlit 예제",
    page_icon="🏭",
)


#### 📌 1. 제목 설정하기 ####
st.title("1. 제목 설정하기")
st.header("Header")
st.subheader("subheader")
######################
st.divider()


#### 📌 2. 위젯 만들기 ####
st.title("2. 위젯 만들기")

## 버튼 만들기
if st.button("버튼을 클릭하세요"):
    st.write("✅ 버튼이 클릭되었습니다.")

## 체크박스 만들기
checkbox_btn = st.checkbox('체크박스 버튼을 클릭하세요')
if checkbox_btn:
    st.write('🔥🔥🔥 체크박스 버튼이 클릭되었습니다.')

## 체크박스 디폴트 
checkbox_btn2 = st.checkbox('디폴트 체크박스', value=True)
	
if checkbox_btn2:
    st.write('저는 이미 클릭이 되어 있습니다. 🙃')

## 라디오 버튼 만들기
selected_item = st.radio("라디오 버튼", ("A", "B", "C"))
	
if selected_item == "A":
    st.write("✅ A를 클릭했어요!")
elif selected_item == "B":
    st.write("✅✅ B를 클릭했어요!")
elif selected_item == "C":
    st.write("✅✅✅ C를 클릭했어요!")

## 옵션 선택
option = st.selectbox('원하는 옵션을 선택하세요.',
                       ('1번 옵션', '2번 옵션', '3번 옵션'))
	
st.write('✅ 당신의 선택 :', option)

## 다중 옵션 선택
options = st.multiselect('원하는 옵션을 선택하세요.',
                            ('1번 옵션', '2번 옵션', '3번 옵션'))

st.write('🔥🔥당신의 선택 :', options)

## 슬라이더 만들기
number = st.slider('숫자를 선택하세요.', min_value=0, max_value=100, value=50)
st.write('✅ 당신이 선택한 숫자는', number, '입니다.')
######################
st.divider()

#### 📌 3. 이미지 다루기 ####
## 클릭을 해야지만 이미지가 나타남
if st.checkbox('이미지 보기'):
    st.title("3. 이미지 다루기")
    image_url = "https://media1.tenor.com/m/A64OVBBLwU4AAAAd/%EB%82%98%EB%A3%A8%ED%86%A0%EC%82%AC%EC%8A%A4%EC%BC%80%EC%8B%B8%EC%9B%80%EC%88%98%EC%A4%80-anime.gif"

    # HTML을 사용하여 이미지 크기 조정
    st.markdown(f'<img src="{image_url}" width="700" height="350">', unsafe_allow_html=True)
    ##########################
    st.divider()

#### 📌 4. 화면 구분 다루기 ####
st.title("4. 화면 구분 다루기")
# 탭으로 화면 구분
tab1, tab2 = st.tabs(["Tab 1", "Tab2"])
with tab1:
    tab1.write("tab1에 대한 내용")
    st.metric(label="Temp1", value="273 K", delta="1.2 K")
with tab2:
    tab2.write("tab2에 대한 내용")
    st.metric(label="Temp2", value="273 K", delta="-1.2 K")

st.divider()

# columns로 화면 구분
col1, col2 = st.columns(2)
with col1:
    st.write("첫번째 열")
    
with col2:
    st.write("두번째 열")
    

##########################
st.divider()

#### 📌 5. 모델 사용하기 ####
st.title("5. 모델 사용하기")
st.write("""
#### Iris 데이터셋
- 3개의 클래스가 있습니다.
- 4개의 feature가 있습니다.
- 150개의 데이터가 있습니다.         
- 모델: RandomForestClassifier
#### 모델을 학습시키고 싶으시면 아래 버튼을 클릭하세요.

""")
# 데이터 불러오기
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)

## 버튼을 클릭하면, 모델을 학습하고 pickle로 저장
if st.button("모델 학습"):
    ## 모델이 학습될때까지 스핀
    with st.spinner("모델 학습 중..."):
        time.sleep(3)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        ## ⭐️⭐️⭐️⭐️ 모델 로컬에 저장 ⭐️⭐️⭐️⭐️
        ## joblib.dump(model, './iris_model.pkl')
        ## 만일 저장된 모델을 불러오고 싶다면
        ## model = joblib.load('./iris_model.pkl')
        
        st.write(f"✅ 모델이 학습되었습니다.")
##########################

# 붓꽃 데이터셋을 사용한 예측
st.header("🌺 붓꽃 데이터셋을 사용한 예측")

model = joblib.load('./iris_model.pkl')
# 입력 받기
col1, col2, col3, col4 = st.columns(4)
with col1:
    sepal_length = st.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
with col2:
    sepal_width = st.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
with col3:
    petal_length = st.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
with col4:    
    petal_width = st.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))

# 입력 데이터를 데이터프레임으로 변환
input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=iris.feature_names)

# 예측
prediction = model.predict(input_data)

species_mapping = {
    0: 'Setosa',
    1: 'Versicolor',
    2: 'Virginica'
}
predicted_species = species_mapping[prediction[0]]
# 예측 확률 
prediction_proba = model.predict_proba(input_data)

st.write(f"✅ 예측된 붓꽃 종류 => {iris.target_names[prediction][0]}")
## 예측 결과가 OO이면 Alert창
if predicted_species == 'Setosa':
    st.warning(f"🔴 경고: Setosa라는 예측이 나오면 {prediction_proba[0][0]}확률로 경고를 합니다.")
elif predicted_species == 'Versicolor':
    st.success(f"🟢 안전: Versicolor라는 예측이 나오면 {prediction_proba[0][1]}확률로 안전합니다.")
else:
    st.info(f"🟡 정보: Virginica라는 예측이 나오면 {prediction_proba[0][2]}확률로 정보를 제공합니다.")
st.write("🤖 확률 TABLE")
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

st.divider()

if st.checkbox('다이캐스팅 데이터로 해볼까요?'):

    FILE_PATH = 'data/casting.csv'

    ## 2. 대시보드 제목
    st.title('Streamlit 핸즈온')

    ## 3. 데이터 불러오기
    def load_data(FILE_PATH):
        data = pd.read_csv(FILE_PATH, encoding='cp949', index_col=0)
        ## ✅ 전처리 로직 
        data['registration_time'] = pd.to_datetime(data['registration_time'])
        ## 3-1. 필요한 컬럼만 추출
        df = data[['registration_time','mold_code','count',
            'molten_temp', 'facility_operation_cycleTime',
            'production_cycletime', 'low_section_speed', 'high_section_speed',
            'molten_volume', 'cast_pressure', 'biscuit_thickness',
            'upper_mold_temp1', 'upper_mold_temp2', 'upper_mold_temp3',
            'lower_mold_temp1', 'lower_mold_temp2', 'lower_mold_temp3',
            'sleeve_temperature', 'physical_strength', 'Coolant_temperature',
            'passorfail']]
        ## 3-2. mold_code를 문자열로 변환
        df['mold_code'] = df['mold_code'].astype(str)
        ## 3-3. 날짜, 시간, 요일 컬럼 생성
        df['date'] = df['registration_time'].dt.date
        df['hour'] = df['registration_time'].dt.hour
        df['weekday'] = df['registration_time'].dt.weekday ## 0:월요일, 6:일요일
        df['date_time'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00')
        df['weekday'] = df['weekday'].map({0:'월', 1:'화', 2:'수', 3:'목', 4:'금', 5:'토', 6:'일'})
        # 3-4. 평균 생산시간 계산
        df['average_cycle_time'] = (df['facility_operation_cycleTime'] + df['production_cycletime']) / 2
        ## 3-5. 정상품, 불량품 생성
        df['pass'] = df['passorfail'].apply(lambda x: 1 if x == 0 else 0)
        df['fail'] = df['passorfail'].apply(lambda x: 1 if x == 1 else 0)

        ## 3-6. 요일별, 시간별, mold_code별 평균 생산 시간과 생산량 계산
        grouped_data = df.groupby(['date_time','mold_code','weekday','hour'])['average_cycle_time'].agg(['mean','median','count']).reset_index()
        ## 3-7. 요일별, 시간별, mold_code별 양품/불량품 수 계산
        grouped_data2 = df.groupby(['date_time','mold_code','weekday','hour'])['pass'].sum().reset_index(name='pass_count')
        grouped_data3 = df.groupby(['date_time','mold_code','weekday','hour'])['fail'].sum().reset_index(name='error_count')
        ## 3-8. 데이터 병합
        merge_grouped_df = pd.merge(grouped_data, grouped_data2,
                                    on=['date_time','mold_code','weekday','hour'], how='left') 

        merge_grouped_df = pd.merge(merge_grouped_df, grouped_data3,
                                        on=['date_time','mold_code','weekday','hour'], how='left')

        ## 3-9. 생산량, 생산시간, 양품/불량품 비율 계산
        merge_grouped_df['mean'] = merge_grouped_df['mean'].round(1)
        merge_grouped_df['median'] = merge_grouped_df['median'].round(1)
        merge_grouped_df['error_ratio'] = (merge_grouped_df['error_count'] / merge_grouped_df['count']).round(2)
        merge_grouped_df['pass_ratio'] = 1 - merge_grouped_df['error_ratio'].round(2)
        merge_grouped_df['date'] = merge_grouped_df['date_time'].dt.date

        return merge_grouped_df

    data = load_data(FILE_PATH)


    ## 4. 체크박스를 통해 데이터를 확인할지 말지 결정
    if st.checkbox('raw 데이터 확인'):
        st.subheader('raw 데이터') 
        st.divider()
        st.write(data)

    ## 5. 히스토그램 시각화
    st.subheader('시간별 생산량 히스토그램')

    # 6. Plotly를 사용한 히스토그램
    fig1 = px.histogram(data, 
                    x=data['hour'], 
                    nbins=24, 
                    color='mold_code',
                    title='시간별 생산량 히스토그램'
                    )

    fig1.update_xaxes(title_text='24시간')
    fig1.update_yaxes(title_text='생산량')

    st.plotly_chart(fig1)

    ## 7. 특정 mold_code의 생산량 시각화
    st.subheader('특정 mold_code의 생산량 시각화')
    # mold_code 선택 옵션 추가
    selected_mold_code = st.selectbox('mold_code 선택', data['mold_code'].unique())

    # 선택한 mold_code에 해당하는 데이터 필터링
    filtered_data = data[data['mold_code'] == selected_mold_code]

    fig2 = px.histogram(filtered_data, 
                    x='hour', 
                    nbins=24, 
                    title=f'시간별 생산량 히스토그램 (mold_code: {selected_mold_code})'
                    )

    fig2.update_xaxes(title_text='24시간')
    fig2.update_yaxes(title_text='생산량')

    st.plotly_chart(fig2)


    # Streamlit에서 날짜 범위 선택 위젯을 가로로 배치
    col1, col2, col3 = st.columns(3)

    with col1:
        start_date = st.date_input('시작 날짜', value=data['date_time'].min())

    with col2:
        end_date = st.date_input('종료 날짜', value=data['date_time'].max())

    with col3:
        # mold_code 선택 위젯
        unique_mold_codes = data['mold_code'].unique()
        selected_mold_code = st.multiselect('Mold Code 선택', unique_mold_codes)


    # 선택한 날짜 범위에 맞게 데이터 필터링
    filtered_data = data[(data['date_time'] >= pd.to_datetime(start_date)) &
                                    (data['date_time'] <= pd.to_datetime(end_date)) &
                                    (data['mold_code'].isin(selected_mold_code))]

    st.write(filtered_data)

    ## 정상품 비율
    pass_ratio = filtered_data['pass_count'].sum() / filtered_data['count'].sum()
    ## 불량품 비율
    fail_ratio = filtered_data['error_count'].sum() / filtered_data['count'].sum()

    ## 모듈폴더에서 도넛차트 함수 가져오기
    make_chart = MakeChart()
    # 필터링된 데이터로 도넛 차트 생성
    donut_chart_pass = make_chart.make_donut(pass_ratio.round(2) * 100, '정상품', 'green')
    donut_chart_fail = make_chart.make_donut(fail_ratio.round(2) * 100, '불량품', 'red')

    with col1:
        st.write('정상품 비율')
        st.altair_chart(donut_chart_pass, use_container_width=True)

    with col2:
        st.write('불량품 비율') 
        st.altair_chart(donut_chart_fail, use_container_width=True)

    # Streamlit을 사용해 대시보드에 차트 표시
    tab1, tab2, tab3 = st.tabs(['생산량', '불량률', '평균 생산 시간'])


    with tab1:
        # 날짜별, 요일별, 시간별 생산량 시각화
        fig1 = px.bar(filtered_data, x='date_time', y='count',
            color='weekday',
            title='날짜별, 요일별, 시간별 생산량',
            labels={'count': '생산량', 'hour': '시간', 'weekday': '요일'})

        # x축 범위 설정
        fig1.update_xaxes(range=[str(start_date), str(end_date)])
        st.plotly_chart(fig1)

    with tab2:
            # 날짜별, 요일별, 시간별 불량률 시각화
            fig2 = px.bar(filtered_data, x='date_time', y='error_ratio',
                    color='weekday', title='날짜별, 요일별, 시간별 불량률',
                    labels={'error_ratio': '불량률', 'hour': '시간', 'weekday': '요일'})
            
            ## 0.8 부분에 빨간색 점선 추가
            fig2.add_hline(y=0.8, line_dash='dot', line_color='red', annotation_text='불량률 80%', annotation_position='top right')


            # x축 범위 설정
            fig2.update_xaxes(range=[str(start_date), str(end_date)])
            st.plotly_chart(fig2)

    with tab3:

            # 날짜별, 요일별, 시간별 평균 생산 시간 시각화
            fig3 = px.bar(filtered_data, x='date_time', y='mean',
                    color='weekday',
                    title='날짜별, 요일별, 시간별 평균 생산 시간',
                    labels={'mean': '평균 생산 시간', 'hour': '시간', 'weekday': '요일'},
                    color_continuous_scale='Plasma'
                    )

            # x축 범위 설정
            fig3.update_xaxes(range=[str(start_date), str(end_date)])
            st.plotly_chart(fig3)

## 6. 페이지 이동 
## pages 폴더를 만들고, 각 페이지별로 파일을 만들어서 페이지 이동을 구현할 수 있습니다.
## 예를 들어, pages 폴더 안에 PredictPage.py 파일을 만들고, 아래와 같이 코드를 작성합니다.
st.markdown('''
- [ ] [예측파트](/PredictPage)
''')