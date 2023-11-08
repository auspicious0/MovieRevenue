# 영화 수익 예측 (bagging, RandomForest)

본 문서는 Kaggle의 competition 데이터셋을 활용하여 데이터 분석 및 bagging, RandomForest를 통한 분류 모델 구축을 목표로 합니다. 이 데이터셋은 boxoffice 관련 정보를 담고 있습니다. 데이터를 적절하게 전처리한 후, 모델링과 예측을 수행합니다.

## 목차
[1. 패키지 설치 및 그래프 설정](#1-패키지-설치-및-그래프-설정)

[2. 데이터 수집](#2-데이터-수집)

[3. 데이터 전처리](#3-데이터-전처리)

      [3-1. 결측값 처리](#3-1-결측값-처리)
   
      [3-2. 데이터 변수의 형 변환](#3-2-데이터-변수의-형-변환)
   
      [3-3. 중간값 처리 및 이상값 처리](#3-3-중간값-처리-및-이상값-처리)
   
[4. bagging 및 RandomForest 분석](#4-bagging-및-randomforest-분석)

      [4-1. 회귀 분석](#4-1-회귀-분석)
   
      [4-2. 분류 분석](#4-2-분류-분석)
   
[5. 문의](#5-문의)





## 1. 패키지 설치 및 그래프 설정

프로젝트를 시작하기 전, 필요한 R 패키지를 설치하고 그래프 설정을 합니다.
 ```
#패키지 부착, 출력 그래프의 크기를 설정
install.packages(c("tidyverse","data.table","caret"))
library(tidyverse)
library(data.table)
library(repr)

options(repr.plot.width=7,repr.plot.height=7)
```

## 2. 데이터 수집

유방암 데이터는 Kaggle에서 제공되며 다음 링크에서 얻을 수 있습니다:

[Kaggle Dataset](https://www.kaggle.com/competitions/tmdb-box-office-prediction)

데이터를 다운로드하고 분석에 활용합니다.

```
#https://www.kaggle.com/competitions/tmdb-box-office-prediction
#https://drive.google.com/file/d/1pdeaqGlwkZi1fmvgG6OJEyWZQvZdlvUg/view?usp=sharing
#https://drive.google.com/file/d/13QDXi_v_9di42Lw0vUAR3okJXXkacSG7/view?usp=sharing
system("gdown --id 1pdeaqGlwkZi1fmvgG6OJEyWZQvZdlvUg")
system("gdown --id 13QDXi_v_9di42Lw0vUAR3okJXXkacSG7")
system("ls",TRUE)
```

## 3. 데이터 전처리

불러온 데이터를 적절히 결합한 후 결측값 처리, 이상값 처리 등을 수행합니다. 데이터의 구조를 확인하고 필요한 변수를 팩터(factor)로 변환합니다. 

```
mr <- bind_rows(fread("movie_revenue_test.csv",encoding = "UTF-8") %>% as_tibble(),fread("movie_revenue_train.csv",encoding = "UTF-8")%>%as_tibble())#move_revenue
mr %>% show()
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/d86df3df-0a75-4dbf-ae39-4ce49f9ef67c)


### 3-1. 결측값 처리

몇가지 문제가 발생하였습니다.

결측값이 너무 많습니다. 따라서 결측값 처리에 관한 합리적인 방법이 필요합니다. 결측값을 모두 omit해버리면 제대로 된 모델링이 되지 않을 수 있습니다.

결측값인데 결측값으로 인식되지 않는 값이 많습니다. "", "0" '[]' 등의 값이 결측 데이터로 인식되지 않습니다.

따라서 추가적인 조치를 진행하겠습니다.

```
# "0"을 na.strings에서 제외하고 데이터 읽기
mr_test <- fread("movie_revenue_test.csv", encoding = "UTF-8", na.strings = c("", "#N/A", "[]")) %>% as_tibble()
mr_train <- fread("movie_revenue_train.csv", encoding = "UTF-8", na.strings = c("", "#N/A", "[]")) %>% as_tibble()

# "0" 값을 NA로 대체
mr_test[mr_test == 0] <- NA
mr_train[mr_train == 0] <- NA

mr <- bind_rows(mr_test, mr_train)
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/8e6f92d9-bb8c-41f7-8604-53166fef5aa4)


### 3-2. 데이터 변수의 형 변환 

id, 영화에 관한 평가(긴 문장 평가), 영화 제목 등 종속변수에 영향을 주지 않을 것으로 예상되는 변수를 삭제합니다.

종속변수는 revenue로 하고 나머지를 설명 변수로 처리해 수익을 예측해 보겠습니다.

chr형 변수 genres, original_language, status는 factor형으로 변환하겠습니다.

이중 정수형 변수 budget, popularity, revenue, runtime은 결측값 처리 후 이상값을 처리하겠습니다.
```
mr <- select(mr,budget,genres,original_language,popularity,runtime,revenue,status)%>%
  mutate_at(c("genres","original_language","status"),factor)

# 'genres' 열에서 "{'id': xxx, 'name': 'yyy'}" 중 'yyy' 부분 추출
mr$genres <- gsub(".*'name':\\s*'([^']*)'.*", "\\1", as.character(mr$genres))

mr <- mr %>%
  mutate_at(c("genres"),factor)
```

### 3-3. 중간값 처리 및 이상값 처리 

budget, revenue 변수는 결측값은 너무 많기 때문에 중간값으로 정리하고 나머지 결측값은 삭제하겠습니다.

그 후 정수형 데이터는 이상값 처리를 진행하겠습니다.

```
install.packages("Hmisc")
library(Hmisc)
mr$budget <- impute(mr$budget, median) #mean, median, 특정숫자

#mr$revenue <- impute(mr$revenue, median) #mean, median, 특정숫자
#아무래도 revenue 는 반응변수 종속변수이다 보니 같은 숫자가 너무 많으면 
#문제가 될 것으로 예상됩니다. 또한 남은 결측값이 많긴 하지만
#데이터가 너무 많아 후에 있을 분석 진행에 차질이 발생하여
#na.omit으로 삭제한 후 진행하겠습니다.
mr <- mr %>% na.omit()
```

이상값 처리를 진행해보겠습니다.

```
# 이상치 및 결측값 처리 함수

calculate_outliers <- function(data, column_name) {
  iqr_value <- IQR(data[[column_name]])
  upper_limit <- summary(data[[column_name]])[5] + 1.5 * iqr_value
  lower_limit <- summary(data[[column_name]])[2] - 1.5 * iqr_value

  data[[column_name]] <- ifelse(data[[column_name]] < lower_limit | data[[column_name]] > upper_limit, NA, data[[column_name]])

  return(data)
}
table(is.na(mr))
boxplot(mr$budget,mr$popularity,mr$runtime,mr$revenue)
# 이상치 및 결측값 처리 및 결과에 대한 상자그림 그리기
mr <- calculate_outliers(mr, "budget")
mr <- calculate_outliers(mr, "popularity")
mr <- calculate_outliers(mr, "runtime")
mr <- calculate_outliers(mr, "revenue")

table(is.na(mr))
mr <- na.omit(mr)
table(is.na(mr))
boxplot(mr$budget,mr$popularity,mr$runtime,mr$revenue)#char형 변수를 제외하고 정수형 변수만을  boxplot을 그려보겠습니다.

```
![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/60efe2ae-f0b9-4475-86e4-14c259a60e14)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/6ef56431-3c9a-44f1-8a10-9b9fe3b9cdf7)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/22d4dcd9-6210-40bd-9dd1-3d3571b0be44)

## 4. bagging 및 RandomForest 분석

데이터를 학습 및 테스트 세트로 분할하고 모델을 생성하여 성능을 평가하고 직관적 이해를 돕기 위해 시각화 해보겠습니다.

### 4-1. 회귀 분석

무작위로 데이터를 분리하지 않고 반응변수를 중심으로 8:2로 나누기 위해 caret::createDataPartition을 사용하겠습니다.

```
install.packages("caret")
library(caret)
index <- caret::createDataPartition(y = mr$revenue, p = 0.8, list = FALSE)
train <- mr[index, ]
test <- mr[-index, ]
```

우선 bagging 모델을 생성하고 bagging 모델의 예측력을 확인해 보겠습니다.

```
model_bagging <-ipred::bagging(revenue ~ ., data = train, nbagg = 100)
predict_value_bagging <- predict(model_bagging, test, type = "class")%>%
  tibble(predict_value_bagging = .)
predict_check_bagging <- test %>% select(revenue)%>%dplyr::bind_cols(.,predict_value_bagging)
predict_check_bagging
```


![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/80125650-8b87-4a4d-a11b-9db9f9fdf0e7)


예측값이 우수하지 않아 보입니다. 

따라서 회귀 분석은 여기서 종료하겠습니다.

### 4-2. 분류 분석

분류 분석을 위해 정수형 변수 revenue를 factor형으로 변환해야 합니다. 

이를 위해

revenue의 평균값 미만 데이터는 0,

revenue의 평균값 이상 데이터는 1로 변환 후

factor형으로 형변환하겠습니다.

```
# revenue 열의 평균값 계산
revenue_mean <- mean(mr$revenue, na.rm = TRUE)

# 변환: revenue 열 값이 revenue 평균값 미만인 경우 0, 이상인 경우 1로 변경
mr$revenue <- ifelse(mr$revenue < revenue_mean, 0, 1)

# factor 데이터 유형으로 변환
mr$revenue <- factor(mr$revenue)
```

모델을 사용하여 test 데이터로 예측을 수행한 후 예측값을 저장하고 실제 데이터와 대조하여 확인해 보겠습니다.
(앞 코드와 동일합니다.)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/92413583-297a-4e99-b4bc-49e7cf34415d)

예측된 결과 어느 정도 예측을 수행한 것으로 확인할 수 있습니다. 

confusionMatrix를 활용하여 성능지표를 확인하겠습니다.

```
cm <- caret::confusionMatrix(predict_check_bagging$predict_value_bagging,test$revenue)
cm
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/b7b702e4-9943-45ee-9a6f-14567ba835fd)

이제 train 데이터로 RandomForest 모델을 만들어 보겠습니다.

```
library(randomForest)
model_rf <- randomForest(revenue ~ ., data = train, na.action = na.omit, importance = T, mtry = 7, ntree = 1000)
model_rf
```

만든 랜덤포레스트 모델로 예측을 수행한 후 실제 값과 결과를 비교해 보겠습니다.

(앞 코드와 동일합니다.)

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/e7e29874-e987-4a85-b032-288c381130b1)

예측을 잘 수행한 것을 확인할 수 있습니다.

이제 예측값을 저장한 데이터와 실제 데이터 사이의 confusionMatrix를 생성한 후 성능지표를 확인해 보겠습니다.

```
cm <- caret::confusionMatrix(predict_check_rf$predict_value_rf,test$revenue)
cm
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/555aa85a-9a96-4fe6-981c-69677ef9c7ff)


RandomForest 보다 bagging이 정확도, 민감도 등 여러 측면에서 나은 결과를 보이는 것을 확인할 수 있습니다. (75프로)

모델에서 변수의 중요도를 그림으로 나타내 보겠습니다.

```
varImpPlot(model_rf, type = 2, pch = 19, col = 1, cex = 1, main = "")
```

![image](https://github.com/auspicious0/MovieRevenue/assets/108572025/6075d09e-17a6-4b59-aa30-3e565217c99a)

수익(revenue)에 가장 중요한 요소는 budget(예산)과 인기, 상영시간, 장르 순으로 이루어진 것을 확인할 수 있습니다.


## 5. 문의
프로젝트에 관한 문의나 버그 리포트는 [이슈 페이지](https://github.com/auspicious0/MovieRevenue/issues)를 통해 제출해주세요.

보다 더 자세한 내용을 원하신다면 [보고서](https://github.com/auspicious0/MovieRevenue/blob/main/boxoffice_RandomForest.ipynb) 를 확인해 주시기 바랍니다.
