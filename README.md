# 영화 수익 예측 (bagging, RandomForest)

본 문서는 Kaggle의 competition 데이터셋을 활용하여 데이터 분석 및 bagging, RandomForest를 통한 분류 모델 구축을 목표로 합니다. 이 데이터셋은 boxoffice 관련 정보를 담고 있습니다. 데이터를 적절하게 전처리한 후, 모델링과 예측을 수행합니다.

## 목차
1. [패키지 설치 및 그래프 설정](#1-패키지-설치-및-그래프-설정)
2. [데이터 수집](#2-데이터-수집)
3. [데이터 전처리](#3-데이터-전처리)
    1. [이상값 처리](#3-1-이상값-처리)
    2. [데이터 변수의 형 변환 및 삭제](#3-2-데이터-변수의-형-변환-및-삭제)
4. [의사 결정 트리 분석](#4-의사-결정-트리-분석)
    1. [데이터 분할](#4-1-데이터-분할)
    2. [의사 결정 트리 모델 학습](#4-2-의사-결정-트리-모델-학습)
    3. [모델 시각화](#4-3-모델-시각화)
    4. [가지치기 (Pruning)](#4-4-가지치기pruning)

5. [모델 평가](#5-모델-평가)
    1. [예측](#5-1-예측)
    2. [혼돈 메트릭스 (Confusion Matrix)](#5-2-혼돈-메트릭스confusion-matrix)
6. [결론](#6-결론)





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

그 후 다시 bagging, randomforest 분석 후 시각화하는 과정을 보이겠습니다.



이제 train 데이터를 가지고 DecisionTree 모델을 학습하겠습니다. 

```
install.packages("rpart")
library(rpart)

model_bc <- rpart(formula = diagnosis ~ ., data= train, method = "class")
summary(model_bc)

```

```
Call:
rpart(formula = diagnosis ~ ., data = train, method = "class")
  n= 359 

          CP nsplit rel error    xerror       xstd
1 0.67415730      0 1.0000000 1.0000000 0.09192627
2 0.06741573      1 0.3258427 0.5056180 0.07049103
3 0.06179775      2 0.2584270 0.4606742 0.06771241
4 0.01000000      4 0.1348315 0.3146067 0.05708944

Variable importance
        radius_worst           area_worst      perimeter_worst 
                  16                   15                   13 
           area_mean          radius_mean       perimeter_mean 
                  13                   13                   12 
        texture_mean        texture_worst concave points_worst 
                   3                    3                    3 
      concavity_mean           texture_se  concave points_mean 
                   1                    1                    1 
   compactness_worst      concavity_worst     compactness_mean 
                   1                    1                    1 
             area_se        smoothness_se 
                   1                    1 

Node number 1: 359 observations,    complexity param=0.6741573
  predicted class=B  expected loss=0.2479109  P(node) =1
    class counts:   270    89
   probabilities: 0.752 0.248 
  left son=2 (279 obs) right son=3 (80 obs)
  Primary splits:
      radius_worst         < 16.805    to the left,  improve=80.95968, (0 missing)
      perimeter_worst      < 105.95    to the left,  improve=79.58408, (0 missing)
      area_worst           < 865.7     to the left,  improve=79.26065, (0 missing)
      concave points_mean  < 0.051455  to the left,  improve=76.84859, (0 missing)
      concave points_worst < 0.1416    to the left,  improve=72.36913, (0 missing)
  Surrogate splits:
      area_worst      < 865.7     to the left,  agree=0.992, adj=0.963, (0 split)
      perimeter_worst < 109.75    to the left,  agree=0.969, adj=0.863, (0 split)
      area_mean       < 696.05    to the left,  agree=0.964, adj=0.837, (0 split)
      radius_mean     < 15.045    to the left,  agree=0.961, adj=0.825, (0 split)
      perimeter_mean  < 96.42     to the left,  agree=0.955, adj=0.800, (0 split)

Node number 2: 279 observations,    complexity param=0.06179775
  predicted class=B  expected loss=0.06810036  P(node) =0.7771588
    class counts:   260    19
   probabilities: 0.932 0.068 
  left son=4 (251 obs) right son=5 (28 obs)
  Primary splits:
      concave points_worst < 0.1349    to the left,  improve=13.611100, (0 missing)
      concave points_mean  < 0.048785  to the left,  improve=12.540700, (0 missing)
      concavity_mean       < 0.093405  to the left,  improve=10.696770, (0 missing)
      perimeter_worst      < 101.65    to the left,  improve=10.011740, (0 missing)
      area_worst           < 727.1     to the left,  improve= 9.007121, (0 missing)
  Surrogate splits:
      concave points_mean < 0.04804   to the left,  agree=0.943, adj=0.429, (0 split)
      concavity_mean      < 0.093405  to the left,  agree=0.935, adj=0.357, (0 split)
      compactness_worst   < 0.361     to the left,  agree=0.935, adj=0.357, (0 split)
      concavity_worst     < 0.3373    to the left,  agree=0.932, adj=0.321, (0 split)
      compactness_mean    < 0.1332    to the left,  agree=0.921, adj=0.214, (0 split)

Node number 3: 80 observations,    complexity param=0.06741573
  predicted class=M  expected loss=0.125  P(node) =0.2228412
    class counts:    10    70
   probabilities: 0.125 0.875 
  left son=6 (10 obs) right son=7 (70 obs)
  Primary splits:
      texture_worst   < 19.91     to the left,  improve=10.414290, (0 missing)
      texture_mean    < 16.37     to the left,  improve= 9.252306, (0 missing)
      concavity_mean  < 0.07265   to the left,  improve= 7.023810, (0 missing)
      concavity_worst < 0.2314    to the left,  improve= 6.156410, (0 missing)
      texture_se      < 0.4923    to the left,  improve= 5.327789, (0 missing)
  Surrogate splits:
      texture_mean  < 16.37     to the left,  agree=0.988, adj=0.9, (0 split)
      texture_se    < 0.47315   to the left,  agree=0.912, adj=0.3, (0 split)
      symmetry_mean < 0.14035   to the left,  agree=0.900, adj=0.2, (0 split)
      radius_se     < 0.2171    to the left,  agree=0.900, adj=0.2, (0 split)
      perimeter_se  < 1.5295    to the left,  agree=0.900, adj=0.2, (0 split)

Node number 4: 251 observations
  predicted class=B  expected loss=0.01593625  P(node) =0.6991643
    class counts:   247     4
   probabilities: 0.984 0.016 

Node number 5: 28 observations,    complexity param=0.06179775
  predicted class=M  expected loss=0.4642857  P(node) =0.07799443
    class counts:    13    15
   probabilities: 0.464 0.536 
  left son=10 (17 obs) right son=11 (11 obs)
  Primary splits:
      texture_mean        < 19.45     to the left,  improve=7.810924, (0 missing)
      texture_worst       < 27.49     to the left,  improve=6.706349, (0 missing)
      area_worst          < 724.05    to the left,  improve=5.357143, (0 missing)
      concave points_mean < 0.04944   to the left,  improve=3.778571, (0 missing)
      smoothness_se       < 0.005792  to the left,  improve=3.500000, (0 missing)
  Surrogate splits:
      texture_worst  < 27.49     to the left,  agree=0.893, adj=0.727, (0 split)
      texture_se     < 1.452     to the left,  agree=0.786, adj=0.455, (0 split)
      concavity_mean < 0.10298   to the left,  agree=0.750, adj=0.364, (0 split)
      area_se        < 24.89     to the left,  agree=0.750, adj=0.364, (0 split)
      smoothness_se  < 0.0073995 to the left,  agree=0.750, adj=0.364, (0 split)

Node number 6: 10 observations
  predicted class=B  expected loss=0.2  P(node) =0.02785515
    class counts:     8     2
   probabilities: 0.800 0.200 

Node number 7: 70 observations
  predicted class=M  expected loss=0.02857143  P(node) =0.1949861
    class counts:     2    68
   probabilities: 0.029 0.971 

Node number 10: 17 observations
  predicted class=B  expected loss=0.2352941  P(node) =0.04735376
    class counts:    13     4
   probabilities: 0.765 0.235 

Node number 11: 11 observations
  predicted class=M  expected loss=0  P(node) =0.03064067
    class counts:     0    11
   probabilities: 0.000 1.000 
```

CP 0.01로 더이상 분기하지 않습니다. 그 지점의 오류율(rel_error)과 교차검증오류율(xerror),교차검증오류의 표준편차(xstd)의 값을 확인합니다.

이 지점은 가지치기(pruning) 을 위한 최적의 lowest level 선택에 사용됩니다.

Variable importance 값은 둘레(perimeter_worst)이 가장 크고 반경(radius_worst), 지역(area_worst)이 그 다음을 차지합니다.

그 다음으로 모델에 관한 설명이 나오는데 이는 그림을 통해 살펴보겠습니다.

### 4-3. 모델 시각화

모델을 시각화 해 직관적으로 이해해 보겠습니다.

```
par(mfrow = c(1,1), xpd = NA)
plot(model_bc)
text(model_bc, use.n = TRUE)
```


<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/58e3afb0-b4de-459a-8df2-64a1452aeb29.png" width="400" height="400"/>

위 그림은 식별이 어렵습니다. 따라서 더 식별이 편한 그림으로 바꿔 보겠습니다.

```
install.packages("rpart.plot")
library(rpart.plot)
rpart.plot(model_bc)

install.packages(c("rattle","rpart.plot"))

library(rattle)
library(rpart.plot)
library(RColorBrewer)

fancyRpartPlot(model_bc)
```

<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/b43809e6-36ac-4b7f-a178-dd7574e3d6f0.png" width="400" height="400"/>
<img src="https://github.com/auspicious0/BreastCancer/assets/108572025/78a9cb98-7bbf-4e79-8232-18b30d9a3bbe.png" width="400" height="400"/>


radius_worst, area_worst, perimeter_worst 순으로 Variable importance를 차지하고 있으나

1. 데이터 양이 많지 않고
 
2. Variable importance 값 차이가 크지 않기 때문에 예상과는 다른 결정트리가 나왔습니다.

해당 트리가 과잉적합에 빠지지 않도록 모델 model_bc에 가지치기(pruning)을 하려고 합니다.


### 4-4. 가지치기(pruning)

우선, 교차 검증 오류율(xerror)이 최소가 되는 CP를 min_xerror_cp에 저장하게습니다

```
min_xerror_cp <- model_bc$cptable %>%
  as_tibble() %>%
  filter(xerror == min(xerror)) %>%
  pull(CP)
print("min_error_cp = ")
min_xerror_cp
```

```
[1] "min_error_cp = "
0.01
```

위에서 구한 min_xerror_cp 값을 이용하여 모델에 가지치기(pruning)를 수행하여 model_pr에 저장하겠습니다.

```
model_pr <- rpart::prune(model_bc, cp = min_xerror_cp)
fancyRpartPlot(model_pr)
```
![image](https://github.com/auspicious0/BreastCancer/assets/108572025/4e6b362c-ceed-45f7-895d-c2938a1dd00c)

## 5. 모델 평가

### 5-1. 예측

```
predict_value <- predict(model_pr, test, type = "class") %>% tibble(predict_value = .)
predict_check <- test %>% select(diagnosis) %>% dplyr::bind_cols(., predict_value)
predict_check
```

```

# A tibble: 39 × 2
   diagnosis predict_value
   <fct>     <fct>        
 1 M         M            
 2 B         B            
 3 B         B            
 4 M         M            
 5 B         B            
 6 B         B            
 7 B         B            
 8 M         M            
 9 M         M            
10 B         B            
# ℹ 29 more rows
```
모두 예측을 수행한 것을 확인할 수 있습니다.혼돈 메트릭스(confusionMatrix)를 활용하여 모델을 분석해 보겠습니다.

### 5-2. 혼돈 메트릭스(Confusion Matrix)

```
cm <- caret::confusionMatrix(predict_check$predict_value, test$diagnosis)
cm
```

```
Confusion Matrix and Statistics

          Reference
Prediction  B  M
         B 30  0
         M  0  9
                                     
               Accuracy : 1          
                 95% CI : (0.9097, 1)
    No Information Rate : 0.7692     
    P-Value [Acc > NIR] : 3.599e-05  
                                     
                  Kappa : 1          
                                     
 Mcnemar's Test P-Value : NA         
                                     
            Sensitivity : 1.0000     
            Specificity : 1.0000     
         Pos Pred Value : 1.0000     
         Neg Pred Value : 1.0000     
             Prevalence : 0.7692     
         Detection Rate : 0.7692     
   Detection Prevalence : 0.7692     
      Balanced Accuracy : 1.0000     
                                     
       'Positive' Class : B      
```
정확도 (Accuracy): 1.0

전체 예측 중에서 올바르게 분류한 비율로, 1.0또는 100%입니다.(TP + TN) / (TP + TN + FP + FN)

민감도 (Sensitivity): 1.0

실제 양성 중에서 올바르게 양성으로 분류된 비율로, 1.0또는 100%입니다.

특이도 (Specificity): 1.0

실제 음성 중에서 올바르게 음성으로 분류된 비율로, 1.0또는 100%입니다.

정밀도 (Precision): 1.0

정밀도는 모델이 양성으로 예측한 샘플 중에서 실제로 양성인 샘플의 비율을 나타냅니다. TP / (TP + FP)

재현율 (Recall): 1.0

재현율은 실제로 양성인 샘플 중에서 모델이 양성으로 예측한 샘플의 비율을 나타냅니다. TP / (TP + FN)

데이터가 작기 때문에 가능한 결과라고 생각하지만 100프로의 정확도 및 정밀도 등을 보인다는 점이 의미가 있고 결정트리를 통해 분석하기 좋은 데이터다라는 결론을 내립니다.

## 6. 문의
프로젝트에 관한 문의나 버그 리포트는 [이슈 페이지](https://github.com/auspicious0/BreastCancer/issues)를 통해 제출해주세요.

보다 더 자세한 내용을 원하신다면 [보고서](https://github.com/auspicious0/BreastCancer/blob/main/DesicionTree.ipynb) 를 확인해 주시기 바랍니다.
