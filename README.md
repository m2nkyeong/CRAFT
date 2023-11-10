# CRAFT
>CRAFT : Character-Region Awareness For Text detection Review    
>CFRAT GitHub Link : https://github.com/clovaai/CRAFT-pytorch.git

### CRAFT
* CRAFT의 목적 : Scene Text Detection
* Scene Text Detection 이란?    
  ▶️ 어떤 시각 데이터 (e.g. image, video clips etc.) 內 문자(text)의 위치를 식별(detection)하고 지역화(localization) 하는 것
  
* 기존의 문제점     
  * Bounding box가 직사각형 모양이 아닌 경우일 때, 결과가 좋지 않다.

* CRAFT의 제안
  * Region score & Affinity score(지역 점수 & 친밀도 점수)
  * Weakly-supervised learning(약한-지도 학습)



#### Ground Truth Label Generation
>먼저, 학습을 위해 합성 이미지에서 라벨링하는 방법이다.     
> real world dataset들과 다르게 만들어진 이미지들은 연구자들의 의도에 맞게 character 단위로 annotation이 제공된다.     
>그렇지만 이렇게 만들어진 데이터셋 역시 2가지의 score를 위한 라벨링 과정을 거쳐야 한다. 

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fb0kpq6%2Fbtq0Apl0ulu%2FdPtcMmAwgGhZCgKNvDsbQ0%2Fimg.png" width="60%" height="60%"></p>  

* **Region score** : 각각 픽셀이 character의 중앙에 있을 확률
* **Affinity score** : 픽셀이 인접한 character들의 중앙에 있을 확률(이 점수를 기반으로 개별 문자가 하나의 단어로 그룹화될 것인지가 결정)     
  * "**인접한 두 문자**"의 기준?     
    ➡️ 하단 이미지처럼 Region score(좌측 하단)는 각 픽셀이 문자의 중심에 가까울수록 확률이 1에 가깝고 문자 중심에서 멀수록 확률이 0에 가깝도록 예측하는 모델을 학습하는 것이다.
      이와 동시에 동일 모델은 각 픽셀이 인접한 문자의 중심에 가까울 확률인 Affinity score(우측 하단)도 예측할 수 있어야 한다.
      잘 학습된 모델을 통해 입력 이미지의 Region score와 Affinity Score를 예측하면 최종적으로 다음과 같은 Text Detection 결과를 얻을 수 있다.
<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*dh1tP51LL3QlPJvV7Anc-Q.jpeg" width="40%" height="40%"></p> 
<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*38BUBJ7F2yGim7WDevA-BA.jpeg" width="40%" height="40%"></p> 


#### Weakly-Supervised Learning
* Synthetic dataset은 character level의 annotation이 있었지만, Real world dataset에는 word level까지밖에 존재하지 않는다.    
* 이를 극복하기 위해 **weakly-supervised learning 방식**을 사용한다.     
* weakly-supervised learning이란 Stanford 대학의 연구팀에서 제시된 개념으로, 간단히 말하자면 training data를 프로그래밍을 통해 만들어내는 방식이다.
<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgIxrn%2Fbtq0y08yMO9%2FKZn5DoiMtwWW1Tuz3pUt3K%2Fimg.png" width="60%" height="60%"></p>      

* 위와 같이 주어진 word 부분을 crop해내고, 학습된 모델을 통해 word마다 character별로 Region score를 예측한다.
* 이후에는 watershed algorithm을 사용해 character region을 뽑아내어 이를 바탕으로 character level bounding box가 생성되고 원래 이미지에 반영된다.
* 이러한 일련의 과정을 거쳐 기존의 word-level annotation까지만 되어있던 dataset이 character-level의 정보까지 담게 되었고, 이후에는 위의 synthetic dataset을 처리한 것과 마찬가지의 방법으로 ground truth label을 만들어주면 된다.      
  ▶️ 그러나 weakly-supervised learning을 통해 탄생한 pseudo ground truth 역시 모델을 통한 예측이므로, 100% 신뢰할 수는 없기 때문에 각각의 pseudo ground truth에 대해 얼마나 신뢰할 수 있는지 계산할 필요가 있다.


#### Interim Model을 개선하는 방법
* 기존의 많은 Word-level Annotation 기반 데이터 집합을 활용하기 위해 해당 데이터들이 주어졌을 때, 앞에서 학습한 interim model을 이용하여 Character-box를 생성하는 방법을 소개한다.

<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbgIxrn%2Fbtq0y08yMO9%2FKZn5DoiMtwWW1Tuz3pUt3K%2Fimg.png" width="60%" height="60%"></p>     

* 위 이미지는 Word-level Annotation과 함께 실제 이미지가 주어졌을 때 Character-level Annotation을 구하기 위한 일련의 과정이다.

1. Word box 단위로 이미지를 Crop한다.
2. Interim model을 이용하여 Crop된 이미지의 Region score를 예측한다.
3. atershed algorithm을 이용해 문자 영역을 분리하여 Character box를 결정한다.
4. 마지막으로 분리된 Character box들이 Crop되기 이전의 원본이미지 좌표로 이동시킨다.

* Word-level Annotation으로부터 Character-level Annotation을 도출한 후 각 Character box를 이용해 Region score map과 Affinity score map을 구할 수 있다.

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*hWxBV_Dq3h8Rnl_PN8etLA.png" width="60%" height="60%"></p>     

* 이 방법을 통해 얻은 Ground Truth는 Interim model에 의해 예측된 것으로 실제 정답과는 오차가 존재한다. 따라서 Pseudo-Ground Truth라 부른다.
* Interim model로 예측한 Character box를 통해 얻은 Pseudo-Ground Truth는 학습 시 모델의 예측 정확도에 악영향을 줄 수 있다.     
➡️ 따라서, 학습 과정에서는 Pseudo-Ground Truth의 신뢰도를 반영하여 최적화한다.


#### pseudo ground truth의 신뢰도
* Pseudo-GT의 신뢰도(Confidence score)는 Interim model이 Cropped image에서 Character box를 얼마나 잘 분리했는지 정도로 결정된다.
* Confidence score는 각각의 Word Box: w 마다 다음과 같이 계산된다.
  
<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*L3EfP74mvespMDkoyaBbJQ.png" width="40%" height="40%"></p>  

* 실제 단어의 길이와 Interim model에 의해 Split된 Character box 개수 사이의 오차 비율을 계산하는 것이다.

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*jduSlZ9QaaeMypxZOrruAQ.png" width="50%" height="50%"></p>  

* 예를 들어, ‘COURSE’ 라는 단어(L(w)=6)의 Word Box를 Character Box로 분리하는 과정에서 5개의 Character Box가 예측되었다면, ‘COURSE’ Word box의 Confidence score는 5/6이 된다. ➡️ {6-min(6, 6–5)}/6

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*Jgw5dOlfM3eT1S9HnPm-WQ.png" width="50%" height="50%"></p> 

* 주의 해야할 점은 Word Box의 Confidence score가 0.5보다 낮을 경우이다.
* 이 경우, 모델 학습 시에 부정적인 영향을 줄 가능성이 매우 높기 때문에 추청된 Character Box를 그대로 사용하는 대신에 Word Box를 단어 길이 L(w)로 등분하고 Confidence Score는 0.5로 설정한다.
* 당연히 Pseudo-Ground Truth도 등분된 Character Box로 만들어낸다.

#### Generate Confidence Map
* 이미지의 각 Word-box에 대해 Confidence Score를 계산한 후 이 값들을 이용해 다음식을 만족하는 Confidence Map을 만든다.

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*7S7uCUIQjaIpIMznf2Meyw.png" width="50%" height="50%"></p> 

* 픽셀 p에 대해 p가 원본 이미지에서 Word-Box의 영역에 포함된다면 해당 좌표와 대응되는 Confidence Map의 좌표값은 Word-Box의 Confidence Score로 설정한다.
* 나머지 영역은 모두 1로 설정한다.

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*TyJbxiOL_Zc9qu-2Kva_7Q.png" width="50%" height="50%"></p> 

* 이렇게 만든 Confidence map은 Pseudo-GT 데이터 집합을 학습할 때 목적함수(Loss Function) 내에서 사용된다.

<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*yzer-vF0kGizphV1873A-Q.png" width="50%" height="50%"></p> 

* 결국 Word-box에 대응되는 픽셀들은 Interim model의 문자 영역 예측 정확도에 비례하는 가중치가 부여되는 것이다.
* 학습이 진행됨에 따라 Interim model은 문자를 점점 더 잘 예측하게 된다.

* 결과
<p align="center"><img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*pEylC3C_xkjLXpIs_dBagw.png" width="50%" height="50%"></p> 

#### Inference
<p align="center"><img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FbfZ6hs%2Fbtq0AqSL4ta%2FrnpGipk9dsE4YedAtd9RHk%2Fimg.png" width="50%" height="50%"></p> 

[추론 과정]
1. 우선, 이미지 사이즈의 binary map M의 값을 모두 0으로 초기화하고, M(p)를 예측된 region score와 affinity score에 대해 정해놓은 threshold를 넘긴다면 1로 설정한다.
2. M에 대하여 Connected Component Labeling을 수행하고 모든 라벨에 대응하는 connected component들을 포함하는 최소의 직사각형을 그려 QuadBox를 얻게 된다.
3. 추가적으로, scanning direction을 따라 character별 local maxima로 위 그림에서의 blue arrow들을 그리고, 그 중 가장 긴 것으로 길이를 통일해준다.
4. 각 arrow들의 중간점을 이어 yellow line을 그린다.
5. blue arrow들을 yellow line에 대해 수직이 되도록 회전시켜 red arrow로 만들고, 가장 바깥쪽에 있는 red arrow들을 yellow line을 따라 움직여 text region을 모두 포함할 수 있도록 한 후, 각 red arrow들의 양 끝 점인 control point들을 이으면 text polygon이 완성된다.
