### 1. 프로젝트 소개  

#### 프로젝트 명  
세분화된 한국어 형태소 규칙에 기반한 의존구문분석 모델 개발  

#### 개요  
- 의존구문분석이란 자연어 문장을 지배소, 피지배소 의존 관계로 분석하는 구문 분석 방법론으로 한국어에 적합한 자연어 처리 방식
- 시중에 koBERT, Klue/Roberta-Base(or Large), KoElectra=base 등의 의존구문분석 모델이 이미 있으나 개선할 여지가 남아 있음  

#### 목표  
- 부산대학교 AILAB의 KLTagger를 활용하여 성능이 뛰어난 의존구문분석 모델 개발  
- 의존구문분석의 결과를 그래프로 시각화하여 출력해 사용자의 이해를 돕는 웹사이트 개발  

  
### 2. 팀소개

1.  김준기(junki121@pusan.ac.kr) - 모델 테스트, 분석 결과 시각화 그래프 구현
2.  박기훈(vpfmtlsl@gmail.com) - 모델 구현, 데이터 전처리, 웹 소켓 구현
3.  정대성(muntory2972@naver.com) - 모듈 구현, 데이터 전처리, 웹 서버 구축

### 3. 시스템 구성도

![p3](https://user-images.githubusercontent.com/37135296/195856424-e27f6345-7ba9-4cee-bc68-b62274cf9c3c.png)
![p2](https://user-images.githubusercontent.com/37135296/195856431-b77e7c0c-4f49-4da0-bf8f-c8674ff58891.png)
![p1](https://user-images.githubusercontent.com/37135296/195856436-65e7b56e-66a3-426c-92ac-474f51f044b5.png)

### 4. 소개 및 시연 영상

[![부산대학교 정보컴퓨터공학부 소개](http://img.youtube.com/vi/zh_gQ_lmLqE/0.jpg)](https://youtu.be/zh_gQ_lmLqE)
추가예정
### 5. 설치 및 사용법  

아래 주소에서 손쉽게 구문분석기를 사용할 수 있습니다.  
#### http://krparser.pythonanywhere.com/  

![p4](https://user-images.githubusercontent.com/37135296/195858085-cc078cd4-536a-49c3-9e99-373b46eaa983.png)  
![p5](https://user-images.githubusercontent.com/37135296/195858090-5b15e7bf-4c9a-4f12-9cd8-eddee58458b0.png)  
