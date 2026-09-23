# PC-ALM Neural Studio (NumWriting)

> **In-Browser Continual Learning & Augmented Lagrangian Predictive Coding**  
> GitHub Pages에서 별도 서버 없이 동작하는 100% 클라이언트 사이드 손글씨 숫자 인식 및 실시간 런타임 학습 스튜디오

---

## 🌟 프로젝트 소개

본 프로젝트는 C 언어 네이티브 서버 기반이었던 `Sample` 프로젝트를 순수 **HTML5, CSS3, JavaScript(ES6+)** 환경으로 완전 포팅한 고성능 브라우저 신경망 애플리케이션입니다.

- **GitHub Pages 완벽 호환**: Node.js, Python, C 서버 등 백엔드 런타임 없이 정적 호스팅만으로 모든 기능 구동
- **순수 JS PC-ALM 신경망 엔진**: TypedArray(`Float32Array`) 기반의 초고속 벡터/행렬 연산 (추론 레이턴시 < 0.1ms)
- **브라우저 데이터베이스 (IndexedDB) 저장**:
  - 사용자가 런타임에 직접 교정한 손글씨 샘플 및 전/후 확률 변화, 썸네일 이미지 영구 보존
  - 온라인 적응 학습으로 갱신된 신경망 가중치를 IndexedDB에 자동 저장하여 새로고침/재방문 시에도 유지
- **클린 아이보리 & 소프트 화이트 테마**: 모바일 터치 및 PC 브라우저에 최적화된 반응형 폴리모픽 UI

---

## 📐 핵심 기술 및 수학적 원리 (PC-ALM)

PC-ALM은 글로벌 역전파(Backpropagation) 대신 각 계층 간의 국소 제약조건을 기반으로 학습하는 **예측 부호화(Predictive Coding)** 모델입니다.

1. **증강 라그랑지안(Augmented Lagrangian) 수식**:
   $$\mathcal{L}(h, \theta, \lambda) = \frac{1}{2} \|y - W_L h_{L-1} - b_L\|^2 + \sum_{i=1}^{L-1} \lambda_i^T (h_i - \mu_i) + \frac{1}{2} \sum_{i=1}^{L-1} \|h_i - \mu_i\|^2$$
   - $\mu_i = \text{LeakyReLU}(W_i h_{i-1} + b_i)$
2. **PI 피드백 제어기 동작**:
   - P항 (비례항): 계층 간 잔차 $r_i = h_i - \mu_i$
   - I항 (적분항): 누적 라그랑주 승수 $\lambda_i \mathrel{+}= \alpha \cdot r_i$
   - 결합 오차 $\tilde{r}_i = r_i + \lambda_i$를 통해 하위 계층까지 오차 소실 없이 신호 전달
3. **국소 가중치 갱신(Hebbian Update)**:
   $$W_i \leftarrow W_i - \eta_\theta (\delta_i h_{i-1}^T + \text{decay} \cdot W_i)$$

---

## 📂 파일 구조

```
NumWriting/
├── index.html               # GitHub Pages 메인 엔트리포인트
├── css/
│   └── style.css            # 반응형 클린 아이보리 & 화이트 디자인 시스템
├── js/
│   ├── pcalm.js             # PC-ALM 코어 신경망 엔진 (Primal-Dual 역학, 적응 학습)
│   ├── db.js                # IndexedDB 스토리지 관리자 (가중치 및 학습 이력 저장)
│   ├── mnist_samples.js     # 오프라인/정적 환경용 엄선된 200개 MNIST 테스트 샘플
│   ├── default_weights.js   # file:// 프로토콜 대응용 내장 기본 가중치 (Base64)
│   └── app.js               # 캔버스 드로잉(CoM 정규화), 실시간 추론, 이벤트 컨트롤러
└── data/
    └── pcalm_mnist.bin      # 원본 437KB PC-ALM 가중치 바이너리 파일
```

---

## 🚀 실행 및 배포 방법

### 1. 로컬에서 바로 실행
- `index.html` 파일을 더블 클릭하여 브라우저에서 바로 열거나, 간단한 정적 웹서버를 이용할 수 있습니다:
```bash
# Python 내장 서버 예시
python -m http.server 8000
# 브라우저에서 http://localhost:8000 접속
```

### 2. GitHub Pages 배포 방법
1. GitHub 저장소를 생성하고 본 `NumWriting` 폴더의 파일들을 푸시합니다:
```bash
git init
git add .
git commit -m "Initial release of NumWriting"
git branch -M main
git remote add origin https://github.com/<사용자명>/<저장소명>.git
git push -u origin main
```
2. 저장소의 **Settings** -> **Pages** 메뉴로 이동합니다.
3. **Build and deployment** -> **Source**에서 **Deploy from a branch**를 선택하고, `main` 브랜치의 `/ (root)`를 지정 후 **Save**합니다.
4. 수 초 후 부여된 `https://<사용자명>.github.io/<저장소명>/` URL로 접속하면 즉시 서비스가 시작됩니다!

---

## 💡 주요 기능

- **마우스 & 모바일 터치 드로잉**: 브러시 두께 조절, 지우기, 모바일 스크롤 방지 드로잉 지원
- **MNIST 규격 CoM 정규화**: 20x20 바운딩 박스 정렬 및 무게중심(Center of Mass) (13.5, 13.5) 자동 정렬
- **30ms 실시간 연속 추론**: 선을 긋는 도중에도 60fps에 가깝게 실시간 확률 바 갱신
- **실시간 런타임 적응 학습 (Continual Online Learning)**:
  - 잘못 인식된 숫자에 대해 실제 정답을 선택하고 "틀림! 정답 [X]로 즉시 런타임 학습" 버튼 클릭
  - 브라우저 메모리 상에서 가중치를 즉각 업데이트하고, 전/후 확률과 손실 변화량 즉시 피드백
- **브라우저 데이터베이스 (IndexedDB)**:
  - 학습된 가중치 자동 보존
  - 학습 기록 썸네일 확인, JSON 백업/가져오기, 가중치 .bin 다운로드 지원
- **실시간 신경망 내부 역학 모니터링**:
  - 은닉 계층별 활성화 강도 $\|h\|$, 최대 라그랑주 승수 $\|\lambda\|_{\max}$, 제약오차 $\|r\|_{\max}$ 게이지 바 제공
