# **Prefix Sum (Scan)**

병렬 스캔(Scan) 또는 접두사 합(Prefix Sum) 커널의 최적화는 **Sequential Recurrence Relation를 가진 계산을 병렬화**하는 과정에서  
**Work Efficiency**과 **병렬성(Parallelism)** 사이의 균형을 맞추는 것을 목표로 합니다.  

최적화는 다음과 같이 **단계적으로 진행**됩니다.  
**단순 병렬화 → 작업량 감소 → 메모리 접근 최적화**

---
## **Experiment Results**

## **Experiment Results**
- SECTION_SIZE 512  
- SUBSEC_SIZE 16  

### **# of elements: 512**
| Kernel                | Excution time (ms) |
|----------------------------|----------------|
| **Sequential Scan**          | 72.707069 ms    |
| **Kogge-Stone Scan**  | 0.021504 ms     |
| **Brent-Kung Scan** | 0.020480 ms  |
| **Coarsened Scan**   | 0.018432 ms  |


### **\# of elements: 10240**  
| Kernel                | Excution time (ms) |
|----------------------------|----------------|
| **Segmented Scan**   | 0.064512 ms  |
| **Single-pass Segmented Scan:**   | 0.030720 ms  |

---

## **1단계: Kogge-Stone Algorithm**

### **원리**
- 각 출력 요소에 대해 **병렬 리덕션(Reduction) 트리**를 구성하되, **Partial Sum**을 공유하여 계산 복잡도를 줄임.
- **N개 요소**에 대해 **log₂(N)** 반복으로 모든 계산을 완료 → **매우 빠른 속도**.
- 각 단계에서 스레드는 **stride 거리만큼 떨어진 요소의 값을 자신의 값에 더함**.

### **문제점**
- **Low Work-efficiency**  
  - 순차 알고리즘: `O(N)` 연산  
  - Kogge-Stone: `O(N * log₂(N))` 연산 → 데이터가 커질수록 비효율적.
  
- **Write-after-Read Hazard**  
  - 스레드가 값을 덮어써서 **Race Condition** 발생 가능.  
  - **해결책:** 임시 변수(register) + `__syncthreads()` 또는 **이중 버퍼링(Double-buffering)**.

---

## **2단계: Brent-Kung Algorithm**

### **최적화 원리**
- **두 단계로 분리하여 작업 효율성 향상.**
  1. **Reduction Tree**: 희소 리덕션 트리로 **부분 합** 계산. (총 `N-1` 연산)
  2. **Reverse Tree**: 부분 합을 전파하여 최종 결과 계산.

### **핵심 효과**
- **작업 효율성 향상:** 총 연산량 `2N - 2 - log₂(N)`, 즉 `O(N)` (순차 알고리즘과 동일).
- **속도 저하:** 2단계 수행으로 **Kogge-Stone보다 약 2배 느릴 수 있음** (자원이 충분한 경우).

---

## **3단계: Thread Coarsening (스레드 조밀화)**

### **최적화 원리**
- **스레드 하나가 여러 개의 입력 데이터를 순차적으로 처리**하여 효율성 향상.
- **3단계로 수행:**
  1. 각 스레드가 **subsection**에 대해 순차 스캔.
  2. **각 섹션의 마지막 값**만 모아 Kogge-Stone/Brent-Kung 스캔 수행.
  3. 스캔 결과를 **이전 섹션의 결과에 더해 최종 결과 완성**.

### **핵심 효과**
- **작업 효율성 극대화:** 병렬화 오버헤드를 줄이고 순차 스캔의 장점 활용.
- **메모리 접근 최적화:** 공유 메모리 사용으로 **coalesced 접근** 구현.

---

## **4단계: Hierarchical Scan (계층적 스캔)**

### **필요성**
- **단일 스레드 블록/공유 메모리의 용량 초과 시 필수적.**

### **최적화 원리**
1. 입력을 **Scan Block**으로 분할.
2. 각 블록이 **독립적으로 스캔 수행**.
3. 각 블록의 **총합을 배열 S**에 저장.
4. **배열 S에 대해 병렬 스캔** 수행 → 각 블록의 **누적 합 계산**.
5. 계산된 누적 합을 **각 블록 결과에 더하여 최종 완성**.

- 보통 **3개의 커널**로 구현: 초기 스캔 → 블록 합 스캔 → 최종 덧셈.

### **문제점**
- 3개의 개별 커널을 순차적으로 실행하는 방식입니다. 이 과정에서 불필요한 전역 메모리(Global Memory) 읽기 및 쓰기 작업이 발생.
---

## **5단계: Single-Pass Scan (Stream-based Scan)**

### **최적화 원리**
- **계층적 스캔의 중복 전역 메모리 접근 제거** → 메모리 효율 개선.
- **단일 커널 내 처리**: 인접한 스레드 블록 간 **도미노식(partial sum) 전달**.

#### **인접 블록 동기화**
- i-1번 블록 완료 시 **flags 배열**을 `atomicAdd()`로 업데이트.
- i번 블록은 flags를 **폴링(polling)**하여 대기 후 이전 블록 합을 가져와 계산.
- **`__threadfence()`**로 메모리 쓰기 순서 보장.

### **핵심 문제 및 해결책**
- **Deadlock(교착 상태):** 블록 실행 순서가 보장되지 않을 때 발생.
- **해결:** `atomicAdd()`로 **동적 블록 인덱싱 (Dynamic Block Index Assignment)**  
  → 논리적 실행 순서 보장.

---

## **최종 결론**

스캔 커널의 최적화는 **작업 효율성과 속도 간 트레이드오프 관리**가 핵심입니다.

- **Kogge-Stone:** 빠르지만 **비효율적**.
- **Brent-Kung:** 효율적이지만 **단계 수가 많음**.
- **Thread Coarsening:** 순차 계산과 병렬화의 장점을 혼합.
- **Hierarchical Scan:** **대규모 데이터** 처리 가능.
- **Single-Pass Scan:** **메모리 병목 제거**로 최종 성능 극대화.

