# Rovent 구현·인수

## 목표와 기준

시작 기준은 PR25·PR26이 통합된 `609819ded605795c9ff0653adad391e875b1c4b2`다.
`feature/rovent-reciprocal-nnuec`에서 별도 PR로 구현한다. 사용자 제공
`Rovent_Integrated_Handoff_20261016.zip`과 depth/mate 결정 메모는 설계 자료이며,
현재 소스·관측과 구별한다. 이 문서가 새 작업의 항목·배정·인수 기록을 연결한다.

이번 목표는 **학습 실행을 제외한 플랜 구현**이다. 학습 도구·recipe·checkpoint·자료
계약은 구현하되 optimizer update·본학습·대규모 수집을 실행하지 않는다. CPU/mock,
실제 CPU 수치, 로컬4050 정확성, 원격 비용·기력 비교의 증거는 따로 관리한다.
기존 미관측·NOT_READY는 새 문서나 녹색 CI로 완료 처리하지 않는다.

큰 체크포인트마다 총괄이 항목별 근거와 남은 소비자를 대조하여 사용자에게 알리고,
완료한 담당에게 다음 독립 항목을 배정한다. 작은 의미 단위의 커밋·push·Draft PR
공유는 전체 검증을 기다리지 않는다. 학습·원격 실행·병합을 자동으로 진행하지 않는다.

## M0 최소 계약

[rovent 검색 계약](../crates/rz-contracts/src/rovent.rs)은 별도
`rovent-search/0.1` envelope다. 기존 EvalRequest/EvalOutput·PALS0.1·P/C/V artifact
역할 번호와 V1~V5 기록을 바꾸지 않는다.

- `SearchRecipient`는 P/C/R 수신자이며 R은 `Role::Validator`의 별칭이 아니다.
- `ExecutionMode`의 Hybrid/CPU × P-C-R/P-C는 Collection/Deployment 목적과 독립이다.
  실제 backend와 native R 조립을 검사하며 조용한 CPU fallback이나 숨은 R은 허용하지 않는다.
- `QuestionEnvelope`은 원 root authority·parent·sender/recipient·정확 anchor·root-ply·
  원 first move·prefix/line/offset·view·질문·근거 revision·root budget·원 deadline을 보존한다.
  Store pin·Rules 검증·예산 ledger는 실제 manager의 소유다. digest가 capability를 대신하지 않는다.
- `SearchGoal`의 selective chess-ply depth와 mate move는 nodes·시계·NN 횟수와 독립이다.
  mate의 양쪽 horizon은 각각2N-1/2N이며 overflow와 등록 상한을 거부한다.
- typed framing 검사가 실제 질문 소비·합법성·물리 완료를 입증하지는 않는다.

## 확정된 depth·mate·root 제한

네 모드 모두 `searchmoves`, `depth`, `mate`를 정식 지원하는 것이 완료 조건이다.
parser만 받거나 원 driver가 무시하는 경로는 지원 완료가 아니다. legacy driver는
명시 capability가 없으면 새 목표를 검색 취소 전에 거부한다.

root 제한은 Rules의 합법 순서에서 한 번 봉인하고 P 제안·C 귀속·R root 작업·proof·
최종 선택·실패 fallback·ponder 보유 결과까지 전달한다. 제한은 root 첫 수에만 적용한다.
불법 또는 빈 명시 목록을 unrestricted나 체스 terminal로 바꾸지 않는다.

depth는 root-relative 선택적 chess-ply horizon이다. 실제 경계 평가와 선언된 후보·
반박·Repair 의무가 완료돼야 한다. 생성 길이·R 지역 depth·빈 큐를 완료로 대체하지 않는다.
전폭 또는1..D 전역 barrier를 필수로 두지 않고 시간제 go에 새 depth pass를 강제하지 않는다.

depth·mate 요청이 없는 선택 탐색은 별도 `selective_frontier` 상태로 완료를 기록한다.
명시적으로 봉인한 비어 있지 않은 책임 집합이 모두 완료되고 실제 native 사용량과
예약·미완료 실행이 정산돼야 한다. 빈 큐·빈 seal·role budget 소진만으로 이 상태를
발행하지 않으며, depth·mate 증명이나 물리 완료를 대신하지 않는다. 이전 소스의
`pending` 원자료는 새 상태로 재분류하지 않고 새 바이너리의 실제 실행으로 확인한다.

원 role-call 상한이 실제로 소진된 종료는 `stopped_role_budget`으로 구분한다.
이는 `go nodes`의 `stopped_work_budget`과 다른 원인이며, 수락 중인 role 결과를
버리거나 이미 완료한 depth·mate·선택 탐색을 중단으로 덮는 근거가 아니다.

`native-validation-diagnostics`의 NNUEC 상세 관측은 기존 root receipt와 별도
`rz-rovent-nnuec-owner-observation/1` domain을 쓴다. 원 authority와 admission seal,
실제 준비·설치 ACK·joined engine·퇴역 readback의 제한 크기 사본이며 새 capability가
아니다. 상세 누락과 미관측은 보존하고, accepted C의 의미 연결이나 조건부 scalar value를
설치 성공만으로 추론하지 않는다. 마지막 engine의 stack 계수는 해당 engine 수명의
값으로 표시하며 root 전체의 재개 합계로 사용하지 않는다.

같은 opt-in에서 원 manager의 bounded CPU slice handle을 root 입장 전에 설치하고,
root 반환 뒤 `rz-rovent-nnuec-cpu-slice-observation/1`로 실제 정산 prefix를 관측한다.
root당 최대32개이며 원 Job·질문·view·grant·ExecutionId·Yield ack·정산·worker 재개
시간을 보존한다. 원 admission과 같은 authority로 대조해야 하며, 누락·overflow 뒤의
전체 건수는 unknown이다. 이 관측은 token 권한·accepted C·조건부 value·물리 완료를
발급하지 않는다. 기존8KiB 행·1MiB stream과 마지막 lifecycle 행의 예약을 공유한다.

mate는 실제 Rules terminal에서 OR/AND로 전파한다. 한 유효 공격과 필요한 모든 합법
방어의 근거가 닫히면 강제 승리·패배 모두 조기 종료할 수 있다. 거리 상한과 정확 최단
거리를 구분하고 처음에는 정확 거리만 `score mate`로 출력한다. NN값·외부 mate·단일
협조적 PV는 증명이 아니다. 제한 root의 패배는 그 후보 집합에 조건부다.

P-C에서는 R/PVS/qsearch/SEE/외부teacher 호출이 실제0이어야 한다. 부족한 방어는 P/C
질문으로 처리한다. infinite/ponder의 계산 완료와 bestmove 출력 권한은 분리한다.

`go nodes`의 Rovent 단위는 `rovent-position-visits/1`이다. 실제 root 입장,
P/C 또는 mate 검토의 상태 전이, native R의 실제 방문 수를 합산한다. qsearch·SEE는
기존 native nodes의 부분이므로 다시 더하지 않는다. NN 호출·물리 입력 완료·PUCT
simulation으로 환산하지 않는다. 원 nodes 한도와 역할별 호출·native 상한을 각각
보존하며, 다른 엔진의 nodes와 같은 작업량이라고 해석하지 않는다.

## 구현 항목과 배정

진행 표는 source/소비/검사/실제 실행을 구분한다. 아래 첫 기록은 완료표가 아니다.
실행 원자료·장비 식별·개인 경로는 저장소 밖 보고서에 보존한다.

| ID | 책임 | 첫 구현/남은 인수 |
|---|---|---|
| RVH-E01 | 외부 연결 | 기존 SF 관측 재사용 조건 확인; Repair→C→Fresh 실제 소비 이월 |
| RVH-E02 | 외부 프로필 | 엔진·목적·옵션·자산·capability 분리; legacy guard 보존 |
| RVH-E03 | Reckless | 공식 pin·라이선스·실제 지원 제약 확인; query별 연결·종료 |
| RVH-E04 | 자료·외부 | raw CP/mate/PV·teacher와 내부 Fresh/Rules 근거 분리 |
| RVH-E05 | arena·프로필 | teacher/checker/opponent 목적과 전체 자원·시계·자산 기록 |
| RVH-A01 | 총괄 | 검색 envelope 게시; 실제 producer/consumer 연결 필요 |
| RVH-A02 | coordinator | 6방향 bounded 질문·합류·ancestor 대기·무진척·취소 |
| RVH-A03 | coordinator/R | anchor·prefix·응수별 frontier와 원 예산 소유 |
| RVH-A04 | CPU/R | 넓은 조사·지속 심화·R→P/C 실제 요청과 응답 소비 |
| RVH-A05 | CPU/runtime | paused owner admission 첫 변경; feedback 재개·정산 인수 |
| RVH-A06 | runtime | 미착수 local-first 분배·bounded 대기·결과 집계 |
| RVH-N01 | NNUEC Rust/참조 | 필수 명시 config·weights FP32 모듈; 독립 수치 대조 |
| RVH-N02 | NNUEC | B/C/L delta·실패 원자성·undo/full rebuild |
| RVH-N03 | NNUEC | anchor·수순 순서·승격·반박/복구 관계 보존 |
| RVH-N04 | CPU/NNUEC | 실제 조건부 value·후보 소비; 활성 전 N07 |
| RVH-N05 | 문맥·자료 | 보존·다음 계산 효용과 실제 결과/비용; heuristic 진단 분리 |
| RVH-N06 | 문맥 | raw/compact/view·착수 rebase·lease·가상토큰 중복 방지 |
| RVH-N07 | CPU/TT | history·model·C/L·조건 namespace·same-owner 재개 |
| RVH-M01 | 총괄/UCI | 네 모드 실제 조립·P-C R0·명시 backend |
| RVH-M02 | 총괄/runtime | 관리자·P/C·R·외부 실행의 전체 CPU/메모리·V-free 출전 |
| RVH-T01 | 자료 | 당시 입력/질문/문맥·출처 seal과 미래 target 분리 |
| RVH-T02 | 학습 도구 | 두 frozen actor·공통 학생/한 learner·checkpoint 후 교사별 fork |
| RVH-T03 | 자료·출전 | teacher/V 경험·V-free 경험·계보/전역 holdout |
| RVH-T04 | 학습 도구 | optimizer 소유·fork/resume·checkpoint/export 구현; 실제 학습 제외 |
| RVH-I01 | UCI | ponderhit 정상 시계 전환·miss/stop/root 교체·한 번 출력 |
| RVH-I02 | 총괄/UCI | Rovent 표시·artifact/호환 alias; 과거 domain 대량 치환 없음 |
| RVH-I03 | UCI/search | parser 첫 변경; root 제한·goal·horizon/proof·네 모드 인수 |
| RVH-V01 | 총괄 | 기존 원 근거의 영향 입력 대조; 미관측·NOT_READY 보존 |
| RVH-V02 | 담당별 | 새 기능의 실제 의미·오류·동시성·수명·수치 회귀 |
| RVH-V03 | 계측/arena | 기능 안정 후 별도 비용·동일 시간 대조; 구현 검사와 구별 |
| RVH-V04 | 총괄 | 체크포인트·잔여 배정·문서·Draft PR·최종 항목별 감사 |
| RVH-O01 | 조건부 후속 | 지역 커널 최적화는 병목 근거 후 한 변수 실험 |
| RVH-O02 | 조건부 후속 | 원본 외부 NNUE FT는 권리/필요성/학습 허용 후 별도 |
| RVH-O03 | 조건부 후속 | NUMA/다중GPU/원격은 실제 topology·장비 프로필별 |
| RVH-O04 | 조건부 후속 | compact 직접 입력·공동학습은 별도 모델/encoding |

## 통합 순서와 검증

첫 CPU 통합 체크포인트에서 확인한 결과는 아래와 같다. 실행 원본은 저장소 밖
`rovent-implementation-20261017-1` 보고서에 보존한다. 이 표는 작업 중인 소스의
로컬 검사이며 최종 통합 SHA·목표 GPU·성능·학습·대국 인수를 뜻하지 않는다.

| 실제 실행 | 결과와 범위 |
|---|---|
| 공통 계약 | 29개 통과 |
| search 라이브러리 | 723개 통과, 기존 선택 실행 2개 미실행; 최초 reciprocal·NNUEC·공통 depth/mate·동기 호환 owner 포함 |
| root 제한/batch 계약 | `experimental-batch` 조합 31개 통과 |
| NNUEC V1 독립 reference | Python 13개 통과; Rust caller와 별도 13입력 대조에서 raw/WDL 최대 차이 0 |
| 학습 준비 | neural import를 차단한 21개 통과; optimizer·tensor 복원·학습·실제 export는 미실행 |
| 새 명세와 CLI | 의미 검사 13개·CLI 9개 통과; 선언/저장 구조 검증이며 actual 실행 권한은 발급하지 않음 |

수치 대조는 명시적으로 만든 작은 미학습 FP32 파라미터만 사용했다. NNUEC V1의
정확 합산·B/C/L·조건별 TT 근거를 보존하며, 제안/반례/복구의 kind와 directed
parent/supersedes를 구별하는 입력은 **별도 V2 모델**로 추가한다. V1 자산을 V2로
자동 변환하지 않는다. 문맥 효용은 아직 실제 보존/경로 정책을 바꾸지 않는 관측이다.

상대 prefix 봉인, 공통 nodes 한도, native accepted-role checkpoint, 비동기 manager,
새 UCI owner와 early-terminal depth 영수증의 후속 변경은 위 최초 실행 이후에도
진행 중이다. 최종 소스를 고정한 뒤 영향 검사를 다시 실행해야 한다. native
selected-view 모델은 동기 호환 owner에서 명시적으로 거부하며, 실제 ticket facade와
R-free 조립이 연결되기 전 네 모드의 실제 신경망 지원 완료로 세지 않는다.

1. M0 계약·기준선과 UCI root 제한, NNUEC reference, 외부 프로필을 병렬 연결한다.
2. 기존 stage/worker/ledger를 재사용해 질문 그래프와 R-free 실제 경로를 만든다.
   suspended parent의 owner를 idle로 취급하거나 paused task를 다른 worker로 옮기지 않는다.
3. 같은 root-ply controller·Rules reducer로 네 모드의 depth/mate와 UCI 수명을 연결한다.
4. NNUEC의 명시 FP32 수치·delta·순서·undo를 대조한 뒤 context/local TT namespace를
   먼저 잠그고 실제 CPU 가치 소비를 활성화한다. 미확정 폭을 기본 모델 사양으로 고정하지 않는다.
5. 자료/학습 준비 consumer·다중teacher provenance·전체 계보 holdout을 연결한다.
   실제 학습을 하지 않아도 잘못된 resume/fork·state 변경·입력 누출은 검사한다.
6. 기능별 CPU·실제 NN/GPU·외부 owner 검사를 인수하고 남은 이월 항목을 따로 보고한다.

원 상호 질문·취소·정산·physical fence를 우회하는 테스트 전용 제품 경로는 추가하지 않는다.
일반 대국·학습·기력/성능 승격은 단위 검사나 구현 파일 존재로 대신하지 않는다.

## 두 번째 통합 체크포인트

후속 작업 중 search 746개가 통과했고 기존 선택 실행 2개는 미실행으로 남겼다.
이후 semantic Store/model과 R owner를 값 이동으로 분리한 소스는 제품 컴파일을
통과했으며 768개 중 764개 통과·fixture 2개 실패·기존 선택 실행 2개 미실행을
관측했다. 실패는 앙파상 FEN의 fullmove 번호와 실제 Proposal bundle이 없는 peer
재시도 fixture에서 발생했다. Rules·필수 bundle 검증은 유지하고 수정 후 재검사한다.

NNUEC V2는 별도의 미학습 FP32 모델로 등록했다. 네 record kind와 directed
parent/supersedes, 원 first move와 rebased suffix를 구별한다. 독립 Python 검사
13개, utility reference 10개와 Rust/Python 고정 입력 20개 대조를 통과했으며 후자의
raw/WDL 최대 절대 차이는 0이었다. 이 결과는 작은 합성 입력의 수치 동등성이고
실제 R evaluator 설치·GPU·기력·성능의 근거로 확대하지 않는다. utility는 계속
실행 권한과 known target을 발급하지 않는 shadow 관측이다.

새 corpus parser의 Rust 컴파일과 Python 계약 검사 14개, 학습 창·sampler의
표준 라이브러리 검사 5개도 통과했다. recovered wire 자료는 원 native/teacher
실행 권한을 발급하지 않는다. Torch forward/gradient·tensor 복원과 실제 trainer
실행은 이 로컬 결과에 포함하지 않았다. 이번 목표에서 실제 학습은 제외한다.

PR의 CPU 모델 CI에서 관측한 54개 실패는 모두 기존 fixture의 고정 Rust source
pin 변경 거부였다. 검증을 끄거나 실행 때 현재 hash를 자동 신뢰하지 않는다.
기존 closed source pair를 보존하고 실제 영향 경로를 검토한 새 조합을 별도로
등록한 뒤 같은 SHA의 CI로 재검사한다. bindings 성공과 나머지 CI 상태를 구분한다.

남은 핵심 연결은 실제 네 모드의 native 조립, R-free selected-view owner,
공통 node quota와 Candidate/Counter/Repair depth·Rules mate frontier, 실제
NNUEC 준비/worker 설치, 외부 UCI 실실행, typed 원자료·trainer/checkpoint 소비다.
각 접점의 source 준비와 실제 인수를 나누고 Draft PR에서 계속 공유한다.

후속 수정·module 등록 뒤 search 783개 중 **781개 통과·기존 선택 실행 2개
미실행**, experiments 234개 통과를 확인했다. 여기에는 semantic/R 필드 분리,
실제 accepted-source publication과 first move 변조 거부, Frontier의 typed 원
질문·Fresh/Rules·lease 한도, 공통 원장 영수증, NNUEC V2 CPU/utility 검사가
포함된다. 새 worker evaluator 설치는 이때 제품 컴파일만 확인했으며 설치 전용
수명 회귀와 상위 실제 job 연결은 다음 인수다. search 구현 digest에 새 Frontier
소스를 포함한 provenance 갱신도 별도로 기록한다.

## 세 번째 통합 체크포인트

이후 작업 중인 소스에서 search **798개 통과·기존 선택 실행 2개 미실행**,
`pals-adaptive,onnx-cpu` UCI 라이브러리 **604개 통과**를 관측했다. P-C 전용
semantic port, 독립 pending 질문, 설치 전용 evaluator 수명, typed Frontier
입력과 원 종료 오류의 보존이 포함된다. 이 checkpoint 뒤에도 PC의 실제
Frontier 소비·NNUEC 최초 준비·정상 예산 종료를 연결하고 있으므로 이 결과를
최종 소스나 네 모드의 신경망 종단 인수로 재사용하지 않는다.

표준 라이브러리 기반 준비·corpus 계약 검사는 Windows에서 **81개 통과·Linux
전용 9개 건너뜀**, WSL Ubuntu에서는 **90개 모두 통과**했다. 이 중 새 학습
도구의 순수 계약 검사는 48개다. 모델·optimizer 실행을 차단한 검사이며,
실제 학습·Tensor 복구·수치 export·GPU 성공을 의미하지 않는다. 별도로 준비한
PyTorch forward/gradient·Tensor 수명 검사 20개는 로컬 Torch 부재로 미실행이다.

새 `rz-rovent-source-aggregate/1`은 원 commit과 정렬된 component map을 길이
프레임으로 묶는다. `rovent_source_aggregate_sha256`의 독립 Python 고정값 대조와
입력 한도/식별 거부 검사 두 개가 통과했다. 이 값은 선언한 source set의 식별이며
compiled/pinned bytes 대조나 실행 권한을 대신하지 않는다. corpus의 runtime
식별은 실제 loaded runtime library의 SHA-256이며, 전체 runtime 구성의 집계로
해석하지 않는다. 원 bundle·backend·native configuration·lock은 별도로 보존한다.

원격 `3a0be9f` CI의 Linux/Windows 실패는 Ponder 활성화 fixture가 합법적인
예측 착수를 포함한 출력을 거부한 사례였다. 원 제한 착수와 상대의 합법적인
예측 착수를 따로 확인하도록 수정한 뒤 위 로컬 604개가 통과했다. 독립 bindings
lock의 누락된 `memoffset` 의존 항목도 package version 변경 없이 동기화했다.
모델 CI의 source pin 거부와 최종 SHA의 CI 인수는 여전히 별도이며, 자동 hash
신뢰나 기존 receipt 재서명으로 해결하지 않는다.

Linux 전용 수집·학습 parent의 첫 교차 컴파일에서는 collector 접점 네 곳의
타입/가시성 불일치를 관측했다. 이력은 보존하고 실제 NNUEC module·원 native
입력 key·공개 SHA 경계로 수정한 뒤 같은 feature의 Linux 대상 lib/tests 교차
컴파일이 통과했다. 교차 컴파일 통과 자체도 Linux
자원 적용·실제 child·외부 ELF·GPU·학습 성공과 구별한다.

다음 병렬 인수는 실제 네 모드 launcher와 startup 설정, 원 질문에 묶인
Candidate/Counter/Repair·mate missing queue, 동일 preparation actor의 최초
NNUEC R 준비, 원 수집 cap과 trainer parent, 외부 UCI 실제 실행이다. 이후
CPU/비학습 Tensor·목표 GPU 수치·물리 수명을 확인한다. 실제 학습은 이번 목표에서
제외하며 Draft PR의 병합도 이 checkpoint에 포함하지 않는다.

이 checkpoint 당시 학습 도구의 추가 소스 감사에서는 같은 부모가 보유한 live
corpus/window custody로만 재개할 수 있고, 부모 재시작 뒤 durable dataset/checkpoint를
소비하는 제품 경로가 없음을 확인했다. 당시 T04 미완료 기록은 보존한다. 후속은 caller가 exact pin한 원
publication과 전체 rows/Rules/target/계보를 재검증하는 historical dataset 자격,
checkpoint 복구 자격과 새 trainer 호출부를 연결한다. 과거 native 실행·물리 완료
권한은 재발급하지 않으며, 미회수 실행과 checkpoint 이후의 비용 unknown도 유지한다.

다음 고정 소스의 추가 통합에서는 search **812개 통과·선택 실행 2개 미실행**,
UCI **607개 통과**를 관측했다. 최초 NNUEC 준비의 회귀 13개에는 동일 actor의
실제 handoff·미인수 엔진 회수·다른 actor 거부·실제 Ready 경계가 포함된다.
이는 작은 미학습 CPU 모델과 소유권 검사이며 전체 launcher의 실제 NN 인수가 아니다.
초기 node 0/1·제한 child·Rules-only mate와 원 cleanup ceiling도 검사했다.

독립 소스 감사에서 아직 인수하지 않은 두 실제 연결도 확인했다. P-C의 missing
defense와 accepted selection은 원 controller에서 먼저 청구한 checked snapshot을
Store에 채택해야 하며, 그 이전 Rules 전이를 허용하지 않도록 보완한다. 학습
window의 기존 단일 runtime 필드는 collection ORT library와 trainer Python
executable을 같은 값으로 요구하므로 새 명시 codec에서 분리한다. 같은 가짜 SHA를
쓴 fixture 통과를 실제 실행 가능성으로 해석하지 않는다.

## 네 번째 통합 체크포인트

collection의 실제 ORT library 식별과 trainer의 Python executable·Torch endpoint
식별을 나눈 명시적인 학습 window `/2`를 추가했다. 이전 `/1` parser·canonical
bytes·digest는 보존한다. 새 profile은 optimizer·loss의 부동소수 계수를
`coefficient-f64-bits-v1`로 봉인하여 Python과 Rust의 JSON 지수 표기 차이를 피한다.
source와 모델 의미, weights와 raw 파일 hash, collection epoch와 training epoch도
서로 대체하지 않는다.

historical corpus/checkpoint·unknown 비용 보존·새 child authority·resume/fork/export
소비자를 연결하고, Rust trainer CLI와 원자적 publication 경로를 작성 중이다.
저장된 자료의 검증은 과거 native 실행 권한을 재발급하지 않는다. 원 프로세스가
미회수 상태이면 실제 OS 회수 증거를 요구하며, 완료 이후의 unknown 비용을 0으로
바꾸지 않는다. 순수 metadata 검사 **35개 통과**를 `f7aabf4`로 공유했고, 후속 소비자
계약 검사는 Windows **87개 통과·Linux 전용 8개 미실행**, WSL Ubuntu **95개 모두
통과**했다. 실제 모델·Torch tensor 복원·optimizer·학습의 인수가 아니다.

기본·`onnx-cpu` 단독·`pals-adaptive` 단독 UCI 라이브러리의 세 컴파일 조합이
통과했다. 앞선 원격 `215442d`의 네 CPU/bindings/model job 실패는 optional
dependency를 기본 feature에서 참조한 동일 오류였다. 소스에서 feature 경계를
수정했으며 원 실패 로그를 보존한다. 이 로컬 컴파일 결과를 새 HEAD의 CI 성공으로
표시하지 않는다.

## 다섯 번째 통합 체크포인트

원 `go`를 ponder 정규화 전에 보존하고, startup 자산·설정과 매 검색의 실제
Rules/history·허용 수·시계·예산을 구분하는 typed root admission을 등록했다.
기존 simulation 단위와 새 position-visits 단위를 명시적으로 나누며, Rovent
capability를 등록한 실제 입구만 `nodes 0`을 허용한다. 잘못된 새 `go`는 기존
검색을 취소하기 전에 거절한다. 과거 root descriptor의 검증은 새 live 권한을
발급하지 않는다.

PCR의 P/C 선택은 실제 head를 소비하거나 graph를 변경하기 전에 원 controller에서
Rules 경로를 청구한다. 이미 지불한 snapshot은 재생 없이 채택하며, 원 native
예약이 살아 있는 `NodesReserved`만 같은 accepted head를 보존한 채 기다린다.
PreparingRoot는 같은 semantic Store·ledger·Frontier·Source·최초 view/seal·실제
Ready를 move한다. 일반 검색의 passive mate는 이미 검사한 Rules 증거만 읽는다.
typed mate 전략과 실제 최종 착수의 연결 및 PCR GoalLocal 발주는 후속이다.

Rust `rz-rovent-trainer` entry, 원자적 corpus publication, historical dataset과
checkpoint 재검증, 실제 parent/child 호출·boundary publication 소비자를 등록했다.
이는 실행 가능한 제품 경로의 소스 등록이며, 실제 trainer·tensor 복원·optimizer
실행 성공은 아직 아니다. 별도 R readout과 auxiliary 자료·teacher worker의 초안은
이 단위에 섞지 않았으며 원 P/C 자료로 다시 이름 붙이지 않는다.

현재 Python 순수 계약 묶음 189개는 Windows에서 **180개 통과·Linux 전용 9개
미실행**, WSL Ubuntu에서 **189개 모두 통과**했다. 실제 신경망 import와 optimizer
실행을 차단했다. 원 producer의 dependency directory·training epoch 0·명시 runtime
bundle 필드 및 read-budget을 함께 검사한다. 이전의 겹치는 묶음과 합산하지 않는다.

새 Linux 교차 컴파일 첫 두 실행에서는 UCI 연결의 타입 오류 세 건과 Arena의
RoleFailure 표현·새 PalsError 분기 누락 두 건을 확인했다. 원 실패와 정리 영수증을
보존하고 좁게 수정했으며, 전체 target·새 회귀 검사 결과는 다음에 별도로 기록한다.
원격 `95e7a83` CI의 누락된 process-boundary fixture, fdopen 테스트 인수 불일치와
독립 source pin 거부도 각각 구분한다. 누락 파일과 fixture는 보완했고, 고정 pin은
최종 소스 감사 후 정확한 literal 조합으로 갱신한다. 이를 exact-HEAD CI 성공으로
표시하지 않는다.

## 여섯 번째 통합 체크포인트

validation 008에서 Linux 네 패키지의 전체 타깃 교차 컴파일과 UCI 기본·
`onnx-cpu` 단독·`pals-adaptive` 단독 컴파일이 통과했다. 같은 실행의 search는
원 실패 fixture를 보존하고 수정한 뒤 **848개 통과·기존 선택 실행 2개 미실행**이다.
실제로 검사된 terminal mate를 단순 depth 종료로 기대한 두 fixture를 보완했고,
restricted-loss fixture는 전체 합법 수 목록과 실제 부분집합을 구분한다.

UCI는 **635개 통과·2개 실패**였다. 유효하지 않은 all-zero Rules 선언을 사용한
fixture와 시작점 이전 시계의 거부 검사를 구분해 수정한다. Windows의 미세한
음수 기간 변환에 의존하지 않도록 원 `Instant` 순서를 직접 검사하며, 원 1ns
회귀를 삭제하거나 정상 시간으로 바꾸지 않는다. 이 실행은 experiments 단계에
도달하지 않았으므로 해당 패키지의 신규 통과를 주장하지 않는다.

원 R 자료의 selected anchor와 record별 source history를 분리해 보존하는 최신
순수 검사 묶음은 Windows·WSL에서 **각각 47개 통과**했다. 실제 Rust serde 입력을
production Python validator에 전달하는 별도 wire 진입점도 작성했다. 앞의 189개
묶음과 이력을 단순 합산하지 않으며, 순수 자료 검사는 Rules 재생·native 권한·
tensor·optimizer 성공의 증거가 아니다.

새 등록 소스에는 private Bound 인스턴스 설치 권한, 동일 준비 actor의 실제 CPU
evaluator 해제와 별도 cache ack, 원 source snapshot 공유, 순수 historical review
검증 및 독립 Collection CPU_T worker가 포함된다. 영수증이 View pin을 보유하지
않으며 같은 의미 digest의 다른 Bound를 인수하지 않도록 한다. 이 단위는
`f6579e7`의 별도 검사 worktree에서 검증하고, 원 feature worktree의 병렬 구현과
구분한다. 소스 등록·읽기 감사와 실제 검사 결과를 서로 대체하지 않는다.

빠진 terminal fixture 보정을 별도 커밋한 `d12cc13` 고정 소스에서는 search
**878개 통과·기존 선택 실행 2개 미실행**, Linux 네 패키지의 전체 타깃 교차
컴파일 통과, experiments **253개 통과**를 확인했다. 관리 실행의 자식 종료와
임시 경로 정리도 확인했다. Linux 전용 teacher pipe의 실제 실행이나 이후
SourceSeal·GoalLocal·제품 entry의 인수 결과는 아니다. 독립 Collection teacher와
기존 PC corpus의 별도 순수 묶음 47개도 Windows·WSL에서 각각 통과했다.
이 47개는 위의 R helper 묶음 47개와 서로 다른 실행이며 단순 합산하지 않는다.

다음 인수는 네 모드 제품 입구와 실제 Ponder 시계·출력, 원 depth/mate 의무의
후속 질문과 native 소비, Q별 NNUEC 종단 수명, 독립 teacher와 R publication,
학습을 실행하지 않는 tensor·checkpoint·export 및 로컬 GPU 정확성이다. 실제
학습·원격 성능/기력 비교·Draft PR 병합은 진행하지 않는다.

실제 Candidate/Counter/Repair admission, retired Fresh, Rules terminal과 명시적
round 종료를 공통 Frontier에 전달하는 manager hook을 등록했다. 첫 탐색 재검사에서
**824개 통과·1개 실패·기존 선택 실행 2개 미실행**을 관측했다. 실패는 selection
seal 전에 전체 완료 depth를 기대한 새 fixture였다. terminal 의무와 전체 선택
완료를 구분하도록 fixture를 수정하고 재검사하며, 제품 완료 조건을 완화하지 않는다.

같은 등록 단위를 수정 뒤 재실행하여 search **825개 통과·기존 선택 실행 2개
미실행**, `pals-adaptive,onnx-cpu` UCI **617개 통과**, experiments **236개 통과**를
확인했다. 관리 빌드의 자식 종료·임시 경로 정리도 확인했다. 기존 준비·corpus의
9개 Python 모듈은 현재 소스에서 Windows **93개 통과·Linux 전용 9개 미실행**,
WSL **102개 통과**다. 앞의 95개 소비자 묶음과 겹치는 검사가 있어 두 수를 독립
검사 총수로 합산하지 않는다. 이는 작업 중인 소스의 CPU 계약 검사이며, 새 제품
entry·실제 NNUEC R·GPU·학습의 인수로 확대하지 않는다.

남은 실제 조립은 다음과 같다.

- 네 모드 제품 UCI entry, 실제 `go`별 root admission, 초기화·새 게임·ponder·정리.
- 원 node quota를 먼저 청구한 PCR 전이와 GoalLocal 후속 작업. 일반 검색의 passive
  mate 전파, 명시 mate의 PCR missing defense, 정확 거리의 UCI 출력도 연결한다.
- 최초 준비 전후 동일 semantic·ledger·actor·Ready worker를 이동하고, 후속 질문마다
  실제 view-bound NNUEC 설치·namespace·R 결과의 P/C 소비를 연결한다.
- actual collection 목적의 명시 startup, 원 Rules replay·teacher·R readout·divergence
  출처와 durable publication을 연결한다. V-free profile의 task는 미지원/unknown을
  보존하고 별도 PCV 실행을 추가하지 않는다.
- trainer CLI와 실제 부모·child, 비학습 tensor 복원·export, 네 모드 신경망·로컬
  GPU 수치·물리 수명과 최종 SHA CI를 각각 인수한다.

성능·기력 비교와 실제 학습은 이 checkpoint에 포함하지 않는다. O01~O04의 조건부
후속 연구를 기반 구현의 선행 조건으로 추가하지 않는다.

## 일곱 번째 통합 체크포인트

검사는 별도 검증 worktree의 고정 SHA에서 수행하고, 병렬 작업 중인 소스와
구분한다. 다음 수치는 서로 겹치는 suite가 있으므로 합산하지 않는다.

| 고정 소스·실제 검사 | 결과와 한계 |
|---|---|
| `3792d7d`, Linux 네 crate 전체 target 타입 검사 | 통과. `pals-collection-onnx`와 그 의존 feature 조합이며 실제 ELF·신경망 실행은 아님 |
| 같은 소스, 공통 계약 | 15개 통과 |
| 같은 소스, search | 903개 통과·5개 실패·기존 선택 실행 2개 미실행 |
| `e7e488b`, Linux 타입·공통 계약 | 통과·15개 통과 |
| 같은 소스, search | 906개 통과·4개 실패·기존 선택 실행 2개 미실행. UCI·experiments 단계에는 도달하지 않음 |
| `45cc094`, 제출 입력·기존/새 corpus·teacher·R ancestry 순수 검사 | Windows·WSL 각각 215개 통과. neural import를 차단했으며 forward·optimizer 실행 없음 |
| `fb777dc`, R auxiliary 순수 검사 | Windows·WSL 각각 83개 통과. 위 suite와 중복됨 |
| 같은 소스, native build bundle 순수 검사 | Windows 7개 통과·Linux 전용 3개 미실행, WSL 10개 통과 |

search 실패 자료는 보존한다. 잘못된 fixture의 미청구 Rules 경로와 모델 value 등록을
수정하되 제품 guard를 완화하지 않는다. 새 CPU observation의 실제 출처는 동기 경로에도
연결하며, 십진 f32 고정 WDL의 오차는 허용 오차를 늘리지 않고 정확한 입력으로 고친다.
이 후속 수정과 새 제품 세션의 검사는 위 결과 이후에 별도로 수행한다.

새 소스는 다음 경계를 연결했다. 이것은 최종 인수표가 아니다.

- 원 selected view의 source snapshot 배열을 공유하고, 실제 worker가 활성화한
  capture lineage와 Job 수준의 raw/검색 소비/P-C view 소비 witness를 구분한다.
- 이미 지불한 GoalLocal 전이를 재시도할 때에도 취소·마감·불명확한 native 수명을
  검사한다. 미회수 입력과 Repair 보고서는 같은 semantic owner에 보존한다.
- 같은 ticket의 Ponder 시계와 출력, 초기 C·미착수 materialization·종료한 worker의
  서로 다른 cleanup 권한, 명시 startup configuration `/2`를 연결한다.
- 외부 R의 실제 current-root source·stop/drain·reset proof를 별도 타입으로 제공한다.
  외부 reported nodes를 native credit으로 환산하지 않는다.
- submitted input `/1`, raw `/3`, corpus `/2`를 별도 형식으로 소비하고 원 형식의
  canonical과 ancestry를 유지한다. known divergence의 실제 owner 근거 연결은 남았다.

남은 필수 작업은 네 모드의 제품 실행·정상 종료, 초기 C 중단 후 같은 owner 회수,
외부 R의 실제 발주·Store·receipt, raw3 수집·퇴역·publication의 종단 연결, 실제
Repair/teacher/R 출처 검증, 비학습 tensor·checkpoint·export 및 로컬 4050 정확성이다.
최종 source closure·고정 source pin·동일 SHA CI도 확인해야 한다. 실제 학습은
이번 목표에서 제외하며, 로컬 검증 결과를 원격 성능·기력 비교로 보고하지 않는다.

## 여덟 번째 통합 체크포인트

P-C/P-C-R의 깊이는 사용자 제공 depth·mate 설계에 따라 root 상대의 선택적 체스
ply로 유지한다. 생성 chunk, 지역 CPU depth, 문맥 길이, 왕복 횟수를 UCI depth로
바꾸지 않는다. 메이트 후보 한 줄과 강제 메이트 증명도 구분한다. Rules의 OR/AND
근거로 필요한 전체 방어가 닫힌 강제 승패는 조기 종료할 수 있으며, 정확한 최단
거리를 확인하기 전에는 `score mate`의 정확 거리로 표시하지 않는다.

다음은 별도 고정 worktree에서 실제 실행한 검사다. 소스 등록과 미실행 후속은
이 표의 성공으로 확대하지 않는다. suite 사이에 겹치는 검사가 있으므로 합산하지 않는다.

| 고정 소스·검사 | 실제 결과 |
|---|---|
| `f39f828`, Linux 네 crate 전체 타깃 타입 검사·공통 계약 | 통과·15개 통과 |
| 같은 소스, search | 925개 통과·기존 선택 실행 2개 미실행 |
| 같은 소스, UCI | 실제 검사 중 NNUEC owner fixture에서 Windows stack overflow로 비정상 종료. 전체 UCI 성공·experiments 신규 성공은 아님 |
| `40b926b`, 준비 owner의 타입 크기 관측 | layout 검사 통과. 준비 transport가 약 37KiB임을 확인했으나 RSS·peak·성능 개선의 측정은 아님 |
| 같은 소스, Ponder | 세 사례 모두 대기 timeout으로 실패 |
| `8c7c291`, 기본 Windows stack의 준비 transport·Ponder | layout 1개와 Ponder 3개 통과. 실제 첫 Ready까지 도달한 뒤 원 native/manager 자원 한도 불일치로 owner 검사 1개 실패 |
| `8371962`, Linux 전체 타깃 타입 검사 | 새 profiled 접점의 API·fixture 접근·금지된 unsafe 사용으로 실패. 실제 ELF fixture 실행 전임 |
| `6c6bf61`, Linux 전체 타깃 타입 검사 | 첫 실패의 private 접근·unsafe 경계는 수정됨. 초기 AttackRanking fixture 의존 선언 누락과 cleanup 진단 타입 오류로 실패 |
| `ec0f740`, 제출 입력·corpus·teacher·divergence·R ancestry 순수 검사 | Windows·WSL 각각 243개 통과. neural import·forward·optimizer 실행 차단 |

stack overflow 보완은 테스트 stack 크기를 올리지 않고 원 준비 소유자 묶음을 같은
힙 할당에 유지하는 방식이다. 이후 자원 fixture는 같은 명시 한도를 native service,
Deferred CPU plan, manager에 전달하도록 수정했다. 실제 오류 자료와 원 guard는
보존하며, 이 자원 수정 이후의 종단 통과 여부는 다음 고정 소스 검사에서 확인한다.

외부 R의 실제 source·requested/issued constraint·Dispatch·동일 terminal·Store 게시·
P/C continuation·종료 증거를 연결했다. 기존 설정 `/1`·`/2`의 bytes는 유지하고
새 설정 `/3`에서 독립 외부 질문 한도를 명시한다. 프로세스 직접 종료의 증거는
semantic close 전용이며, 과거 ready fence나 실제 join을 다음 root 권한으로 바꾸지
않는다. 디스크 회수 전 원 packet·View·Store pin을 유지하는 durable sink와 실제
제품 입구 연결은 아직 별도 인수 대상이다.

새 raw3·corpus2의 producer·writer·publication·historical reader와 공유 Collection
launch를 등록했다. 초기 `counter=None`의 실제 AttackRanking과 나중 visible-counter
입력은 서로 다른 원 source로 보존한다. 후발 witness를 같은 원 retired C row에
연결하되 이미 게시한 입력을 다시 쓰거나 미래 반례를 과거 tensor에 넣지 않는다.
원 source 해시만 같다는 이유로 새 helper closure를 자동 허용하지 않는다.

남은 작업은 다음과 같이 계속 병렬 진행한다.

- 외부 R의 durable retirement와 정상 다음 root·newgame·process close를 실제 인수한다.
- 실제 Collection 상위 실행기, 게시 전 유한 C row 보관, 원 Repair·teacher·R 출처와
  FD publication·재개·검증 소비자를 종단 연결한다.
- native R의 실제 capture·semantic witness와 optional peer→외부 R 후속을 연결한다.
- 네 모드의 depth·mate·searchmoves·Ponder와 NNUEC 원 owner 회수를 같은 소스에서 검사한다.
- 기존 source review의 helper closure를 엄격하게 등록하고, 최종 형식·CI를 확인한다.
- optimizer를 실행하지 않는 tensor·checkpoint·export와 로컬 RTX 4050의 수치·물리
  수명을 확인한다. 이 로컬 실행은 성능·기력 자료로 사용하지 않는다.

실제 학습·원격 성능 및 기력 비교·PR 병합은 이번 체크포인트에서 실행하지 않는다.

## 아홉 번째 통합 체크포인트

고정 소스 `c865391`의 Linux 대상 네 crate 전체 타깃 타입 검사와 공통 계약 15개가
통과했다. Windows에서 search는 961개 통과·기존 선택 실행 2개 미실행,
experiments는 267개 통과, arena는 286개 통과다. UCI는 704개 통과·2개 실패이며
전체 성공으로 표시하지 않는다. 두 실패는 실제 PC 질문의 공개 revision 증가와
테스트 평가기의 고정 `record_revision=0`이 충돌한 경우다. `913fb5e`에서 전용
fixture가 원 질문 revision을 봉인하도록 수정했으며 제품 수락 guard는 유지한다.
수정 후 재검증 결과는 이 원 실행과 별도로 기록한다.

원 소스와 빌드·실행 증거를 연결하는 경계도 보완했다.

- `742b8cc`의 source review는 명시적으로 검토한 helper closure·실제 compiled
  search 사실·원 binary/build/manifest pin을 요구한다. 현재 HEAD를 자동 승인하는
  registry는 두지 않으며, 최종 literal profile 등록과 실제 소비자 검사는 남아 있다.
- `6998950` 이후 native upper는 원 resource plan의 FD·bytes·SHA와 같은 lock으로
  모델 준비 전에 자원 scope를 적용한다. 시작 admission과 종료 후 엄격 비교를
  분리하며, 불명확한 ancestry·역할별 thread·peak는 실제 적용 성공으로 바꾸지 않는다.
- `4b4cd23`의 외부 Collection은 원 joined worker source와 같은 RootAdmission을
  `/2` admission 및 raw3 소비자에 연결한다. own `/1`·기존 raw2 bytes는 유지한다.
  보고된 외부 nodes를 native credit으로 환산하지 않는다. 엄격 자원 HOLD보다
  원 corpus 게시·FD 무결성 확인을 먼저 수행한다. 실제 수집 종단 인수는 미실행이다.
- `57b4723`은 compiled search source 목록의 실제 production child 13개 누락을
  보완한다. 새 search·resolver implementation pin을 등록하며 과거 pin·semantic
  revision·기록을 재작성하지 않는다. V5 JSON/Rust 목록도 같은 생산 경로를 등록한다.
- 확실한 Windows 프로세스 미시작 오류는 원 `CreateProcessW` 실패 코드와 함께
  보존하고 소유 lease·scratch만 회수한다. 시작 여부가 불명확한 실패에는 적용하지
  않는다. 전용 저장소 검사 18개 통과·Linux 전용 3개 미실행이며, 최초 오류 193의
  환경 원인이 규명됐다는 뜻은 아니다.

신경망 import와 optimizer를 차단한 자료 소비자 검사는 `c865391`에서 Windows·WSL
각각 264개가 통과했다. `639eb32`의 명시적 미학습 NNUEC initializer 검사까지 더한
순수 Windows 검사는 271개가 통과했다. 실제 초기화·독립 수치 대조·LoadedCollection
원 발급·GPU·학습·성능 인수는 서로 다른 단계다. Torch 의존 slot/recheck suite는
순수 실행기의 import 차단에 걸렸으므로 이 통과 수에 넣지 않는다.

잔여 작업과 병렬 책임은 다음과 같다.

1. 총괄·Historical: UCI 두 실패 수정, 최신 Linux 타입·feature·형식·CI와 최종 source
   literal profile을 확인한다. 독립 NNUEC 수치 대조는 비교기 자체의 실패도 보존한다.
2. E·N·F: 실제 native R observer 설치, 원 semantic retirement 회수, 원 R publication과
   연결한 별도 semantic 부속 자료 및 엄격 소비자를 닫는다. 실제 Collection에서
   게시·재개·FD 검증까지 관측하기 전에는 종단 완료로 표시하지 않는다.
3. CPU·Manager·Core·F: N05의 원 utility 입력·실제 descendant·새 witness·같은 비용
   조건을 연결한다. manager fence만으로 whole cost를 닫지 않고 같은 actor의 필수
   release/cache ACK까지 확인한다. 알려지지 않은 비용·효용은 마스킹하며 실제
   보존/재방문 ablation이 없는 앞 네 축의 target을 만들지 않는다.
4. 총괄·N·F: 실제 미학습 P/C source와 NNUEC 자산의 등록·reference·checkpoint·export
   소비를 검사하고, local RTX 4050에서 수치·물리 수명·정상 종료를 확인한다.
   이 로컬 검증을 원격 성능·기력 자료로 사용하지 않는다.

실제 optimizer update·본학습·원격 비교·PR 병합은 이번 목표의 완료 조건에 추가하지
않는다. 소스 준비·CPU 통과·실제 모델/GPU 인수를 분리하고 목표는 계속 진행한다.

## 열 번째 통합 체크포인트

`639eb32`의 후속 CPU 실행에서 UCI 706개가 모두 통과했다. 명시적 미학습 NNUEC
자산을 이용한 독립 산술 대조도 20개 입력에서 통과했다. Rust가 생성한 실제 입력을
별도 stdlib reference로 계산한 결과이며, Rules 인코더의 독립 구현·기력·GPU·학습
증거는 아니다. 최초 비교기 오류와 수정 후 영수증은 각각 보존한다.

`1db90f1`의 원 utility action은 질문·view·실제 descendant·새 witness를 연결하고,
같은 actor의 해제와 cache ACK 및 PinZero 이후에 비용을 닫도록 구현했다.
validation030은 utility 입력 준비의 오류 변환 두 곳과 root 후보 반복 한 곳에서
컴파일에 실패했다. `1d9256e`에서 이 세 곳과 collection origin의 포맷을 수정했다.
해당 고정 소스의 순수 소비자·utility 명세 검사는 신경망 import·optimizer를 차단한
Windows·WSL에서 각각 369개 통과했다. 최신 Rust 전체 검사는 진행 중이며, 새
CaptureError 표시 변환 오류가 발견돼 전체 통과로 표시하지 않는다.

사용자가 승인한 N05 후속 범위는 선택형 cohort 비교 실행기다. 단일 root의 서로
다른 시각에 얻은 결과를 사후에 같은 비교 쌍으로 바꾸지 않는다.

- 입력·가중치·정확 Rules 이력·후보·예산·자원·준비 절차를 결과 전에 등록한다.
  실제 Ready 상태에서 봉인한 문맥으로 독립 root의 P/C/R trial을 최대 세 개 실행한다.
- trial의 원 authority·시계·hard deadline·완료·미해결·물리 종료를 따로 보존한다.
  원 cache 준비와 실제 관측 근거가 부족하면 known pair를 만들지 않는다.
- cohort 비용은 실제 route·제한된 역사 자료 materialization·같은 actor의 물리 drain과
  PinZero를 포함한다. durable dataset I/O는 별도 수집 비용으로 기록하며, 원 root-local
  비용에서 사후에 시간을 빼지 않는다.
- 새 비용 domain은 `rz-rovent-utility-cohort-cost/1`, 명세는
  `rz-rovent-utility-training-preparation/2`다. 기존 preparation `/1`과 root-local
  source·bytes·unknown 상태를 유지한다.
- 첫 자료 단위는 원 cohort publication 하나와 별도 durable ACK다. 다context 집계는
  이 자료를 암묵적으로 확장하지 않으며, 과거 dataset은 학습 계보 검증과 구분한다.

병렬 잔여는 CPU·Manager·E의 실제 cohort 발급·실행·게시, MF·F의 원 FD 소비와
utility checkpoint/export 연결, 총괄의 최신 Rust·형식·CI 및 최종 source literal
검사다. optimizer를 호출하지 않는 tensor 검사와 실제 모델·로컬 GPU 수명 인수는
순수 검사의 통과 수와 별도로 관리한다. 실제 학습·원격 비교·PR 병합은 실행하지 않는다.


## 열한 번째 통합 체크포인트

`4fa43c2`의 Linux 대상 workspace 전체 feature·전체 target 타입 검사와 형식 검사가
통과했다. 같은 소스의 runtime 82개, search 971개, UCI 712개 검사도 통과했다.
search의 기존 선택 실행 2개는 미실행이다. 신경망 import·optimizer를 차단한 Windows
순수 소비자 검사는 462개 통과했다. suite 간 중복이 있으므로 이 수를 합산하지 않는다.
같은 소스의 Clippy는 UCI 구조·기계적 지적으로 실패했으며, 후속 수정의 통과를 이
원 실행에 소급하지 않는다.

`bfe9037`은 선택형 cohort의 실제 Ready inventory, 원 trial admission, 강제 recipient의
원 offer 선택, 같은 actor의 cache ACK와 물리 퇴역·PinZero를 연결한 WIP다.
trial마다 새 native product를 준비하며 원 authority·시계·입력·자원 근거를 보존한다.
서로 다른 trial의 절대 tick이나 마지막 publication의 clock을 수치로 비교하지 않는다.
trial 간 순서는 원 durable ACK와 등록 sequence로만 연결한다. 이 고정 소스의 후속
검사는 진행 중이며 첫 전체 feature 빌드는 JSON macro 안의 배열 문법으로 실패했다.
이를 정상 실행이나 cohort 양성 인수로 표시하지 않는다.

원 Store 관측의 기존 binary framing은 일부 conditional comparison basis를 전부
포함하지 않는다. SHA·known flag·종료 bool만으로 과거 원 근거를 복구하지 않는다.
새 cohort의 양성 효용을 위해 실제 owner에서 별도 basis를 봉인하고 원 observation·
witness·publication과 대조한다. 근거가 빠진 과거 자료는 계속 unknown/masked다.
기존 framing·codec·canonical bytes와 이미 게시된 기록은 바꾸지 않는다.

계속 병렬 진행하는 잔여는 다음과 같다.

1. CPU·Core·Manager·E: 별도 conditional basis와 실제 cohort 상위 실행기·bounded
   materialization·durable publication·역사 reader를 끝까지 연결한다. 실제 물리
   퇴역 전에 원 View·Bound·자료 owner를 해제하지 않는다.
2. MF·F: tagged PC/R/utility 조상과 현재 cohort의 원 FD 소비, 다섯 상태 checkpoint,
   export 및 재개 소비자를 연결한다. 신규 utility window의 명시 checkpoint 여유로
   새 forward/update admission을 일찍 닫되 원 hard deadline은 늘리지 않는다.
3. 총괄: 최신 고정 SHA의 전체 feature·기본 feature·순수 검사·Clippy를 재검증하고,
   실제 source closure와 literal review profile 및 CI를 확인한다.
4. 별도 실행 인수: optimizer를 호출하지 않는 실제 tensor·checkpoint·export와
   local RTX 4050의 수치·물리 수명·정상 종료를 확인한다. 로컬 자료를 원격 성능·
   기력 비교나 실제 학습 완료로 사용하지 않는다.

실제 optimizer update·본학습·원격 비교·PR 병합은 실행하지 않았으며 이번 목표에서
실제 학습을 제외하는 사용자 지시는 유지한다. 목표는 계속 진행한다.

## 열두 번째 통합 체크포인트

`bbb0675`의 Linux 대상 workspace 전체 feature·전체 target 타입 검사와 Clippy가
통과했다. Windows에서 실제로 실행한 UCI 723개, arena 299개, 조건부 근거 8개,
취소된 permit 1개, 기존 backend 식별 byte 호환 1개 검사도 통과했다. 검사 프로세스
종료와 관리 scratch 정리를 확인했다. 서로 겹치는 suite의 통과 수는 합산하지 않는다.

`5341d90`에는 저장된 수치 자료만 검증하는 명시 CLI를 추가했다. 같은 원 FD의
크기·SHA와 변조 상태를 검사하고 기존 numerical admission validator를 호출한다.
현재 native 권한·readiness·GPU 물리 완료를 새로 발급하지 않는다. 시간 검사는
기존 streamed hash·validator의 호출 경계와 작은 read에서 수행하며 내부 강제
선점을 제공한다고 표시하지 않는다. inode의 동시 쓰기를 동결하는 장치도 아니다.
Linux 기본 feature 3개·인수 feature 5개, Windows 기본 3개·인수 4개 검사가
통과했다. 같은 소스의 compiled inventory 검사와 실제 Linux 실행 파일 빌드,
Linux 대상 전체 Clippy도 통과했다.

실제 비학습 모델 검사는 다음 두 실행을 구분한다.

- `024f303`의 단순 초기화·P/C export·Python 수치 검사는 성공했지만 원 G0
  initialization evidence가 없는 진단 실행이다. 이 자료를 정식 G0로 재표기하지 않는다.
- `5341d90`에서는 새 디렉터리에서 실제 `initialize_g0_with_evidence`를 호출했다.
  seed 23의 미학습 checkpoint, V-free P/C export, CPU Torch reference와 실제
  Rust·ORT CPU raw를 생성했다. 기존 validator가 Fresh B1·FP32·TF32 off의
  12그룹·100사례와 입력 tensor 974,988 bytes를 대조해 통과했다. 실제 native
  NN 입력 완료는 public/role을 합쳐 200개이며, 사례 수·탐색 소비·방문 수와 다르다.

원 raw에서 재계산한 최대 절대 차이는 logits `2.1457672119140625e-6`, policy
`2.3100085400074022e-8`, WDL `2.0932634803294548e-7`이다. 기존 절대·상대
허용 오차를 유지했다. Rust native worker join과 외부 process exit 0을 보존했다.
optimizer·backward 호출 및 시도는 모두 0회다. 이는 tensor 수치·기본 수명 인수이며,
독립 Rules 인코더·현재 GPU·대국·성능·기력·학습 인수로 확대하지 않는다.

### N05의 별도 개입 정책

기존 `/1`은 이미 실제로 Ready인 작업 중에서 선택한다. 보통 같은 first의 P→R→C는
한 frame의 순차 transition이며, reciprocal 진행 중의 원 custody는 clean snapshot
조건을 만족하지 않는다. 서로 다른 first의 P/R offer를 같은 target의 비교로 묶을 수
없다. 따라서 `/1` 실행 선언과 CPU 검사만으로 실제 P/C/R 비교 쌍이 만들어졌다고
표시하지 않는다.

사용자의 추가 선택에 따라 `/2`는 기본 탐색과 분리한 선택형 실험 정책이다.
`RoventUtilityInterventionCohortV2`와 `FrozenProposalReviewV1`을 명시하고,
같은 bounded prelude에서 실제로 만든 공통 target bundle과 원 NNUEC 입력을
고정한다. 독립 root마다 실제 P 제안 연장, C divergence 또는 own R 검증 중 하나를
발주한다. 처음 정책은 실제 counter 근거 없는 Repair를 만들지 않는다.

공통 target·Rules 전체 이력·선택 records·관계·first·NNUEC 입력의 seal과, 각 arm의
실제 purpose별 query/view seal은 별도로 보존한다. 기존 Ready offer·accepted role
출처를 위조하거나 서로 다른 role query를 같은 full view라고 부르지 않는다. Search가
발행한 move-only target/candidate를 coordinator의 별도 admission으로 소비한다.
원 source frame의 이중 실행을 막고 그 원문·증거는 root 물리 종료까지 보존한다.

profile은 `rz-rovent-utility-comparison-cohort/2`, target은
`rz-rovent-utility-intervention-target/1`, trial·pair는 각각 기존과 다른 `/2`
domain이다. `/1` canonical bytes·parser·역사 기록은 그대로 유지한다. 최소 두 개의
선언 arm이 필요하며 실제 전제가 없는 arm은 명시적인 unsupported/미해결이다.
원 shared credit·deadline·실제 비용·같은 actor의 release/cache ACK·PinZero·durable
publication을 보존한다. 선언이나 역직렬화 자체는 실행 권한을 발급하지 않는다.

남은 병렬 작업은 `/2`의 Search·manager·Arena·엄격 FD reader 연결과 실제 독립
trial 인수, 네 모드의 실제 UCI 및 외부 교사 연결, 현재 source/binary/CI 확인이다.
로컬 RTX 4050 수치·물리 수명·정상 종료는 별도로 진행하며 성능 자료로 사용하지 않는다.
실제 학습·대규모 수집·원격 성능/기력 비교·PR 병합은 이번 목표에서 제외한다.

## 열세 번째 통합 체크포인트

`5341d90`의 정식 미학습 G0와 원 P/C export를 로컬 RTX 4050에서 검사했다.
새 CUDA bindings에 맞춘 CPU Torch reference와 실제 Rust·ORT CUDA raw를 각각
생성했다. 기존 numerical validator가 Fresh B1·FP32·TF32 off의 P/C 12그룹·100사례,
입력 tensor 974,988 bytes, 실제 native NN 입력 완료 200개를 대조해 통과했다.
최대 절대 차이는 logits `1.9073486328125e-6`, policy `1.874199417883915e-8`,
WDL `1.1690709406098065e-7`이다. 기존 절대·상대 허용 오차를 바꾸지 않았다.

원 GPU 수집의 native worker join과 외부 process exit 0을 별도 기록으로 보존했다.
저장된 raw를 다시 검증하는 CLI는 이 과거 자료의 수치만 판정하며, 현재 프로세스의
물리 완료나 실행 권한을 새로 발급하지 않는다. 모델·runtime source 다섯 component의
원 식별을 보존하며 이후 다른 통합 SHA에 자동으로 소급하지 않는다. optimizer update와
backward 실행은 0회다. 이 결과는 GPU tensor 수치·기본 수명 검사이며 네 모드의 전체
UCI, 실제 conditional witness, 외부 교사, 대국·성능·기력·학습 인수를 대신하지 않는다.

첫 saved-scope 조립은 원 CUDA kit 소유자와 root 실행자의 UID 차이를 거부해 실패했다.
원 실패를 보존하고, 후속 조립은 명시한 원 소유자 UID와 같은 장치·retained FD·정규
파일·원 SHA를 확인했다. CUDA DSO만 원 inode의 읽기 전용 hardlink로 공유하고 원
checkpoint·모델·raw는 각각 복사했다. 원 파일의 내용·권한·소유자는 바꾸지 않았다.
이 saved-scope는 startup의 `owned_files` 요건을 만족하는 실행 자산으로 사용하지 않는다.

`5bc1d09`의 순수 명세 검사는 두 suite에서 각각 5개 통과했다. Windows에서 Linux 전용
arena metadata·wire fixture 필터는 0개였으므로 해당 검사를 실행했다고 표시하지 않는다.
같은 SHA의 Linux 대상 workspace 전체 feature 타입 검사는 통과했다. CI의 Linux·Windows·
bindings 세 job은 `rustfmt` 불일치에서 중단됐으며, `2be377f`가 그 형식을 수정했다.

`2be377f`에는 원 machine 소유의 중립 custody와 별도 개입 arm, 엄격한 Python `/2`
reader를 연결했다. 신경망 import를 차단한 자료 검사 44개가 통과했으나 실제 witness의
양성 효용 검사는 아니다. 첫 Rust 검사048은 계약·Store의 서로 다른 `Move16` 혼용으로
컴파일에 실패했다. `67177e2`에서 명시 변환으로 수정했으며 후속 실제 검사로 확인한다.

남은 연결은 `/2`의 UCI·Arena 실행과 원 Store/Rules 근거를 재사용하는 strict historical
join, 같은 고정 소스의 컴파일·실행·source inventory, 네 모드 및 외부 교사의 실제 인수다.
서로 다른 trial의 원 cost epoch·authority·clock을 유지하며 공통 비교 맥락으로 위장하지
않는다. 준비 명세는 계속 실행 권한 `false`, optimizer update `0`인 수동 자료다.

### N05 후속 통합 계약

실제 producer는 길이 한도 또는 Rules 종료에 도달한 Proposal을 게시한다. 따라서
그 full target을 단순 연장하는 최초 P arm 정의는 정상 자료에서 항상 unsupported가
될 수 있다. `/2`의 P는 같은 original first에서 제한된 재제안으로 구체화한다.
중립 full Proposal·record·reference bytes는 그대로 보존하고, P의 실제 query prefix만
원 Rules root에서 재생한 `[original first]`로 둔다. query proposal은 원 full target을
유지한다. C prefix는 빈 수열, R prefix는 원 full target이다. 실제 first endpoint의
종료·합법수·기존 길이/credit/deadline 검사를 유지하고, target을 잘라 넣지 않는다.
기본 탐색 정책은 이 실험 정책을 선택하지 않으면 바뀌지 않는다.

비교의 공통 조건에는 실제 source FrameBudget의 role·CPU·Store 세 축과, 원 witness
의미 식별 집합도 포함한다. 이 witness 식별은 정확한 Rules root·Counter/Repair 종류·
Move16 라인을 사용하며 owner·frame·request·절대 시각을 공통 의미로 비교하지 않는다.
실제 arm 발주 전과 pair 인수에서 두 조건을 모두 대조한다.

새 `/2` dataset을 실제 학습 소비자로 연결하기 위해 별도 `intervention_v2` source family를
구현한다. 기존 recipe·parameter·NNUEC 수학은 유지하고, 원 exact9 준비 본문을 기존
PREP1/PREP2 외피로 재생성하지 않는다. dataset·context·pool은 각각 기존 도메인의 `/2`,
training-registration과 train-window는 `/3`, startup/RPC/ACK/result 및 checkpoint·
retirement·projection·closure·completion·export wrapper는 별도 `/2`다. 구체적인
문자열은 각 기존 도메인의 실제 이름을 사용한다. registration은 기존 29필드에
`data_family`를 더한 30필드, window는 새 schema의 기존 23필드, child startup은
기존 18필드에 같은 family를 더한 19필드로 닫는다. 기존 parser의 허용 범위는 넓히지 않는다.

`ValidatedUtilityNumericalParts`와 private typed family를 통해 검증된 수학 부분과
감독·FD·resource·socket·ACK·atomic 저장 기계만 공유한다. 새 원 dataset/authority가
현재 peer·lease·읽기 예산·원 update journal·completion을 별도로 보유해야 한다.
준비 metadata나 저장된 역사 자료만으로 실제 update·checkpoint·export 성공을
만들지 않는다. 기존 diagnostic 자산은 승격하지 않는다. 이 연결은 소스 구현과
비학습 검사의 범위이며 실제 optimizer update·학습을 실행하는 승인은 아니다.

## 열네 번째 통합 체크포인트

`d0734df`의 Windows 등록 검사에서는 Linux 대상 workspace 전체 feature·전체 target
타입 검사, exact9 metadata 7개, UCI intervention 3개가 통과했다. 별도로 neural import를
차단한 Python metadata·dataset·checkpoint 검사 76개가 통과했으며 실제 update는 0이다.
이 결과를 Linux FD나 실제 trainer child의 실행 근거로 대신하지 않는다.

같은 SHA의 원격 bindings·모델 CI는 기본 UCI feature에서 선택 의존성 `serde_json`을
노출한 반환 타입 때문에 실패했다. `137718b`에서 메서드 전체를 실제 생산 caller와 같은
feature 경계로 제한했다. `932d7ae`의 실제 Linux 기본 UCI 빌드와 workspace 전체 feature·
전체 target 빌드가 통과해 복구를 확인했다. 새 declaration-only startup metadata example은
기존 canonical codec을 사용하며 8개 검사가 통과했다. referenced asset이나 provider를
열지 않으며 lock join·실제 startup·현재 실행 권한은 별도로 남긴다.

`932d7ae`의 Linux intervention 검사 18개와 family 검사 4개가 통과했다. 원 FD 보존,
symlink 거부, 별도 publication/ACK, 원 trial의 독립 시계와 구형 domain 거부를 포함한다.
두 suite에 겹치는 항목이 있어 합산하지 않는다. 명시적으로 실행한 wire fixture는 별도
1개 통과했으며, Rust가 저장한 exact9 원문 5,473 bytes를 Python의 기존 validator와
numerical-parts validator가 그대로 인수했다. 이 synthetic metadata는 실제 cohort나
known target, 실행 권한을 만들지 않는다. source inventory와 embedded source 대조도
1개 통과했다. 마지막 엄격 Clippy는 큰 enum·복잡한 반환 타입 두 곳에서 중단됐으며,
원 owner를 보존하는 Box·타입 별칭 수정 뒤 새 소스로 재검사한다.

기존 G0 GPU 원자료는 별도 검증 후 회수했다. CUDA DSO는 중복 복사하지 않고 원
Linux 위치·inode·hash로 참조하며, 모델은 먼저 회수한 CPU 자료를 참조한다. 따라서
회수본은 부분 scope로 표시하고 완전한 이동식 실행 묶음이라고 보고하지 않는다.
`5341d90` 이후 다섯 모델 component 중 encoding/runtime/reference/export는 같고,
model-native에는 `/2` UCI 수명 소비자가 추가됐다. 과거 수치 근거는 원 source에
그대로 두며 현재 전체 native 실행으로 재태깅하지 않는다.

잔여 감사에서 R 없는 P-C의 NNUEC 공통 문맥 판독이 source 미구현임을 확인했다.
기존 capability 거부를 무조건 제거하지 않고, 원 NNUEC source·same-owner preparation
actor·선택된 view에 묶인 판독·terminal/cache-release ACK를 R-free 소비자에 연결한다.
R worker·TT·PVS·qsearch·SEE·native node ledger는 생성하지 않으며, 판독 값을 P/C
신경망 WDL이나 Rules mate 증명으로 바꾸지 않는다. baseline의 명시적인 `None` 경로도
보존한다. 그 뒤 네 모드 UCI·실제 외부 엔진·독립 cohort·현재 trainer의 비학습 수명과
최종 CI를 각각 인수한다. 실제 학습·대규모 수집·원격 비교·병합은 계속 제외한다.


## 열다섯 번째 통합 체크포인트

N05는 사용자가 선택한 별도 `FrozenProposalReviewV1` 개입 정책을 유지한다.
기본 탐색의 Ready offer를 조작하지 않고 독립 root에서 선택한 P/C/R의 실제
질문을 발주한다. 원 authority·clock·cost epoch·종료 근거를 각각 보존한다.
원 publication·durable ACK와 strict `/2` 소비, 현재 trainer child 종료를
실제로 확인하기 전에는 N05 전체 완료로 표시하지 않는다.

`47b6020`에서 R 없는 P-C에 선택형 공통 NNUEC 문맥 판독을 연결했다. 원 model
owner와 PreparationActor를 재사용하고 질문별 Rules/view/budget/deadline을
봉인한다. 원 terminal·cache-release ACK를 회수한 root closure를 요구하며
늦은 완료·취소·완료 불명에서 원 model/source/buffer 소유권을 유지한다.
R worker·TT·PVS·qsearch·SEE나 R node ledger는 생성하지 않는다. 판독 값은
P/C의 policy·WDL·Rules 메이트 증명·utility 정답으로 바꾸지 않는다.
기존 명시적 `None` 기준 경로도 보존한다.

첫 공유 검사에서는 fixture의 node 한도 64가 기존 semantic 최소값보다 작아
시작 전에 거부됐다. 원 실패를 남기고 fixture만 257로 수정했다. 제품의 최소
한도는 유지했다. `4e12e2f`의 실제 Linux 공유 검사 12개와 `47b6020`의 전체
feature·전체 target 컴파일 및 엄격 Clippy가 통과했다. 겹치는 검사를 합산하지 않는다.

`961c59b`의 선택형 position metadata는 기존 UCI parser와 자체 Rules를 사용해
실제 상태·이력 digest와 Rules profile digest를 읽는다. FEN의 미확인 과거를
유지하고 새 root 권한·시계·session을 만들지 않는다. 인수가 없으면 기존
metadata/1의 필드를 보존한다. 실제 Linux 검사 12개, 자원 선언 검사 2개와
전체 workspace 엄격 Clippy가 통과했다.

`09e591b`는 별도로 생성되는 shared readout helper를 manager와 같은 물리
pool로 선언하지 못하게 했다. 완료·취소·실패·만료 terminal 수와 실제 semantic
소비 수를 별도 getter로 관측하며 제출 수로부터 추정하지 않는다. 실제 Linux에서
공유 12개, 자원 선언 3개, configuration 14개, R 없는 manager 9개와 embedded
source inventory 1개 검사가 통과했다. thread affinity와 물리 pool sharing은
소스나 Rust ThreadId만으로 확인됐다고 표시하지 않는다.

`dfad001`의 별도 비학습 CPU 검사에서 metadata 소비자 139개와 실제 tensor
상태·checkpoint 복원 29개가 통과했다. optimizer step·backward·autograd 호출을
차단했으며 차단 시도와 실제 update는 모두 0이었다. 이 결과는 실제 현재
parent/child IPC와 원 end receipt의 인수가 아니다. 후속은 기존 window/3의
명시적인 `max_optimizer_updates=0`으로 begin/end와 physical fence를 검사한다.
optimizer를 만들 수 있지만 sample·forward·backward·step과 checkpoint 저장에는
진입하지 않으며 `checkpoint=null`인 no-update 종료를 요구한다.

공식 Stockfish 19·Reckless 0.9.0의 배포물·binary SHA를 검증한 뒤 실제 UCI
식별·readiness·정상 quit을 확인했다. 아직 실제 go/stop/drain·피드백·교사 target·
대국 근거로 사용하지 않는다. 모델·NNUEC·runtime 원 자산은 제품의 inputs
namespace와 읽기 전용 단일 소유 파일 조건으로 준비한다. CUDA 라이브러리는
네 모드마다 복제하지 않고 한 벌을 순차 재사용한다. 준비의 pin/hash 확인은
provider나 현재 실행의 물리 완료 증거를 발급하지 않는다.

남은 실제 인수는 네 모드 UCI·공유 NNUEC, 외부 두 엔진의 탐색·취소·피드백,
독립 intervention cohort 게시·재소비, 실제 trainer child의 zero-update 종료다.
로컬 RTX 4050 자료는 기능·수치 검증에만 사용한다. 실제 학습·대규모 수집·
원격 비교·병합은 계속 제외한다.

## 열여섯 번째 통합 체크포인트

N05의 별도 개입 정책은 원 Ready offer를 만들거나 바꾸는 경로와 분리한다.
같은 중립 Proposal을 대상으로 독립 root의 P 재제안·C 반례·R 검토를 실제
발주하고, 각 arm의 권한·시계·cost epoch·물리 종료를 원 기록으로 보존한다.
원 publication과 별도 durable ACK, strict 역사 소비 및 현재 부모·자식의
zero-update 종료를 인수하기 전에는 이 기능 전체를 완료로 표시하지 않는다.

`23017c4`의 실제 Linux 형식 검사·자원 identity 검사 5개·workspace 전체
feature/target 엄격 Clippy와 실행 파일 8개 빌드가 통과했다. 이후 실제
CPU P-C 시작에서 자원 영수증의 source/domain 혼용을 발견했다. `3fb36db`는
로컬 readback schema를 보존하고 ArtifactRef의 공개 producer URL을 분리했다.
실제 자원 evidence writer 검사 3개, Rules metadata 검사 2개, 전체 엄격
Clippy와 실행 파일 9개 빌드가 통과했다. 파일 캐시·RSS·VRAM peak 미관측은
unknown으로 남기며 이 빌드를 실제 네 모드 실행 성공으로 승격하지 않는다.

별도 실행에서 공식 Stockfish 19와 Reckless 0.9.0 각각 정상 완료·진행 후
취소·진행 후 Yield를 검사했다. `23017c4`의 원 외부 probe 6개가 모두
통과했고 실제 worker join·terminal drain을 확인했다. 이는 외부 프로세스
단독 수명 검사이며 P/C의 교사 응답 소비·실제 대국·기력·학습 인수와 다르다.

첫 native 시작의 self-PID와 trainer direct-child 혼용은 `23017c4`에서
구분했다. 이후 자원 영수증 오류, 준비 명세의 question/frame 한도 불일치,
서비스 실행 환경의 cache root 누락도 원 실패와 구분해 보존했다. 한도나
출처 guard를 완화하지 않았고 준비 선언만 정식 소비 조건에 맞게 고쳤다.
CPU P-C는 그 뒤 실제 모델 로딩·uciok·readyok까지 도달했다.

첫 depth root는 반복 재검토 정책을 원 Frontier 연결 전에 적용해 실패했다.
`95cd9d2`는 Frontier 연결 후 정책을 적용하고 실제 readback을 대조하도록
고쳤다. 첫 역할 발주 이전이라는 조건과 실패 port custody를 유지한다.
closed 세션의 다음 Start는 admission에서 거부하고 원 물리 Unknown·실제
actor/native join 조건은 바꾸지 않는다. 실패 실행에서 나온 fallback을
정상 depth 완료로 보지 않으며 이후 native/GPU 실행은 새 source로 재검증한다.

`23017c4`의 원격 CI에서는 외부 worker fixture 4개와 Python 실행 차단
검사가 실패했다. fixture는 비용 청구 없는 hot 삽입 대신 실제 Rules
frontier 경로를 사용하도록 고쳤고, tensor 경계는 owner·경로·family를
읽기 전에 literal execute 승인을 검사하도록 복구했다. `95cd9d2`의 실제
순수 Python 검사 23개가 통과했다. 이어 새 Rust 회귀의 비공개 ID import로
컴파일이 실패했으며 `7e01bdd`에서 공개 계약 타입으로 수정했다. 이 단계의
소스 수정·집중 검사와 아직 완료되지 않은 최신 전체 CI를 구분한다.

`7e01bdd`에서는 Python 23개, P-C 정책 순서 회귀 1개, 외부 프로필 5개가
실제로 통과했다. 외부 worker 4개 재검사는 1개 통과 뒤 raw PV의 Store
projection 불일치로 실패했다. producer가 Discovery에서만 projection을
만들고 Store는 다른 질문에서도 이를 요구하는 더 깊은 제품 접점을 확인했다.
`beaf652`는 원 질문·Rules·quota로 필요한 projection을 만들고, 한도를 넘으면
명시적인 오류와 원 report를 보존하도록 고쳤다. validation을 Discovery로
재태깅하거나 Store의 PV 검증을 완화하지 않는다. 새 fixture의 실행 인수와
ticket 상태 기대 오류도 원 실패와 나누어 수정했다. `b1d465f`에서 외부 worker
집중 검사 5개가 실제로 통과했다. 정상 validation·peer R의 PV 게시, horizon·
node 한도 부족의 원 packet 보존, 종료 이후 재시도·퇴역 수명을 포함한다.

CPU P-C 실행 016은 실제 준비 이후 첫 go에서 물리 Unknown으로 중단됐다.
공유 NNUEC의 원 root budget을 검증하기 전에 Frontier를 retire하는 순서
오류를 확인했다. `b1d465f`는 manager와 driver의 원 영수증 검증을 먼저 하고
Repair report·Frontier 회수·semantic owner 복귀를 뒤에 수행한다. handoff가
실패해도 원 typed closure를 native owner와 함께 보존한다. Unknown의 코드·
분류·실제 join 조건은 유지하고 실패 단계의 진단만 구체화했다. 원 실행 016의
Unknown과 OS 종료·EOF는 재분류하지 않으며 새 실행에서 실제 복귀를 검증한다.

`b1d465f`의 형식 검사, 순수 Python 23개, P-C 정책 순서 1개, 외부 프로필
5개, 위 외부 worker 5개, workspace 전체 feature/target 엄격 Clippy와 실행
파일 10개 빌드가 통과했다. 별도 정확한 모듈 필터의 R-free 목표·종료 검사
2개도 통과했다. 0개가 실행된 보조 명령은 인수 수에 넣지 않는다. 이 결과는
CPU/source 검사이며 네 모드의 실제 NN·물리 종료 인수와 구분한다.

신규 cohort의 준비 접점으로 기존 Rules capture/checked restore를 재사용하는
replay metadata와, 실제 pin된 teacher profile에서 신규 recipe actor 선언을
내보내는 metadata를 추가했다. 후자는 source/options/assets/rights 선언을
별도로 봉인하며 embedded 모델의 unknown 권리를 확인됐다고 표시하지 않는다.
모든 준비 metadata는 실행·Ready·현재 root 권한·학습 완료를 발급하지 않는다.
실제 Rust 예제로 시작 위치 replay와 두 teacher actor 선언을 추출했다.
후속 membership 준비는 원 G0 inventory·원 model metadata·NNUEC raw/semantic
pin과 현재 source를 각각 대조한다. 이 metadata 조립은 tensor 로딩·역사 실행
재인증이나 실제 cohort 자료 게시가 아니다.

실행 019의 CPU P-C는 UCI·메이트·stop·ponderhit·newgame·quit과 OS 종료를
관측했지만 첫 depth 1의 `completed_depth=None`이므로 전체 기능 인수가 아니다.
`e6cd269`는 실제 shutdown closure를 확인한 뒤 원 manager 실패를 보고하도록
고쳤다. 실행 023은 `original P/C frontier rejected goal work`와 Pending 목표를
보존했다. 선택적 Repair가 원 깊이를 넘어 발주되는 원인을 추적하며, 정상
fallback이나 단순 bestmove를 완료 깊이로 재표시하지 않는다.

실행 019의 CPU P-C-R은 첫 go에서 실제 stack overflow와 OS 종료 -6을 남겼다.
원 ELF의 `run_admitted` prologue는 약 1.79MB의 정적 frame을 예약했다. 이는
런타임 stack peak 관측값이 아니다. `f6d8bc5`는 같은 heap custody의 네 단계로
큰 전이를 나누고 실패·원 receipt·물리 owner를 계속 보존한다. stack 크기나
root 권한·물리 완료 조건은 바꾸지 않았으며 새 실제 실행을 별도로 인수한다.

cohort source-only 준비 020과 membership 021은 실제 원 metadata로 통과했다.
원 G0 inventory는 공유 64·P 19·C 21·V 20의 124개이며 V가 없었다고 표시하지
않는다. 신규 선택 membership은 V-free P/C export에 해당하는 104개와 NNUEC
24개를 선택하고, 원 V 20개 descriptor를 제외 근거와 함께 보존한다. 이는
실제 tensor 로딩·publication ACK·학습 또는 현재 child IPC 인수를 뜻하지 않는다.

잔여 실제 인수는 현재 source의 네 모드 UCI·NNUEC 소비·물리 종료,
외부 응답 downstream·독립 cohort 원 자료 게시/재소비, 현재 trainer의
zero-update 부모·자식 종료 및 최종 CI·증거 대조다. 로컬 GPU 결과는 계속
기능·정확성 검증에 한정하고 실제 학습·원격 비교·병합은 실행하지 않는다.

## 열일곱 번째 통합 체크포인트

`bb7543b`는 P-C의 선택적 Repair 끝을 원 go의 실제 horizon으로 제한한다.
정책 자체를 끄거나 Frontier의 거부를 완화하지 않는다. 같은 소스의 실제
CPU P-C 실행 029는 깊이 1의 원 완료 범위와 메이트, movetime, stop·중복 stop,
ponderhit·ponder miss의 원/새 root, newgame·quit을 관측했다. 일곱 착수와
정상 OS 종료·EOF를 확인했고 두 depth root는 `completed_depth=Some(1)`,
native R·qsearch 사용량 0을 기록했다. 이는 고정 제품 소스의 원 closure·join
검사와 실제 transcript를 함께 대조한 기능 근거다. 일반 UCI가 전체 formal
RootReceipt를 파일로 수출했다거나 별도 GPU peak를 관측했다는 뜻은 아니다.

CPU P-C-R 실행 029는 첫 go에서 여전히 stack overflow로 종료됐다. 실제
GDB 진단 036은 정상 퇴역 중 `PreparationActor::try_retire_engine`을 호출하는
`NnuecRootRetirement::step`에서 실패했음을 확인했다. `2fee388`은 원 퇴역
소유 객체를 동일한 heap allocation에 두고 아홉 단계를 개별 함수로 나눈다.
Busy·오류는 같은 engine을 원 슬롯에 복귀시키며 engine ACK → cache ACK →
원 pin 해제 → 실제 Store pin zero → utility 최종 소비 → preparing owner
복귀 순서를 보존한다. stack 증설이나 물리 완료 조건 변경은 하지 않는다.

실제 build 037에서 형식 검사, UCI 전체 feature 라이브러리 검사 785개,
workspace 전체 feature/target 엄격 Clippy와 실행물 10개 빌드가 통과했다.
재사용한 search/Python 검사는 원 build 025의 source와 범위를 따로 기록했다.
파일·디렉터리 동기화와 원 ELF hash readback 뒤 새 실행물 038을 등록했다.
정적 진단의 개별 함수 prologue는 보조 원인 자료이며 런타임 peak·처리량
개선이나 기능 인수의 대체물이 아니다.

새 CPU P-C/P-C-R 실행 038은 각각 일곱 착수와 다음 root·newgame·quit,
정상 반환을 관측했고 각 원 stream과 자원 admission/finished 파일을 회수했다.
P-C-R의 movetime root에서는 실제 native R와 P/C 출력 소비도 관측했다.
그러나 P-C-R depth root의 완료 깊이는 여전히 `None`이다. 원 retired
Frontier의 범위를 그대로 반환하므로 이를 단순 진단 누락이나 전체 depth
성공으로 표시하지 않는다. 일반 candidate continuation도 원 go horizon을
존중하도록 수정하고, 원 물리 closure 이후 manager의 의미 실패를 보고하는
접점을 보완한다. owner가 발급한 실제 goal endpoint만 역할별 Fresh로
넘기며 임의 frame·0-depth R·강제 완료 깊이를 발급하지 않는다.

`bb7543b` CI의 Linux 최초 실패는 대형 fixture ELF의 재해시를 2초 시작
창에 넣은 데서 발생했고 후속 20개는 공유 잠금 poison에 의한 연쇄였다.
Windows는 Linux 전용 import와 필드 사용 경계의 경고 두 건이었다.
`61c0b31`은 테스트의 사전 identity 준비와 실제 시작 시계를 구분하고 기존
공통 handshake 창을 사용하며 OS 전용 경고를 좁게 처리한다. 제품의 hash·
deadline·stop 검사는 바꾸지 않는다. 원 CI 실패와 새 CI의 실제 완료 상태를
구분하며 이 수정 커밋 자체로 새 CI 성공을 선언하지 않는다.

현재 소스의 source-only 준비 038, membership 039와 series/Collection 준비
040은 실제로 통과했다. source aggregate·원 lock·NNUEC·P/C 의미와 실제
membership을 조인했고 metadata 단계에서는 NN import를 차단했다. 이들은
Ready·root·물리 완료·실제 collection 권한을 발급하지 않는다.

별도 `FrozenProposalReviewV1`의 첫 실제 발주 044는 원 R 수집 observer를
연결하는 검사에서 거부됐다. 원 오류는 `native R collection requires
original idle NNUEC Collection root`이며 상위 원 결과의 물리 완료는
Unknown이다. 최초 arm의 자원 admission과 원 stderr·실행 자료를 보존했고
다음 arm·추가 NN/GPU 실행을 시작하지 않았다. 원 프로세스 부재는 확인했지만
이를 누락된 제품 물리 종료 증거로 대신하지 않는다. Collection의 외부
CPU_T와 native CPU_R가 독립적으로 연결되는 조건을 검토·수정한다.

후속 strict historical reader는 기존 제품 API만 사용하는 외부 검증 caller로
실제 빌드했다. 원 publication·별도 ACK가 아직 없으므로 역사 소비 성공이나
semantic dataset SHA를 발급하지 않았다. 별도 utility 초기 자산 041도 원
NNUEC base에 대해 공식 scalar decoder로 검증한 미학습 네 배열이다.
모델 forward·optimizer·학습·기존 diagnostic eligibility 전환은 실행하지
않았으며 이 자산 준비를 현재 부모·자식 IPC 인수로 표시하지 않는다.

남은 순서는 P-C-R의 원 목표·Collection 연결 수정 → 네 모드 기능 인수 →
실제 독립 arm·publication/ACK·원 파일 strict 소비 → 실제 dataset의
zero-update 부모·자식 종료 → 최신 CI와 필수 31항목 대조다. 조건부 O01~O04,
실제 학습·대규모 수집·원격 비용/기력 비교·병합은 별도로 남긴다.

## 18차 체크포인트: 독립 cohort 개입과 반례별 목표 책임

사용자 선택에 따라 N05의 선택형 비교는 기존 Ready offer만 기다리는 경로와
별도로 `FrozenProposalReviewV1` 개입 정책을 사용한다. 같은 입력·모델·예산과
중립 target을 봉인한 뒤 독립 root에서 P/C/R 중 하나를 실제 발주한다. P는
원 first move prefix와 전체 중립 proposal을, C는 빈 prefix와 같은 proposal을,
R은 원 target 전체를 사용한다. 각 arm의 실제 authority·clock·cost·종료 근거는
각각 보존한다. 등록 epoch는 첫 trial의 anchor이며 다른 trial 시계를 합치지
않는다. 기본 탐색과 기존 Ready 기반 형식은 변경하지 않는다.

실제 build 045의 UCI 검사는 787개 통과·1개 실패였다. 진단을 추가한
`69db350`의 집중 실행 050은 깊이 1을 통과했지만 깊이 2에서 원
`Frontier(Duplicate)`를 재현했다. 원 snapshot은 책임 두 개의 완료와 최대
조사 ply 2를 기록했으나 선택 범위는 아직 seal되지 않았고 완료 깊이는
`None`이었다. 원 물리 완료가 Confirmed라는 사실을 의미 목표의 성공으로
대체하지 않는다. 실패 자료와 원 044의 PhysicalUnknown은 그대로 보존한다.

소스 추적에서는 같은 coordinator frame의 다음 queued counter가 첫 counter의
완료된 책임을 다시 조회하는 경로를 확인했다. 후속 수정은 실제 선택된
counter마다 원 질문·Rules prefix·first move·deadline·budget에 묶인 책임을
별도로 보존한다. 기존 일반 등록과 exact retry의 중복 거부는 유지한다.
이전 counter의 미완료 horizon도 삭제하거나 완료로 바꾸지 않으며, 원
controller가 계속 소유한다. 깊이 완료는 실제 책임과 정책 seal의 결과로만
반환한다. 이 문서 갱신 자체는 수정 후 검사나 제품 실행 성공이 아니다.

실제 cohort의 세 arm이 발급·퇴역되고 publication·별도 ACK·strict 소비가
통과했더라도 새 유효 witness가 없으면 선호 action은 `unknown/masked`다.
비용 차이만으로 utility 정답을 만들지 않는다. known pair 0인 실제 자료도
원 무결성 조건을 만족하면 학습 0회 IPC 검증에 사용할 수 있다. 반면 실제
arm/query 발급 실패는 별도 실패이며, 자료 보존을 dataset 인수로 표시하지
않는다. 학습 0회 창은 실제 모델·AdamW owner 초기화와 begin/end·fence·drain을
검증하지만 sample·forward·backward·optimizer step·checkpoint·export를
발행하지 않는다. 이를 학습 완료나 유효 학습 target 확보로 보고하지 않는다.

후속 실제 build 051은 깊이 1·2와 서로 다른 Counter 책임 세 개의 등록·완료를
통과했지만, 추가 chunk 2·depth 3 사례가 Capacity로 실패했다. 조사한 fixture는
legacy 생성자로 root prefix도 2에 묶은 채 Frontier 목표 8을 등록했다. 실제
제품과 같은 ResearcherSemanticProfile로 이미 선언한 목표 8을 연결하고 chunk·
노드·역할·시간 예산과 제품 입력 제한은 유지한다. 원 실패는 보존한다.

같은 frame에서 두 실제 nonzero Repair도 각각 원 책임을 갖도록 선택 source를
확장한다. 실제 P 착수와 Rules 전이가 완료된 생산 지점에서만 새 Repair 책임을
발급한다. zero-step은 원 Counter·Repair record에 묶인 관찰 전용 상태이며 새
P 출력이나 책임을 만들지 않는다. CPU 재검증의 실제 feedback 뒤 현재 예산을
다시 봉인하고, Ready 단계와 실제 Fresh 제출 한 번의 차감을 구분한다. 이전
책임의 history·미완료 horizon과 일반 중복 거부는 유지한다. 새 혼합 관찰 회귀와
두 nonzero Repair 회귀는 실제 후속 검사에서 인수하며 소스 검토로 대신하지 않는다.

## 19차 체크포인트: 현재 수치 대조와 Collection의 R 등록

고정 소스 `b5860ad`의 실제 Search 검사는 1,062개 통과·2개 ignored였으며,
workspace 전체 feature/target 엄격 Clippy도 통과했다. 실제 build 091의 UCI
검사 789개와 inventory 검사 1개, 새 실행물 10개·검증 caller 2개도 통과했다.
Search와 Clippy는 같은 소스의 원 실행 증거를 명시적으로 재사용했으며,
build 091에서 다시 실행한 것으로 합산하지 않는다.

등록 092에서는 NNUEC의 구현 식별이 변경됐으나 원 가중치·구성·인코딩·의미·
namespace는 유지됨을 원문으로 대조했다. 구현 식별까지 같다고 표시하지
않았다. 현재 원 바이너리의 수치 검사 101은 기존 20개 입력을 실제로 다시
실행하고, Rust가 출력한 reference input을 독립 scalar FP32 reference에
전달했다. raw 출력과 WDL의 최대 절대 차이는 각각 0이었다. 이는 CPU 수치
근거이며 실제 Collection·물리 퇴역·GPU·성능·학습 인수를 대신하지 않는다.

CPU P-C 실행 092는 여덟 요청과 정상 종료를 관측했다. P-C-R은 첫 깊이 1·2의
완료와 여덟 착수를 관측했지만, 같은 게임의 raw public record 상한에서
`PinSaturated`로 끝났으므로 전체 기능 통과가 아니다. 이 구성은 archive를
사용하지 않으며 raw 상한은 128, selected review 상한은 64다. 새 root는
원 증거를 삭제하지 않으므로 여러 `go`가 같은 raw 한도를 공유한다.

후속 처리는 이 정확한 public-record producer만 원 controller의 private
원인을 남기고 `Stopped(Capacity)`로 닫는다. 같은 root의 그 원인과 typed
오류를 확인한 경우에만 기존 quiesce·ACK·credit·물리 drain을 수행한다.
발행 거절 전에 실제 소비한 ranking은 한 번만 정산하며, 거절을 이유로 그 소비를
지우거나 완료 depth를 만들지 않는다. 다른 capacity,
pin·archive·물리 오류는 기존 실패 경로에 남는다. 이 수정은 자료 회수나
archive 연결이 아니며, 실제 새 게임 reset 전에는 다음 root도 Capacity일 수 있다.

source 093·membership 094·series 095 재개는 NN·optimizer 실행 없이 실제
metadata를 연결했다. 실제 cohort 096은 첫 arm의 NNUEC 등록에서 거부됐고,
원 제품 결과의 물리 완료는 Unknown이다. 다음 arm은 실행하지 않았으며 원
stderr·자원 admission/finished 자료를 회수했다. OS 종료·EOF를 누락된 원
물리 완료 증거로 바꾸지 않는다. 유효 dataset·utility 0·publication ACK를
만들었다고 표시하지 않는다.

원인은 Collection에서 NNUEC R 등록이 `cpu_t`를 검사한 분기였다. 해당 제품은
`cpu_r=Own`, `cpu_t=Disabled`이며 Arena가 외부 교사를 따로 소유한다. 후속
수정은 NNUEC 등록도 실제 `cpu_r`를 검사하도록 맞추고 원 모델·root budget·
idle·중복·pending 검증은 유지한다. 기존 실제 owner fixture에서 Deployment와
Collection, 잘못된 R endpoint 및 중복 선언을 각각 검사한다. 이 소스 변경
자체는 수정 후 실제 cohort의 성공 근거가 아니며 새 빌드와 실행으로 인수한다.

남은 인수는 수정 후 네 모드·실제 독립 arm과 publication/별도 ACK/strict
소비·업데이트 0회 IPC·일반 외부 교사 및 외부 R 연결·최종 CI와 자료 회수다.
실제 학습·대규모 수집·원격 성능/기력 비교·PR 병합은 이번 목표에서 실행하지 않는다.

## 20차 체크포인트: Collection의 선택 경계와 용량 오류 보존

Collection에서 원 NNUEC R을 등록한 뒤 driver가 다시 teacher endpoint를 고르던
두 번째 분기도 수정했다. manager가 실제 실행할 endpoint를 한 번 결정하고
driver는 그 선택을 소비한다. 기존 Deployment와 legacy Collection의 선택
의미를 자동으로 통합하지 않는다.

소스 `adbe248`의 실제 집중 검사 131은 Collection R owner, 같은 게임의 연속
root에서 raw 기록 보존·Capacity 종료, 다른 Capacity 원인 구분, archive의 새
오류 직렬화 네 검사를 모두 통과했고 managed cleanup을 확인했다. 앞선 도구
세션 중단은 전체 완료로 표시하지 않았다. 중단된 빌드 lease와 scratch는 원
소유권·비활성 상태를 확인해 보존하고, 원 실행 결과는 unknown으로 남겼다.

같은 소스의 전체 build 121은 UCI 791개를 통과했지만 Search는 1,061개 통과,
3개 실패·2개 ignored였다. 최초 실패는 실행 중 record 상한을 낮추고 복구해
같은 root를 계속 사용하는 기존 fixture였다. 새 terminal Capacity 의미와 맞지
않아 정상 경로의 원 packet·중복 발행 검사와 고정 한도의 실제 소진 검사를
분리한다. 나머지 두 실패는 같은 fixture 잠금의 poison이었으며 독립 성공으로
대체하지 않는다. 이 실행은 Clippy·제품 빌드 단계까지 도달하지 않았다.

발행 실패 뒤 raw 관측만 회수하는 호출이 성공해도 원 publication 오류는
사라지지 않아야 한다. 후속 수정은 실제 feedback 발행 완료 전까지 그 원인을
유지하고, 고정 한도 도달·raw-only drain·원 fence 퇴역에서 관측·소비 계수와
오류를 함께 확인한다. 이는 archive 회복이나 검색 재개가 아니며, 수정 후
실제 검사·새 제품 바이너리·cohort 인수는 별도로 남는다.

## 21차 체크포인트: PCR 종료 경합과 review bundle 포화

소스 `315722c`의 전체 build 151은 실제 11단계 모두 통과했다. UCI 794개,
Search 1,066개·ignored 2개, inventory 1개와 전체 feature/target 엄격 Clippy를
확인했고 실행물 10개·검증 caller 2개를 원 depfile·hash와 함께 보존했다.
등록 152는 원 NNUEC의 모델·가중치·구성·인코딩·namespace를 대조했으며 구현
식별의 차이는 그대로 남겼다. 새 바이너리 수치 158의 실제 20 forward는
독립 stdlib FP32 reference와 raw/WDL 최대 절대 차이가 각각 0이었다.

이전 CPU PCR 142의 `UnknownPreparation`은 test-only 소스 `06f35b8`에서
원 worker가 취소된 준비를 퇴역한 뒤 원 queued Prepared를 소비하는 경합으로
재현했다. 실제 142에는 원 이벤트 ID가 없어 세부 분기가 같다고 소급 확정하지
않는다. 후속 수정은 같은 binding의 성공한 취소를 다시 전송하지 않고 Draining의
새 ACK 발주를 닫는다. 이미 큐에 들어간 Accept의 정확한 취소/실제 만료 ACK는
원 ticket·issued seal·output digest와 미수락·미소비를 검사하고 output과 선형
permit을 실제 fence까지 보존한다. 외래 permit·일반 거절·물리 Unknown은
실패로 남는다. focused 159의 수정 후 네 검사는 실제로 통과했고 초기 fixture
컴파일 실패도 별도 자료로 보존했다.

CPU PCR 152는 depth 1/2·Rules mate와 movetime·중복 stop을 지나갔으나 다음
ponder에서 `Review(IndexCapacity)`로 실패했다. 여섯 착수와 exit 0·EOF를
전체 인수로 표시하지 않고 원 자료 다섯 개를 회수했다. 후속 GPU·cohort는
시작하지 않았다. 로그에 bundle ID·개수가 없어 특정 한도 분기를 직접 관측한
것으로 보고하지 않는다.

선택 index에는 raw 128개와 별도로 bundle당 64개 한도가 있으며 기존 compact는
활성 bundle 개수 압력만 처리한다. 같은 anchor·첫 수의 관계 사슬은 raw 한도
이전에 bundle 한도에 도달할 수 있다. 후속 수정은 해당 원 producer만
`BundleRecordCapacity`로 구별하고 현재 frontier의 별도 private 원인과
`PublicReviewIndexCapacity`를 연결한다. 같은 controller와 정확한 typed 오류가
일치할 때만 기존 drain으로 미완료 Capacity 종료를 수행한다. 기본 한도·관계·
증거를 바꾸거나 archive 회복을 구현했다고 표시하지 않으며, 일반 index/view/
sequence·owner·store 오류는 계속 실패한다. 같은 게임의 가득 찬 bundle은 이후에도
Capacity일 수 있다. 이 추가 수정의 집중·전체·실제 제품 검사는 별도로 남는다.

남은 인수는 현재 source의 네 모드, 실제 독립 cohort arm과 publication·별도
ACK·strict 소비, update 0회 parent/child IPC, 일반 두 teacher의 실제 row/head
소비, 외부 R의 counter→Repair→C/Fresh, 최종 CI·자료 회수다. 실제 학습과 원격
성능/기력 비교·병합은 이번 비학습 목표에 포함하지 않는다.

## 22차 체크포인트: 현재 CPU 실행과 Hybrid 초기화 실패 보존

소스 `b010188`의 집중 검사 179는 실제 취소 ACK·원 terminal·capture 퇴역과
review 용량 종료의 아홉 검사를 통과했다. 전체 build 171도 11단계 모두
통과했다. 실제 UCI 796개, Search 1,069개·ignored 2개, inventory 1개와
전체 feature/target 엄격 Clippy를 확인했고 실행물 10개·검증 caller 2개를
보존했다. 별도 최소 feature caller 빌드의 기존 dead-code 경고를 workspace
엄격 Clippy의 결과와 섞지 않는다. 관리 빌드의 임시 영역 정리도 확인했다.

등록 172에서는 원 모델·가중치·인코딩·namespace를 유지하고 NNUEC 구현
식별 차이를 보존했다. 새 바이너리 수치 178의 실제 20 forward는 독립
scalar FP32 reference와 raw 출력·WDL 최대 절대 차이가 각각 0이었다.
CPU P-C와 P-C-R은 각각 등록한 여덟 요청과 stop·ponder·quit 구간을 마쳤고,
원 stdout/stderr·report·admission·finished readback 다섯 개씩을 회수했다.
이는 UCI·OS 종료 범위의 자료이며 별도 제품 receipt 없는 물리 완료나
통제된 성능 비교를 인수한 것으로 표시하지 않는다.

Hybrid P-C 172는 초기 `uci` 응답 전에 60초 제한에 도달했다. 원 quit의
15초 종료 조건도 충족하지 못해 신원을 확인한 원 프로세스에 SIGTERM을
보냈고 OS 종료와 pipe EOF를 관측했다. 원자료 네 개를 회수했으며
finished-live readback은 생성되지 않았다. 원 물리 완료는 unknown이다.
admission은 모델·worker 생성 전의 관측이고 CUDA profile 디렉터리는
생겼지만 placement 파일은 없었다. 이 자료로 CUDA 오류, 특정 loader
병목, NN 실행 여부 또는 시간 부족을 확정하지 않는다. 후속 Hybrid P-C-R과
cohort GPU 실행은 시작하지 않았으며 초기화 경계 진단이 먼저 필요하다.

같은 `b010188`의 원격 Linux·Windows CPU 및 CPU bindings job은 성공했다.
native bundle job은 제품 빌드 전 Python 설치 다운로드의 해시 불일치로
실패했고 원 로그를 보존했다. PALS model CI는 감사 fixture의 engine source
핀 변경으로 실패했다. 원 고정 Git blob 14개를 대조한 결과 engine의
`BundleRecordCapacity` 처리 한 줄만 달라졌고 다른 13개는 같았다. 기존 세
역사 bank와 제품 registry는 보존하고, 검토한 고정 LF source의 새 synthetic
bank에 현재 fixture 소비자를 연결했다. 수정 후 관련 세 모듈의 CPU 검사
180은 141개 모두 통과했다. 이 결과는 전체 PALS CI 재통과나 제품 실행 권한,
Rules 증명·학습 target·실제 GPU 인수를 대신하지 않는다.

선택한 별도 cohort `/2 FrozenProposalReviewV1`은 기본 탐색과 Ready `/1`을
보존한다. 원 neutral proposal을 봉인하고 각 독립 root에서 P/C/R 요청과
예산·clock·종료를 따로 보존하며, 미발급 arm을 unknown 또는 utility 0으로
합성하지 않는다. 소스 감사에서 해당 경계의 추가 누락은 찾지 못했으나,
현재 소스의 실제 세 arm·publication·별도 ACK·strict 소비 인수는 아직 남는다.
그 후 update 0회 IPC, 일반 두 teacher와 외부 R 연결, 최종 CI·자료 회수를
진행한다. 실제 학습·원격 성능/기력 비교·병합은 계속 제외한다.

## 23차 체크포인트: 현재 빌드·CPU 재검증과 중단된 Hybrid 원자료

소스 `e3b02d2`의 전체 build 201은 11단계 모두 통과했다. UCI 797개,
Search 1,069개·ignored 2개, compiled inventory 1개와 workspace 전체
feature/target 엄격 Clippy를 확인했으며 실행물 10개와 별도 검증 caller
2개를 보존했다. 제품 실행물은 dev profile이다. 최소 feature caller의
기존 경고, 실제 workspace Clippy 성공, 실제 제품 실행의 인수는 구분한다.
초기화 진단은 명시 옵션에서만 최대 40회 출력 시도를 하며 기본 off다.
원 오류 owner·startup/cleanup·물리 fence·quarantine 조건은 유지한다.
진단 출력 자체로 provider 준비나 물리 완료를 승인하지 않는다.

새 수치 검사 208에서는 원 요청 20개의 Rust forward를 실제 완료하고,
각 Rust reference input에 대한 독립 scalar FP32 결과와 대조했다.
accumulator·hidden·raw 출력·WDL의 최대 절대 차이는 모두 0이었다.
이전 수치 결과를 새 실행으로 재표시하지 않았다. 등록 209는 같은 build
201의 바이너리와 네 모드 startup metadata·NNUEC registration 원시 핀이
등록 202와 같음을 별도로 확인해 208의 결과를 재사용한다. 재사용 확인은
새 forward 실행이나 GPU 인수가 아니다.

CPU P-C-R 202는 depth 1·2와 메이트 응답 뒤 movetime 응답을 관측하지
못했고 외부 실행기가 quit 뒤 SIGTERM으로 종료했다. 원 report와
stdout/stderr·admission을 회수했으며 finished readback은 없었다.
소스 대조에서 외부 10초 응답·15초 quit 관측이 제품의 원 hard deadline
뒤 30초 cleanup보다 먼저 끝날 수 있음을 확인했다. 내부 교착 원인이나
회복을 이 기록만으로 확정하지 않는다. 제품 한도를 유지하고 외부 관측
창만 별도로 등록한 CPU 209에서는 P-C/P-C-R 각각 여덟 go 시나리오와
stop·ponder·quit을 마쳤다. 두 실행 모두 강제 종료 없이 exit 0·pipe EOF를
확인하고 원자료 다섯 개씩을 회수했다. 202의 실패를 지우거나 별도
제품 receipt 없는 물리 완료를 인수한 것으로 표시하지 않는다.

Hybrid P-C 209에서는 시작 진단의 source pin 구간이 약 43.5초였으며,
마지막 관측은 runtime pin 진입이었다. 이 시간은 초기화 실패 위치를
좁히는 진단이며 통제된 성능 자료가 아니다. 실행 중 등록 boot와 이후
WSL boot가 달라졌고 원 실행·유지 프로세스가 종료됐다. 원 stdout/stderr와
admission을 회수했지만 최종 product report·finished readback은 생성되지
않았다. 물리 완료·실제 provider·첫 추론은 unknown으로 남긴다.
가까운 시각 Windows의 자원 부족 이벤트는 전체 commit 약 99.2%와 WSL
프로세스 commit 약 9GiB를 기록했다. 시간상 연관만으로 VM 재시작 원인을
확정하지 않는다. 후속 GPU·cohort는 실행하지 않았다.

소스 감사에서는 정상 cache-hit 완성 경로도 runtime 원 파일 검증,
retained descriptor 재검증, cached 파일 검증을 각각 수행함을 확인했다.
209가 실제로 완료한 pass나 cache 분기는 아직 unknown이다. 검증 guard를
생략하지 않고 같은 소스의 release 실행물을 별도로 등록하는 후속을
준비한다. 다음 로컬 기능 검증은 CPU 2개·memory.high 3GiB/max 4GiB를
새 lock과 실제 readback에 명시한다. 기존 6/12GiB 자료와 섞어 비교하지
않으며 원 실패를 낮은 한도의 성공으로 바꾸지 않는다. Windows commit
여유의 고정 admission gate를 복원하지 않고, Full GC를 제품에 넣거나
전역 WSL 설정을 변경하지 않는다.

같은 `e3b02d2`의 원격 Linux·Windows CPU, CPU bindings, PALS model job은
성공을 관측했다. native validation bundle은 이 체크포인트 조회에서
진행 중이었다. 전체 CI 완료로 표시하지 않는다. cohort의 출처·소유권
metadata 210/211은 준비됐으나 실제 세 arm·publication·별도 ACK·strict
소비, update 0회 parent/child 종료, 일반 teacher·외부 R 인수는 남는다.
실제 학습·원격 성능/기력 비교·병합은 계속 제외한다.

## 24차 체크포인트: release 기능 검사와 own CPU 재검토 분류 누락

문서 HEAD `dbc58aa`의 원격 다섯 job은 모두 성공을 관측했다. Linux·Windows
CPU, CPU bindings, PALS model, native validation bundle의 결과이며, 실제
로컬 제품 실행 인수와는 구분한다. 이전 `e3b02d2` run의 native bundle은
후속 push로 취소됐다는 상태도 보존한다.

실제 제품 소스 `e3b02d2`의 release 빌드 215는 0~8단계 통과 뒤 독립 reader를
빌드하던 9단계에서 유한 창이 만료되어 exit 124로 끝났다. 컴파일 실패로
분류하지 않으며 자식 종료·scratch 정리와 보존된 관리 캐시를 확인했다.
새 빌드 223은 11단계를 모두 통과했다. UCI 797개, Search 1069개/ignored 2개,
inventory 1개와 엄격 workspace Clippy를 실행했고, release 제품 10개와
독립 caller 2개의 원 바이트·의존 파일을 보존했다. 이후 관리 슬롯의 8GiB
상한에 따른 cache reset도 확인했다. 빌드 성공 자체는 제품 실행 권한이 아니다.

등록 224는 새 실행물과 CPU 2개·3GiB/4GiB의 lock·parent readback을 연결한
passive metadata다. 모델·인코딩·NNUEC 의미 조건과 구현 식별을 구분하며
Ready·root·NN 실행 권한을 합성하지 않는다. 수치 검사 230은 caller 번호
치환이 고정 입력 바이트 수까지 변경한 총괄의 오류로 forward 이전에 거부됐다.
원 입력·핀과 실패 기록은 보존했다. 전체 파생 caller의 숫자 AST를 대조해
달라진 상수 두 곳을 복구한 뒤, 새 검사 234에서 실제 Rust forward 20개와
독립 scalar FP32를 대조했다. 옛 수치 결과를 새 실행으로 세지 않는다.

CPU P-C 224는 여덟 go·stop·ponder·newgame·quit와 exit 0·양 pipe EOF를
관측했다. 별도 제품 receipt 없는 물리 완료는 unknown이다. CPU P-C-R은
depth 1/2·mate 뒤 movetime에서 `RecheckCounterValidation` 질문을
`not an open CPU frame`으로 거부했다. 정상 bestmove·exit 0이 있어도 의미
실패로 판정하고 GPU·cohort 후속 실행을 멈췄으며 각 원자료 5개를 회수했다.

원 producer·coordinator·외부 CPU·후속 Fresh 소비는 이미 이 단계를 지원하지만,
own `is_cpu_stage`에만 누락돼 있음을 실제 trace와 두 독립 소스 감사로 확인했다.
해당 단계를 추가하고 기존 P Repair와 실제 C CriticRecheck producer에서 생성한
CPU 질문의 허용, role 입력과 닫힌 frame의 거부를 검사하는 회귀를 추가한다.
Store task·paused token·closed frame 조건을 완화하지 않는다. 이 시점의 Rust
회귀는 아직 컴파일·실행 전이다. 새 소스의 실제 실행은 별도로 다시 인수한다.

Python 독립 reader도 `/2` producer·Rust reader와 같이 counterexample을 None으로
제한한다. 원 trial·publication·ACK 핀을 다시 봉인한 비정상 입력도 거부하는
부정 사례를 포함한 집중 검사 10개가 신경망 import 차단 상태에서 통과했다.
이는 실제 학습이나 실제 cohort publication 증거가 아니다.

보존 영역이 18GiB를 약 19MiB 초과하여 비활성 build 201의 실행물·의존 파일
24개를 Windows 보존 영역에 복사·fsync·해시 대조한 뒤 정확한 Linux 원 roster만
회수했다. 약 1.50GiB를 반환했으며 모델·데이터·기보·실패 보고서는 삭제하지
않았다. 원 명령을 재현하기 전에는 보존된 custody mapping으로 원 경로를 복원한다.
이 기록과 새 자원 프로필은 기능 검증용이며 성능 개선의 근거로 사용하지 않는다.

## 25차 체크포인트: 새 소스 검사 완료와 실제 후보 문맥 불일치

`f82194b`에서 own CPU 재검토 집중 회귀 1개가 실제 통과했다. 최초 `c5475be`의
검사는 양성 질문 준비를 통과한 뒤 닫힌 frame의 정확한 오류 예상을 잘못 지정해
실패했다. 공통 admission의 `RoleError::InvalidOutput`을 기대하도록 검사만 고쳤고
제품의 closed/owner/Rules/credit 조건은 유지했다.

같은 `f82194b`의 새 빌드 240은 11단계를 모두 통과했다. UCI 797개, Search 1070개와
기존 ignored 2개, inventory 1개, 엄격 workspace Clippy를 확인했다. 제품 단계 5~8은
release, 독립 reader와 metadata caller 단계 9~10은 dev로 명시하고 실제 산출 경로의
원 바이트·의존 파일을 보존했다. 제품 10개와 caller 2개를 회수한 뒤 자식 종료·임시
정리를 확인했으며 관리 슬롯은 약 4.2GB로 8GiB 안에 남았다. 원격 Linux·Windows CPU,
CPU bindings, PALS model, native validation bundle의 다섯 job도 이 SHA에서 모두
성공을 관측했다. 이 결과를 실제 네 모드·GPU·cohort 인수로 바꾸지 않는다.

새 등록 241은 현재 실행물·lock·원 boot/parent와 CPU 2개·memory.high 3GiB/max 4GiB
readback을 연결한 passive metadata다. 수치 검사 247에서 실제 Rust NNUEC forward
20개와 독립 scalar FP32를 대조했고 accumulator·hidden·raw/value/WDL 최대 절대차는
모두 0이었다. 옛 수치 결과를 재사용하지 않았으며 GPU나 학습 검사는 아니다.

CPU P-C-R 241은 depth 1/2·mate·movetime 착수를 관측한 뒤 다음 infinite 착수 부근에서
`published root ranking/candidate question scope` 불일치를 기록했다. stderr에는
오류의 원 frame과 시각이 없어 movetime 말기와 다음 root 중 어느 곳인지 원 로그만으로
단정하지 않는다. exit 0·양 pipe EOF·강제 신호 없음에도 의미 실패로 판정하고 원자료
5개를 회수했으며 다른 CPU·GPU·cohort 후속 실행은 보류했다. 목표는 수정 작업을
계속하는 active 상태이며 원 실패를 정상 대국이나 성공으로 바꾸지 않는다.

독립 소스 검토 두 건은 initial candidate가 원 ranking view를 이어받지 않고 문맥을
다시 선택하는 생산 경계를 지적했다. 실제 소비 이후 보호 횟수와 query 토큰 예산에
따라 record 범위·순서가 바뀔 수 있지만 coordinator는 원 public context를 요구한다.
최초 fanout은 봉인된 ranking 문맥에서 query를 파생하도록 수정 초안을 반영했다.
이미 제출된 view·ticket을 갱신하거나 범위 검사를 느슨하게 하지 않으며, 이전 frame
종료 후 새 view를 만드는 sequential deferred 경로와 구분한다. 보호 소비로 재선택
순서가 바뀌는 사례와 원 token budget이 가득 차 추가 query를 거부하는 사례를
회귀로 추가했으며 이 체크포인트의 컴파일·실행은 아직 전이다. 새 실제 실행의
인수도 남는다.

240~247 실행 준비 스크립트는 새 provenance `/2`로 분리했다. 원 숫자 토큰·64hex와
역사 source를 보존하고 현재 source 8곳만 명시 갱신한 뒤 총괄이 파생 15개의 AST와
숫자 상수·문자열 이외 구조를 대조했다. 일반 teacher용 248/249도 source-only 제안과
독립 검토를 보존했으나 실제 242/244·cohort·teacher·IPC 결과는 아직 없다. 기존
087/088 소비자와 원 receipt는 수정하지 않는다.

새 보존 영역 관측은 약 18.01GB로 18GiB 안이며 추가 build223 회수는 실행하지 않았다.
소스 구현·실제 CPU 수치·기능 실패·미실행 GPU를 구분한다. 성능·기력·실제 학습·병합은
여전히 이번 인수에 포함하지 않는다.

## 26차 체크포인트: frozen scope 회귀와 현재 CPU 실행

최초 ranking fanout은 원 `ReviewView`에서 query만 fork하도록 `9048b6a`에
수정했다. 기존 deferred 후보의 새 문맥 선택과 coordinator의 범위·owner·Rules·
credit 검사는 유지한다. 최초 집중 실행은 컴파일과 순서 회귀를 통과했지만,
4ply 자료를 3ply Store에 넣은 capacity fixture 오류로 실패했다. 검사 설정만
4ply로 고친 `6afa16b`에서 집중 검사 3개가 모두 통과했다. 원 문맥이 가득 찬 경우
추가 query는 `ViewCapacity`로 거부하며 기록을 재선택해 맞추지 않는다.

`6afa16b`의 새 build250은 11단계를 모두 통과했다. formatting·현재 UCI/Search·
inventory·엄격 workspace Clippy와 release 제품 10개·dev caller 2개를 확인하고
원 바이트와 의존 파일을 보존했다. 관리 캐시는 약 4.2GB로 남았고 종료·임시 정리도
확인했다. 이 SHA의 Linux·Windows CPU, bindings, PALS model, native validation
bundle 다섯 CI job도 모두 성공을 관측했다. 새 문서 커밋이나 실제 제품 인수와
원 실행 소스 `6afa16b`의 범위를 구분한다.

새 WSL boot에서는 old241의 parent/inode를 재태깅하지 않았다. registration251
직전 미존재를 확인한 parent를 실제 생성하고 boot·inode·UID·빈 프로세스 상태·
controller와 CPU2·memory.high 3GiB/max 4GiB readback을 대조했다. numeric257은
실제 Rust forward 20개를 독립 FP32와 대조했으며 accumulator·hidden·raw/value·
WDL의 최대 절대차가 모두 0이었다. 이전 수치 실행을 새 실행으로 재사용하지 않았다.

CPU P-C/P-C-R251은 각각 8개 go, stop·ponder·newgame·quit를 마쳤고 primary/secondary
오류 없음, exit0·양 EOF·강제 신호 없음이었다. 각 원자료 5개를 회수했다. 이전
ranking scope 불일치는 이 실행에서 재현되지 않았다. 다만 외부 UCI harness는
제품 physical receipt와 formal R0 증거를 발급하지 않으므로 해당 항목은 여전히
`unknown_without_product_receipt`이며 NN 계수 진단·OS 종료와 구분한다.

source252·membership253·series254의 공식 passive 준비는 통과했다. 실제 별도
intervention255는 첫 root admission에서 `intervention windows exceed original root`로
실패했다. profile의 prelude5s+action10s와 go15000ms가 같은 경계에 있고 실제 root
hard deadline은 시간 관리 reserve를 포함하므로 caller의 여유를 별도로 확인한다.
원 physical completion unknown과 부분 자료를 보존하며 arm·publication·ACK·dataset
성공으로 채우지 않는다. guard를 완화하지 않고 새 입력·실행 ID로 수정한다.

실제 GPU·일반 teacher·외부 R·update0 IPC는 이 새 실행에서 아직 인수하지 않았다.
관리 보존 영역은 약 18.64GB로 18GiB 안이었으며 추가 파일 삭제는 없었다. 위 자료는
기능 검증이며 성능·기력 개선·학습·병합의 증거로 사용하지 않는다.

## 27차 체크포인트: 수명주기 보강과 중단 인계

사용자 요청으로 새 대형 NN/GPU·cohort 실행을 시작하지 않고 현재 준비 작업을
정리했다. intervention255는 prelude5s+action10s가 root hard deadline14990ms를
10ms 넘은 원 실패로 보존한다. 후속 caller는 profile을 바꾸지 않고 go20000ms를
선언하는 새 ID의 source-only 제안으로 남겼다. 262 파생 준비기의 숫자 치환이
`sha256`을 `sha266`으로 바꿔 metadata 준비에서 실패한 기록도 보존한다.
새 272~276 제안은 protocol literal 보호와 좁은 예외를 포함하지만 실제 파생·NN
재시도 인수는 아직 없다. 일반 teacher258 metadata와 259 caller JSON 준비도
통과했으나 실제 teacher root·policy publication·strict 소비와는 구분한다.

수명주기 코드는 `e8d3755`와 `cc2282f`에 반영했다. 고정 purpose 빌드 슬롯, 새 output의 공통
예약·보존 카탈로그, CPU 원 runtime CAS의 유한 상한, 원 owner/종료 증거를 대조하는
명시 복구를 추가했다. 일반 실행과 유지보수는 OS gate·catalog/root inode를 대조한다.
hash와 decode는 한 held FD snapshot을 사용하고 PID 부재·age만으로 회수하지 않는다.
CI 성공 bundle의 compile-only 복제와 검증 환경 scratch의 종료 경계도 연결했다.
세부 사용법과 과거 자료의 처리 한계는 [저장소 유지보수](STORAGE-MAINTENANCE.md)에 둔다.

이 소스의 관리 실행기를 통한 Linux Python3.12 `scripts/tests` 77건 중 74건이 통과하고
Windows 전용 3건은 건너뛰었다. Windows Python3.10의 storage lifecycle·managed storage·followup
65건 중 59건이 통과하고 플랫폼 조건 6건은 건너뛰었다. 두 실행 모두 exit0·자식 트리
종료·scratch 정리를 확인했다. 첫 관리 실행의 Windows 검사는 종료 metadata 증가에
따른 작은 fixture 용량 경계로 실패했고, 원 로그를 보존한 뒤 마지막 metadata 쓰기까지
다시 검사하는 보완과 플랫폼 독립 fixture를 추가했다. CI bundle module은 Python3.10에
tomllib이 없어 Windows 로컬에서는 로드되지 않았고 Linux3.12에서 검사했다. 실제
owner 프로세스를 종료시킨 뒤 원 closure로 복구하는 작은 stdlib 검사와 잘못된 owner·
gate/디렉터리 교체·최종 출력 증가·unknown 예약·손상 CAS 거부를 포함한다.

기존 실험 결과·모델·PGN·구형 lease·미등록 sibling을 일괄 삭제하지 않았고 현재 전체
디스크 사용량을 다시 측정했다는 주장도 하지 않는다. 빌드12GiB·출력18GiB·원 runtime
CAS4GiB는 각각의 등록 영역 기준이며 C: 전체나 VHD 물리 사용량의 보장이 아니다.

현재 실제 제품 실행 근거는 여전히 `6afa16b`/build250이다. 저장소 tooling 변경을
옛 바이너리에 재태깅하지 않는다. 재개 시 원 source·wrapper·자산·자원·boot의 영향
입력을 다시 대조한다. 남은 핵심 인수는 로컬4050의 GPU 두 모드·수치/물리 완료,
N05 실제 action/cohort publication·strict 소비, update0 IPC 종료, 일반 teacher의
실제 P/C target 소비와 외부 CPU_R 연결, 그리고 31개 필수 항목의 최종 근거 감사다.
성능/기력 비교·실제 학습·조건부 후속·병합은 완료로 처리하지 않는다.

## 28차 체크포인트: 재개 후 등록 경계와 제품 수명 관측

Windows Python 3.12의 경로 stat과 FD fstat은 ctime의 의미가 달라 원 관리
실행기가 Cargo 진입 전에 소유권 변경으로 거부했다. `90481c7`은 Windows의
FD↔경로 교차 비교에 공통 birthtime을 사용한다. 같은 FD의 전후 ChangeTime,
inode·link·size·mtime 검사는 유지한다. 새 CI의 Linux·Windows CPU, CPU bindings,
PALS model 네 job은 성공을 관측했으며 native validation bundle은 이 기록 시점에
진행 중이다. 로컬 Python 3.10/3.14 집중 검사와 정확 SHA의 CI를 구분한다.

새 boot의 registration291은 build250/STRIP251 원 바이트를 다시 확인하고 별도
parent·inode·controller·자원 readback과 연결했다. 등록은 passive metadata이며
학습·NN·제품 준비 완료 권한을 발급하지 않는다. 등록 준비281의 ArtifactRef 생성에
메타데이터용 1MiB 상한을 잘못 적용한 실패와 재개 preentry의 과거 parent 불일치는
보존했다. 바이너리에만 명시적 상한을 적용하고 새 parent와 boot를 대조하는
별도 실행 준비기로 수정했으며 원본은 덮어쓰지 않았다.

Hybrid P-C292는 native preparation에서 configuration identity 불일치로 종료했다.
실제 backend 선언이 CUDA bundle JSON의 raw SHA를 사용한 반면 제품은 공식
`CudaRuntimeBundleSpec::digest()`의 semantic SHA를 요구한다. Hybrid P-C-R도 같은
선언 오류가 있어 후속 GPU 실행을 멈췄다. 제품 guard는 유지하고 새 metadata·lock으로
수정한다. 이 실패는 ORT session/forward 전에 발생했으며 GPU 수치·물리 완료 인수가
아니다. 원 실패·종료·EOF 자료와 physical unknown을 보존한다.

`--lifecycle-diagnostics=stderr-v1`은 원 제품의 root/initial/finish receipt를 읽는
선택형 관측이다. 기본은 off이며 policy·credit·물리 완료 권한을 바꾸지 않는다.
root authority·input lock·원 Start·deadline을 대조하고 PC의 실제 R0와 shared
NNUEC closure/readback, 전체 세션 finish를 분리한다. 관측 JSON에서 live authority를
복원할 수 없다. row/전체 byte 상한과 최종 행 공간을 두고 진단 오류도 원 정리 뒤에
처리한다. 최종 행의 complete는 출력 전 관측 범위이며 전달 ACK가 아니므로 정상
process exit와 양 EOF를 함께 확인해야 한다. 새 집중 검사 3개와 launch 검사 5개가
실제 WSL 관리 실행기를 통해 통과했고 자식 트리 종료·scratch 정리를 확인했다.
이 검사는 실제 네 모드의 원 receipt 인수를 대신하지 않는다.

N05의 source272~actual275, 일반 teacher282/283, update0 IPC의 owner287 준비는
서로 다른 단계로 관리한다. 현재 등록291과 원 STRIP251의 번호를 분리하고 원
rawpin2/독립 경로 pin의 차이를 좁은 접점에서 검증한다. 실제 publication·별도 ACK·
strict reader·IPC 종료가 없으면 준비나 구문 검사만으로 완료하지 않는다. 실제 학습,
원격 성능·기력 비교와 병합은 계속 범위 밖이며 목표는 구현·기능 인수를 진행 중이다.

## 29차 체크포인트: 새 실행물·metadata 준비와 GPU 미인수 유지

`e3e7075`의 수명 진단과 공식 bundle 선언 도구를 공유했다. 진단 집중 검사 3개와
launch 검사 5개, bundle metadata example 14개 및 영향받는 library/example의 엄격
Clippy가 통과했다. build298의 다섯 단계와 세 dev 실행물을 별도로 보존했다.
Windows worktree를 WSL에서 검증할 때 Git directory/worktree 경로와 실제
`core.autocrlf=true`를 맞추고 양쪽 HEAD·clean을 확인한 뒤 공개 compiled source를
명시했다. 이 경계를 갖추지 못한 294/297 준비 실패를 실제 source 등록으로 사용하지
않는다. 새 출력은 550MiB 예약에 등록하고 원 ELF·의존 파일·strip 결과를 보존했으며
allocated ELF section 동일성을 확인했다. 이 빌드는 기능 검증용이며 속도 자료가 아니다.

새 CI는 JSON source catalog의 두 새 diagnostics 경로가 실제 embedded source
목록에는 빠졌음을 기존 inventory 검사에서 검출했다. 두 component에 같은 leaf를
추가하며 assertion이나 catalog에서 경로를 제거하지 않는다. 이 수정의 실제 CI는
후속 정확 SHA에서 확인해야 한다. 옛 네 CPU job의 성공과 취소된 native bundle을
새 source의 전체 성공으로 재사용하지 않는다.

공식 Rust bundle digest로 corrected registration295의 네 mode metadata와 lock이
준비됐다. 실제 공개 모델 boundary 다섯 source는 원바이트가 같고 CPU search,
encoding/reference/export/runtime component와 원 NNUEC 조건도 대조했다. 해당
준비기는 모델·ONNX·라이브러리 본문을 열거나 session/NN을 실행하지 않는다.

새 Hybrid P-C 실행302는 wrapper가 XDG cache 환경을 전달하지 않아 runtime 단계에서
종료했다. 원본은 보존하고 명시 환경과 별도 유한 window/lock304를 등록했다.
304는 CUDA inventory·ORT runtime·모델 session·product assembly와 UCI/ready까지
통과했다. 첫 depth 요청에서는 외부 harness의 등록360초와 달리 실제 타이머가
180초로 남아 timeout됐고, quit 한도 뒤 TERM으로 종료했다. 원 exit/양 EOF와
빈 cgroup은 확인했지만 NN 완료·원 root/session physical fence를 얻지 못했으므로
GPU 물리 완료는 unknown이다. 후속 GPU 실행은 중단하고 하네스의 실제 clock과
release 실행물을 준비한다. 준비 성공·device 종류만으로 GPU 수치 인수를 발급하지 않는다.

병렬 metadata 준비272→273→274 factory/series→275 입력 조립은 모두 실제 exit0과
원 child cleanup/EOF로 끝났다. 새 go20000ms와 prelude5s/action10s 조건을 보존하고
actual275 실행은 별도다. safe-v3는 원 prefix와 old002 역사 pin을 보존하면서 현재
preentry291을 원 held FD로 검증·실행한다. 일반 teacher282 source, 공식 새 recipe와
validate,283 caller 및 두 helper003 plan 준비도 완료했다. 이 결과는 실제 P/C teacher
target이나 일반 corpus publication·strict 소비를 대신하지 않는다. owner287은 현재
등록291/STRIP251을 분리한 source·구문 준비이며 실제 dataset 이후 zero276을 진행한다.
actual Collector·teacher/root·optimizer·학습 완료는 아직 이 체크포인트의 결과가 아니다.

## 30차 체크포인트: 실제 개입 실패와 준비 단계 routing 수정

`b16d681`의 embedded source 목록 수정은 새 관리 실행의 inventory 집중 검사에서
통과했다. 해당 SHA의 CI는 CPU bindings 성공과 Windows workspace 검사 단계의
성공까지 관측했으며, 나머지 job의 최종 상태는 이 기록에 포함하지 않는다.
관측 중인 CI나 같은 이름의 과거 성공을 실제 제품 인수로 사용하지 않는다.

원 `6afa16b`/STRIP251의 actual275-v3는 약 1.14초 만에
`intervention route missing`으로 실패했다. 원 프로세스 종료·양 EOF와 빈 delegated
parent는 확인했지만 제품이 보고한 `PhysicalCompletionUnknown`을 보존한다.
원 source·입력·실패 receipt와 준비272~275를 새 실행 성공으로 바꾸지 않는다.

원인은 UCI manager가 observer 등록만 된 prelude에서 일반 R의 실제 준비 완료를
개입 neutral 완료로 처리한 분기였다. source snapshot의 Trial 선택 전에 route를
요구했다. trial·route·rejection이 모두 없는 경우만 원 R 설치를 계속하도록 수정한다.
실제 발급 뒤의 route·원 capture·권한 검사는 유지한다. 기존 NNUEC owner 통합검사에
실제 actor의 installed Bound 관측과 route 발급 뒤 capture 누락 거부를 연결했다.
private 상태 접근은 manager 내부의 test 전용 경계로 제한하며 Search/Arena는
변경하지 않았다. 이 source 수정의 컴파일·실제 재시도는 후속 정확 SHA에서 확인한다.

새 GPU 하네스305의 실제 clock은 최초 시작점 기준 실행360초·정리390초이며
보고서의 360000/30000ms와 일치한다. 독립 AST 집중 검사는 이 두 계산을 확인했다.
원304 타이머 실패는 보존하며 새 lock·leaf·epoch 없이 같은 실행을 반복하지 않는다.

외부 프로필 도구070은 compiled source `6afa16b`와 관리 tooling `b16d681`의 원문
핀을 명시적으로 분리한 새310 경로에서 빌드·보존했다. 원 구형 실행기가 최신
closure metadata를 거부한306 실패는 유지한다. 새 출력은128MiB 예약과14일 검토
기한을 가지며, streaming ELF 해시·보존 FD readback·원 supervisor tree 종료·
retained catalog/sidecar 일치를 확인했다. 이 결과는 외부 R 실행이나 신경망 완료가 아니다.

남은 실제 인수는 N05의 새 source arm/publication·별도 ACK·strict 소비,
일반 teacher의 P/C target·publication·strict 소비, 외부 R의 실제 counter→Repair→
C/Fresh 연결, 실제 dataset 이후 update0 IPC, 현재 네 mode와 독립 Fresh GPU 수치·
물리 종료다. 준비 문서·metadata·빌드와 각 실제 결과를 계속 구분한다.
실제 학습·원격 성능/기력 비교·병합은 이번 범위 밖이다.

## 명시적 archive-on 저장 프로필

긴 같은 게임의 저장 용량 회복은 별도 archive-on 프로필로 선택한다. 기본은
archive-off이며 저장 프로필을 생략한 기존 실행을 자동 전환하지 않는다.
`--storage-profile`, `--storage-profile-bytes`, `--storage-profile-sha256`,
`--archive-root`, `--storage-checkout-root`, `--storage-protected-root`의 여섯
옵션을 함께 지정한다. 프로필은 기존 실행 lock과 별도 domain으로 경로·quota·I/O·index/load
한도와 archive helper 자원을 고정하며, 이 선언 자체가 Store·root·물리 완료 권한을
발급하지 않는다. 실제 원 process/lock·directory capability·최초 go의 opaque
admission을 검증한 뒤에만 원 semantic owner를 archive worker에 이동한다.

PC, NNUEC PCR, own-board PCR은 각각 원 소유 종류를 보존한다. own-board PCR은
전체 Engine Box와 원 CPU checker mirror·TT·Store를 함께 유지하고 PC나 deferred
Researcher로 바꾸지 않는다. 외부 R의 archive-on은 계속 명시적으로 거절한다.
기존 NNUEC 공유 helper와 archive helper를 구별하고 전체 helper 예산에 계수한다.
초기 game 0과 이후 NewGame의 실제 reset은 각각 한 번만 소비하며, 같은 게임의
후속 root에는 새 Store나 대체 권한을 발급하지 않는다.

착수 사이에는 원 선택 문맥·부모 관계·증거를 checkpoint하고 필요한 자료만 원
authority로 복원한다. 디스크 이동을 증거 삭제나 handle 재발급으로 처리하지
않는다. 필수 pin 자료·quota·I/O·deadline 한도를 충족하지 못하면 원 오류와
Capacity/Unknown을 유지한다. native/CPU의 물리 종료 뒤 최종 checkpoint·close와
archive worker join을 별도로 확인하며, 실패·panic·완료 불명은 원 owner와 함께
보존한다. CPU/mock 검사·소스 검토는 실제 긴 게임의 archive 왕복, 목표 GPU,
종단 UCI 정상 종료 및 성능 인수를 대신하지 않는다.

`--lifecycle-diagnostics=stderr-v1`을 함께 선택하면 별도
`rz-rovent-storage-maintenance-observation/1` 행으로 원 maintenance receipt의
authority·nonce·hot 통계·archive owner의 commit/reclaim/load 계수와 실제 phase join을
제한된 범위에서 관측한다. `after_root`는 해당 go의 원 phase와 일치해야 하며,
`after_finish`는 실제 final checkpoint·close·join이 있어야 complete다. 이전 root의
정상 join이나 초기 중단을 현재 phase의 성공으로 사용하지 않는다. 원 lifecycle
domain·실행 결과·물리 완료 권한은 그대로이며, 누락·잠금 실패·출력 한도 초과는
incomplete다. NNUEC/PCR의 미관측 live hot owner를 0이나 실측 peak로 해석하지 않는다.

native CI의 `native-validation-diagnostics` feature는 기존 진단 접점을 포함하는
기본 off validation alias다. 실제 stderr 관측에는 위 CLI opt-in이 계속 필요하다.
이 빌드 등록과 CPU 검사는 실제 cold 회수·복원이나 외부 counter→Repair→C/Fresh의
성공 증거가 아니며, 새 source와 실제 실행을 별도로 등록한다.

2026-10-19의 source `3f953b8e164fe3f1ba325daee2c9ff119bf1667e`는 실제 startup
metadata 자식16개, 독립 CPU reference와 RTX4050 native CUDA의 106 cases/212 NN
inputs를 완료했다. 공식 Rust whole106/212와 같은 first12 raw의 subset100/200
수치 판정 및 정상 물리 종료를 확인했고 원시 자료98개를 회수해 검증했다. 이는
학습0·성능 주장0인 기능/수치/수명 인수다. 이후 저장소 tooling639의 CI 다섯 job
성공, 새 관측 소스의 검사, 실제 archive/UCI 인수와 각각 구분한다. archive 준비의
caller deadline 실패는 원 자식 종료와 함께 보존하며 실제 제품 실행 성공으로
표시하지 않는다.
