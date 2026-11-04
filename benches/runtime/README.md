# Runtime 검증·반복 계측 도구

> **현재 CUDA 실행 경계:** 아래의 bundle 기반 실행·복사 설명은 과거 기록이다.
> 신규 bundle 기반 제품 실행과 bundle 복사 API는 명시적으로 거절하며,
> V1/V5 기록·metadata 읽기는 보존한다. 시스템 CUDA는 별도 V3/V6 경로로
> 준비 중이고 실제 GPU 인수는 별도로 확인해야 한다.

## 메모리 E 대조 자료의 누적 집계

[memory_evidence.py](memory_evidence.py)는 이전·현재 **동일 작업량의 쌍별 자료**를
읽기 전용으로 집계한다. 엔진·GPU·CPU PoC를 실행하거나 기본 옵션을 활성화하지 않는다.
원시 결과와 ledger·집계 JSON은 저장소 밖에 보존한다. 기존 보류·실패 판정을 덮어쓰지 않는다.

Ledger schema 1의 `comparisons`에는 전역 고유 `id`, `environment_epoch`,
`compatibility`, `baseline`, `variant`를 둔다. Compatibility의 필수 차원은
`option/workload/model/runtime/precision/batch/resources/fixed_work/time_scope/peak_kind/other_options`다.
다른 작업·시간 구간·메모리 정의는 별도 series다. 같은 질문의 소스·환경 변화는
epoch로 보존하며 그 사이 합계는 기술 통계다. 역사적 binary 크기·입력 차이는
해당 manifest에서 확인하고, 호환성을 확인하지 못한 결과는 합산하지 않는다.

각 arm에는 `layout=native|prepare|oracle|inference`와
`result={"path":"원시-result.json","sha256":"64자리 digest"}`를 지정한다.
Path는 ledger 기준 상대 경로나 실행 호스트의 절대 경로다. Native에는 `receipt`,
inference에는 `report`와 `reuse_buffers`도 지정한다. 추가 작은 JSON 증거는
`additional_evidence`로 묶는다. 이 도구는 raw 결과의 실제 시간·peak를 사용하며,
전달된 요약 숫자로 대체하지 않는다. Ledger·증거 JSON은 파일마다 16MiB로 제한한다.
이전 결과에 source SHA가 없으면 `registration={"path":"...","sha256":"..."}`의
고정 source를 요구한다. Raw source가 있으면 registration과 일치해야 한다.

CUDA CPU arena 비교의 inference arm에는 `disable_cuda_cpu_arena`도 명시한다.
새 report kind와 실제 옵션·fallback off·buffer reuse off를 대조하여 양쪽 옵션이
뒤바뀌거나 다른 실행을 같은 비교로 집계하지 않는다. 이전 ledger는 그대로 읽을 수 있다.

각 series는 기준·변경 T 합계와 `sum(T1)/sum(T0)`, 실행별 peak 관측값의 합계·평균과
`sum(P1)/sum(P0)`, 쌍별 비율의 범위·중앙값·기하평균을 함께 남긴다. **Peak의 합계는
독립 실행 관측값의 합이며 동시에 필요한 메모리나 시스템 peak가 아니다.** RSS와
cgroup peak를 합치지 않으며 VRAM·Windows commit 미관측을 대체하지 않는다.
공유 기준 실행은 옵션별 비교에 표시하되 실제 실행 건수에서는 한 번 센다.
따라서 공유 기준을 사용하는 옵션 결과는 서로 독립 표본이 아니다.

SHA 변조·중복 ID 충돌·같은 series의 실행 재사용·작업량 불일치·OOM·강제 종료·취소를
거부한다. `diagnostic_only=true` 또는 `performance_measurement=false`인 실행도
`accepted=true`·exit0 여부와 관계없이 성능 comparison에 넣지 않는다. 이 표식을
추가하기 전의 고정 작업량 자료는 기존 실제 receipt와 registration 검사로 대조한다.
동일 comparison의 중복 입력은 한 번만 센다. Conditioning과 실패·성공한 진단은
`excluded=[{"id":"...","reason":"...","evidence":[{"path":"...","sha256":"..."}]}]`
에 보존하고 비교 합계와 구분한다. 누적 평균만으로 이전 실패를 통과 처리하지 않는다.
[채택 기준](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#adoption-gate)은 별도로 적용한다.

```sh
python benches/runtime/memory_evidence.py "$RZ_LEDGER_JSON" "$RZ_FRESH_REPORT_JSON"
python benches/runtime/memory_evidence.py "$RZ_NEW_LEDGER_JSON" "$RZ_FRESH_REPORT_JSON" \
  --include-ledger "$RZ_PRIOR_LEDGER_JSON"
python -m unittest discover -s benches/runtime/tests -p test_memory_evidence.py -q
```

`--include-ledger`는 반복 지정할 수 있다. 이전 ledger를 수정하지 않고 새 실행의 ledger를
추가하며 각 입력 ledger SHA도 출력에 남긴다. 중복 comparison은 한 번만 센다.

## CUDA session의 CPU arena 한 변수 연구

`rz-eval`의 `experimental-ort-cpu-arena`는 기본 off이며 컴파일만으로 활성화되지 않는다.
`inference_bench --b1-cpu-arena=baseline|disabled`는 동일 바이너리의 BT4 FP32 B1,
warm3/timed20과 출력 digest를 대조한다. 여섯 절대 경로 인자는 기존 진입점과 같다.
`maia_check`의 `--experimental-ort-cpu-arena`로 독립 수치·수명 검사를 먼저 실행한다.
기본 batch sweep·UCI 대국 설정에는 적용하지 않는다.

두 비교 arm 모두 준비 단계의 `/proc/self/status`를 8개 경계에서 읽는다.
Session commit 직후에는 직렬화 원본이 아직 살아 있고 다음 관측은 owned 원본 해제 후다.
기존 생성자는 no-op observer를 사용하므로 일반 대국에 반복 PID/GPU 조사를 추가하지 않는다.
실패 보고서는 도달한 관측만 남긴다. RSS endpoint와 process-lifetime VmHWM은
phase peak·live heap·cgroup file cache·VRAM·Windows commit과 구분한다.
이 옵션은 ORT 내부 CPU allocator의 arena만 끄며 CPU node fallback 금지·CUDA 배치 증거·
물리 완료 조건을 유지한다. 새 backend identity는 별도이며 기존 I/O 옵션의 identity는 보존한다.

## Owned ORT flatbuffer 한 변수 연구

**현재 HOLD:** 최초 별도 CUDA 수치 실행은 계산 후 native heap 오류와 SIGABRT로
종료했다. 실패는 보존한다. Worker join 수정 뒤 source `f9ed02a`의 재검사는
12개 참조·batch1/2/4/8/16·native shutdown·exit0을 확인했다. 후속 세 쌍에서
direct ORT의 primary peak RSS는 약25.4% 낮았지만 준비 완료 RSS는 약590.6MiB
높고 전체 T 합계는 약20.2% 늘어 채택 문턱을 통과하지 못했다.
[실패·수명 수정·누적 자료](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#owned-ort-flatbuffer-ab)를 따른다.

`experimental-ort-model`은 기본 off이며 UCI·기존 B1 대국에 적용하지 않는다.
`ort_export SOURCE ONNX EXPORT CUDA_CORE CUDA_BUNDLE NEW_ORT NEW_MANIFEST`로
기존 CUDA FP32·TF32 off·Level1 경로에서 파생 모델을 별도 생성한다. 일곱 경로는
절대 경로이고 두 출력은 저장소 밖의 새 파일이다. 변환·placement probe는 독립
수치 인수를 대신하지 않는다. 원본 가중치·ONNX·export identity를 유지하고 파생
ORT·manifest·export runtime을 추가로 식별한다.

`maia_check`에 `--experimental-ort-model=ORT_ABSOLUTE_PATH`와
`--experimental-ort-manifest=MANIFEST_ABSOLUTE_PATH`를 함께 전달해 먼저 검증한다.
그 뒤 `inference_bench`의 기존 여섯 인자에 `--b1-ort-model=baseline|direct`,
ORT 절대 경로, 파생 manifest 절대 경로를 추가해 동일 바이너리·B1·warm3/timed20을
대조한다. Cache·buffer reuse·CPU arena 변경·I/O Binding·CUDA Graph는 함께 켜지 않는다.

Direct 경로는 Rust 소유 flatbuffer를 session 전체 수명 동안 유지한다. Native session을
먼저 파괴하며 CUDA 완료가 미확정이면 session·모델·입력을 함께 보존한다. 원본 ONNX의
조기 해제 경로와 달리 ready 이후에도 모델 바이트가 남으므로 startup peak와 ready RSS를
별도로 평가한다. ORT의 두 직접 참조 옵션은 ONNX protobuf에 적용하지 않는다.

Ledger arm에는 `zero_copy_ort`와 `derived_manifest` pin을 지정한다. 집계기는 실제
옵션·파생 provenance·retained bytes를 대조하고 baseline=false/variant=true를 요구한다.
해당 series의 기본 채택이나 기존 누적 자료의 판정 변경은 자동화하지 않는다.

`--b1-copied-ort=baseline|copied`는 같은 여섯 인자와 파생 모델·manifest 두 경로를
사용하는 독립 비교다. 수치 검사는 기존 파생 경로 인자에 `--experimental-ort-copy`를
추가한다. 이 옵션은 두 직접 참조 설정을 모두0으로 고정하고 정상 native copy가
끝난 뒤 기존 버퍼 해제 함수를 재사용한다. Direct 수명을 짧게 해석하지 않는다.
새 backend identity·report kind `b1_copied_ort_fixed_work`와 ledger option
`copied-ort-flatbuffer`를 사용하며 arm의 `copy_ort_model`은 baseline=false,
variant=true, `zero_copy_ort`는 양쪽 false다. Retained bytes는 양쪽0이며
다른 옵션은 계속 off다. 별도 수치·정상 종료 인수 전에는 이 A/B를 실행하지 않는다.
복사형 결과를 직접 참조형 누적 series에 합산하거나 UCI 기본 경로로 적용하지 않는다.
**복사형도 HOLD:** source `33f1b86`의 첫 수치 실행은 worker join 뒤 native heap
오류·SIGABRT로 종료했다. A/B 사전 등록·실행은 중단했으며 [실패 기록](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#copied-ort-flatbuffer-ab)을
보존했다. 수치 보고서 `passed`와 CPU CI 성공으로 전체 GPU 실행을 승인하지 않는다.

[종료 진단](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#ort-native-shutdown-diagnostic)은
cuDNN의 exit callback이 `calloc`할 때 heap 손상을 감지한 스택을 확보했다.
contracts 없이 main에서 Run/drop해도 재현했다. 자식의 tcache를 끈 두 진단은 정상
종료했지만 첫 손상 지점·portable 수정·효과 인수는 미확정이다. Allocator 설정과
제품 기본값을 변경하지 않으며 GDB/allocator 진단을 성능 ledger에 넣지 않는다.

[후속 heap 진단](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#ort-heap-first-write-diagnostic)은
정확한 glibc free-list 오류를 확인했다. 작은 native C API의 GLOBAL/LOCAL 실행은 exit0이지만,
원래16개 bootstrap을 LOCAL로 바꾼 전체 수치 경로는 같은 SIGABRT다. 첫 손상 원인은 unknown이다.
관측·부분 fixture·setup 실패9개는 한 excluded collection으로 보존하고 기존30개 비교·고유57회·
공유 기준3개·8개 series 값을 유지했다. 제외는17개이며 새로운 성능 A/B는 없다.
[Raw cache host head 공유](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#raw-cache-shared-heads)는
별도 소스 변경이다. CPU/mock 수명 검사는 통과했지만 GPU/전체 메모리·시간 효과는 미측정이다.
기존 ledger나 진단 숫자를 이 변경의 성능 증거로 사용하지 않는다.

[후속 CPU 할당·A/B](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#raw-cache-shared-heads-cpu-ab)는
고정24,576 hit에서 두 할당/hit 제거를 관측했다. Hook 없는 세 쌍은 합산 시간비0.93543,
peak 관측비1.12781로 HOLD다. 새 series를 분리해 누적33개 비교·고유63회·9개 series·
제외19개로 보존했다. `rules_maia_check --experimental-raw-cache`는 해당 feature에서만
활성화하는 독립 수치 gate다. 첫 GPU 실행은 예제 ID 역행 후 CUDA 종료 오류로 실패했으며,
ID 수정은 CPU admission 회귀로만 검증했다. 부분 수치·예약0을 GPU 전체 인수로 세지 않고
후속 GPU A/B·대국은 정상 종료를 포함한 독립 수치 인수 뒤에 진행한다.

[Native 종료 수정·긴 CPU A/B](../../docs/research/MEMORY-EXECUTION-OPTIMIZATION.md#raw-cache-native-shutdown-long-ab)에서는
수치 gate 예제의 request drain 뒤 native join 누락을 수정했다. 제품 UCI는 이미 join을
확인하고 있었다. 통계 수집 오류도 원래 work/정리 영수증과 owner 보존을 건너뛰지 않게 했다.
원본 BT4 B1 No/Repeat의 fresh12/cache24·두 profile join·exit0을 source b39dc4a에서 확인했고,
별도 의도적 WDL 오류는 원래 failed/exit1을 보존하며 정리했다. 이는 GPU 성능/대국 인수가 아니다.
복사형 ORT heap 문제는 별도 HOLD다. 고정1,572,864-hit CPU 세 쌍은 합산 시간비1.008407,
peak 관측비0.958297·문턱 실패로 cache 기본 off/HOLD다. 이전 짧은 조건과 분리하여 누적
36개 비교·고유69회·공유control3·10개 series/epoch·제외20개를 보존한다.

## D02 CPU/mock trace 재생

[cpu_trace 예제](../../crates/rz-runtime/examples/cpu_trace.rs)는 실제 `Scheduler`와
`rz-telemetry::profile::TraceCollector`를 연결하는 유한한 CPU/mock 재현 진입점이다.
Scheduler의 준비·대기·dispatch·물리 완료·출력 검증 사건과 소비자의 backup·출력
구간을 각각 기록하며, 요청 시작부터 출력까지의 종단 구간도 따로 기록한다.
단계 시간을 더해 전체 시간이나 처리량을 계산하지 않는다.

예제의 숫자 ID, 입력 값, backup checksum은 독립 fixture다. 체스 상태·신경망
가중치·탐색 방문 통계·공통 계약을 대신하지 않는다. `cpu-mock` provider는 실제
CPU 정수 계산 결과와 입력 `Arc`를 opaque lease에 보관하고, **fixture 시계의
3 tick이 지난 뒤** 물리 완료를 알린다. 한 tick은 1µs다. 단계 지연과
`fixture_*_rate`는 이 결정적 시계의 결과다. 실제 CPU 작업과 수명 경계를
검사하지만 실제 신경망의 CPU/GPU 지연·전송·VRAM·강도 근거는 아니다.
`actual_wall`은 시나리오 전체 재생의 `std::time::Instant` 경과 시간이며
`actual_wall_consumed_rate`는 이 벽시계로 나눈 실제 성공 소비 수다.
`actual_consumed_requests`는 observation transport와 독립적으로 성공 receipt를
받아 checksum backup을 한 횟수다. event 손실 때도 이 fixture 소비 수와 불완전한
collector의 `consumed_requests`를 구분한다.

## 재현 명령과 상한

저장소 루트에서 출력 루트를 저장소 밖의 작업 전용 디렉터리로 지정한다.
원시 TSV와 빌드 생성물은 그 루트에 두고 인수에 필요한 보고서는 작업 종료 전에
회수한다. 아래 명령은 해당 루트를 보존하며 자동 삭제하지 않는다. CI에서는
runner 임시 루트를 사용할 수 있다.

```sh
: "${RZ_D_OUTPUT_ROOT:?저장소 밖의 작업 전용 출력 루트를 지정하세요}"
mkdir -p "$RZ_D_OUTPUT_ROOT/reports" "$RZ_D_OUTPUT_ROOT/build"
CARGO_TARGET_DIR="$RZ_D_OUTPUT_ROOT/build" \
  cargo run --locked --offline --release \
  --manifest-path crates/rz-runtime/Cargo.toml --example cpu_trace -- \
  --scenario all --requests 32 --sample-capacity 512 \
  > "$RZ_D_OUTPUT_ROOT/reports/cpu-trace-all.tsv"
```

선택 인자는 `--scenario all|cold|warm|parent-child|sibling|transposition|eviction|long`,
`--requests 1..65536`, `--sample-capacity 0..65536`이다. 기본값은 각각 `all`,
32, 512다. 요청 수는 **시나리오마다** 적용되며 각 시나리오는 새 scheduler와
collector로 시작한다. 알 수 없는 인자·음수·범위를 넘는 값은 실행 전 오류다.
현재 Scheduler 생성자는 같은 값으로 계측 sample ring과 observation event ring을
제한하므로 `--sample-capacity`는 두 상한에 모두 적용된다.

| 항목 | 고정 상한 |
| --- | --- |
| 예약 요청 / batch item / 물리 실행 | 8 / 4 / 2 |
| 요청 host 예약 / batch 추가 host 예약 | 512 / 256 bytes |
| 전체 host 예약 | 4608 bytes |
| device / pinned 예약 | 0 / 0 bytes |
| batch wait / queue age / deadline reserve | 2 / 100 / 2 fixture µs |
| 요청 deadline | 준비 직후부터 1000 fixture µs |
| owner step 수 | `requests * 16 + 64` / 시나리오 |
| 벽시계 중단 | 10초 / 시나리오 |
| collector active 요청 / 실행 | 8 / 2 |

예약 수치는 입력·결과·batch overhead의 scheduler ledger이며 실제 process RSS가
아니다. provider·모델·메타데이터·계측 ring·보고서 정렬의 메모리는 별도다.
관측 transport와 collector의 저장량은 위 유한 설정으로 제한된다. 모든 정상
인수는 `try_recv`로 수행하므로 receipt 대기는 blocking하지 않는다. step·벽시계
상한 초과나 예상하지 않은 terminal 결과는 오류 exit로 남긴다.

## 일곱 입력 관계

| 시나리오 | 독립 입력 fixture 관계 |
| --- | --- |
| cold | 서로 다른 정수 입력 |
| warm | 두 입력을 반복하는 label; warm latent state 없음 |
| parent-child | 연속 입력을 부모→자식 관계로 표시 |
| sibling | 부모 label마다 네 입력, 인접 호환 batch key 적용 |
| transposition | 네 입력을 재방문; 모든 요청을 다시 실행 |
| eviction | 네 다른 label 뒤 원래 입력 재방문; cache 없음 |
| long | 유한한 연속 입력 trace; 누적 latent 갱신 없음 |

이 구분은 계측 도구의 trace class 분리를 검증한다. 실제 exact cache·dedup·
warm-state·eviction 정책의 동작이나 이득을 보고하지 않는다. 시나리오마다 첫
물리 실행에 속한 한 요청을 실행 중 취소하며, 그 요청의 늦은 물리 결과는 소비하지
않는다. 첫 요청이 살아 있을 때 같은 ID로 제출한 독립 시도는 중복 거부를
확인하고, 각 receipt 수신 직전 예약이 남아 있는지도 확인한다. 마지막에는 모든
receipt를 읽고 물리 실행을 drain하여 최종 세 예약 영역이 0인지 검사한다.

## TSV 해석

출력 첫 두 줄은 실행 종류·시계·유한 설정의 주석이고, 나머지는
`scenario`, `field`, `value`, `unit`의 네 열이다. 필드 이름과 시나리오 순서는
고정이다. 요청·실행·미소비 물리 항목은 서로 다른 카운터다. 각 단계는
`stage.<name>.total|retained|dropped|p50|p95|p99`를 제공한다. 백분위는 남아 있는
표본의 nearest rank이며 해당 시나리오와 `cpu-mock` backend의 표본이다.
전송은 GPU provider hook이 없어 `total=0`, 백분위 `na`로 남는다. missing 값을
0 지연으로 해석하지 않는다.

`physical_items_completed`는 완료된 물리 batch의 항목 수,
`consumed_requests`는 실제 successful receipt의 backup 소비 수다.
`unused_items_finalized`는 물리 완료 뒤 더 이상 소비하지 않을 항목 수이며 첫 실행
중 취소한 요청을 포함한다. `completed_unconsumed_items`는 아직 소비 가능성을
종료하지 않은 완료 항목이다. 정상 기본 실행의 최종값은 0이다. 한 항목짜리
시나리오는 실행 중 취소한 한 요청만 있으므로 successful consume이 0이어도 정상이다.

`reserved_peak.*`는 domain별 reservation ledger의 정확한 high-water mark다.
collector에 scheduler의 peak와 현재 값을 각각 전달하므로 receipt가 외부에서
해제되는 순간을 owner observation이 놓쳐도 peak를 유지한다. 서로 다른 domain의
peak가 동시에 발생했다는 의미나 실제 장치 사용량을 주장하지 않는다.

`completeness.*`는 event 손실·표본 eviction·잘못된 관측·active tracking·카운터
포화 등을 표시한다. `--sample-capacity 0`도 실행은 끝나지만 모든 runtime 사건이
생략되고 collector 관측이 불완전하므로 stage/처리량을 정상 전체 profile로
받아들이면 안 된다. 작은 capacity 역시 event transport 손실과 표본 eviction을
명시한다. 이 경우 runtime 결과 검사는 계속되지만 재구성할 수 없는 계측 사건은
추정해 보충하지 않는다.

collector는 최근 종료한 ID만 유한한 tombstone ring으로 기억한다.
`request_ids_evicted`나 `execution_ids_evicted`가 양수이면 전체 과거 ID에 대한
중복 탐지 근거는 없다. 예제는 새 숫자 ID만 사용하며 실행 도중의 의도적 중복
시도는 scheduler에서 거부된다. collector의 `caller_ids_must_be_unique`는 이
호출자 전제다. ledger 전체와 profile 완료 상태는 함께 보고하되 observation
손실을 실행 성공으로 감추지 않는다.

event 손실이 생기면 예제의 요청↔실행 관계 재구성 metadata를 비워 놓고 불완전
관측을 보고한다. 물리 완료 사건이 사라진 실행을 무기한 추적하지 않으며, 보조
execution map도 2개로 제한한다. 이 metadata 정리는 실제 scheduler의 lease나
reservation을 해제하지 않는다.

예제 검사 진입점은 `cargo test --locked --offline --manifest-path
crates/rz-runtime/Cargo.toml --examples`다. 네 검사는 일곱 시나리오의 전체 단계·
성공 소비·취소·늦은 미소비 완료, zero capacity의 불완전 계측, 단일 취소 요청,
2048 요청 trace의 capacity 1/2/4/16 유실과 보조 execution map의 peak 상한을
확인한다. TSV의 `auxiliary.peak_executions`도 같은 high-water mark를 제공한다.

이 진입점은 D02 내부 CPU/mock 도구의 baseline이다. 같은 weights·search 의미·
자원의 variant 대조나 D03 성능 개선 결과는 포함하지 않는다. 실제 B/C 소비자와
목표 GPU의 전송·완료·VRAM·전체 탐색 경로, 신경망 오차와 정식 대국은 별도 인수한다.

## 검증 자료 보존

원본 TSV 8개와 재수집 검사 로그 8개를 포함한 공개 보존 사본은 저장소 밖의
`reports/coordinator-integration/recovered-pr-evidence/exact-blobs/D-f2a6c60`에 회수했다.
26개 파일 모두 보존 커밋의 Git blob identity·길이와 대조했다. 출처는
[D 보존 커밋](https://github.com/daejunnom/RoveZero/tree/f2a6c605dccfe85f8127b52b76380844a4c9c81c/benches/runtime/evidence/aa2fe1c-d02),
인수 요약·manifest 참조는 [총괄 인수 기록](../../docs/INTEGRATION-STATUS.md)이다.
과거 정확한 실행 SHA·argv·UTC·요청별 입력·event stream의 누락과 초기 trace의
미확인 소스를 유지한다. 재수집 69개 통과는 D의 과거 standalone 구성에 대한
결과이며 현재 root workspace의 검사나 GPU·D03 개선 근거로 재사용하지 않는다.

## OPT-00 실제 입력·고정 방문 수 witness

[`optimization_witness`](../../crates/rz-uci/examples/optimization_witness.rs)는 실제
Rules adapter → C의 Classical/Maia 입력 준비 → D의 scripted runtime → B의
`ContractSearch`와 PUCT를 연결한다. 수동 clock과 명시적 합성 policy/WDL을 쓰는
정확성 도구이며 시간·신경망·GPU 성능 표본이 아니다. 제품 실행 경로에는 연결되지 않는다.

```sh
: "${RZ_OPT_OUTPUT_ROOT:?저장소 밖의 작업 전용 출력 루트를 지정하세요}"
mkdir -p "$RZ_OPT_OUTPUT_ROOT/build" "$RZ_OPT_OUTPUT_ROOT/witness"
CARGO_TARGET_DIR="$RZ_OPT_OUTPUT_ROOT/build" \
  cargo run --release --locked --offline -p rz-uci --example optimization_witness -- \
  --visits 16 --history-fill no \
  > "$RZ_OPT_OUTPUT_ROOT/witness/default-no.txt"
```

인자는 `--visits 1..64`, `--fixture all|trace-0|trace-16|trace-64|trace-128|trace-256|…`,
`--history-fill no|always`다. 기본값은 각각 16, all, no다. fixture 이름은 예제의
`fixtures()`가 선언한다. 출력은 64MiB, 각 탐색은 `visits * 16 + 64` pump/30초,
drain은 64 step으로 제한한다. 정상 입력 16개와 all 실행의 guard 6개는 22개 case다.

`CASE/LEAF/HISTORY`는 전체 known history·classification·digest·ordered moves와
선택 경로를, `SELECT`는 실제 PUCT edge 통계의 float bits와 선택 index를 기록한다.
`PREPARE/TENSOR-LE`는 C가 request에서 준비한 실제 key·indices·projection과 dense
tensor의 little-endian 바이트다. D 제출 직전에 같은 불변 request로 준비를 재현하며
D도 독립 prepare 검증을 수행한다. GPU buffer를 관측한 기록으로 해석하지 않는다.
`ACCEPTED/BACKUP/FINAL/DRAIN`은 실제 소비 값·backup 후 통계·종료/오류·물리 예약 해제를
기록한다. owner 발급 번호는 비교 대상에서 제외하되 stale guard는 검사한다.
`UCI-POSITION/UCI-LEGAL`은 반복 runner에 사용할 command와 합법 수 fixture다.

`--features rz-position/experimental-claim-preview`,
`rz-position/experimental-history-digest` 또는 두 feature의 쉼표 조합을 cargo에
지정하여 독립 대조한다. 기준 checkout에는 같은 예제와 public trace fixture를 연결한다.
실제 matrix와 범위의 한계는 [계획 10장](../../docs/research/PERFORMANCE-OPTIMIZATION-PLAN.md#10-opt-00-구현검증과-반복-cpu-진단)에 기록했다.

## OPT-00 유한한 직렬 UCI 비교

[`paired_search.py`](paired_search.py)는 외부 clock을 사용하는 진단 runner다.
표준 Python 3.12만 필요하며 shell을 거치지 않고 manifest의 engine을 실행한다.
새 process마다 `uci → ucinewgame/isready → position/isready → go → bestmove → quit`을
순서대로 수행한다. 정상 합법 수 또는 terminal의 `0000`을 확인하며 늦은 중복 bestmove,
stderr 진단·실패 exit·quit/drain timeout을 성공 표본으로 받지 않는다.

manifest JSON의 schema는 `rz-opt00-paired-search/1`이며 다음 필드를 잠근다.

| 필드 | 필수 내용·범위 |
|---|---|
| `provider`, `profile_mode`, `primary_metric` | `cpu_mock` / `onnx_cpu` / `onnx_cuda`, `off`, `position_bestmove_ns` |
| `seed`, `witness_sha256` | unsigned 32-bit seed, 비교한 correctness witness SHA-256 |
| `baseline`, `candidate` | 절대 binary 경로·SHA-256·40자리 source commit·build features·compiler·argv |
| `fixtures` | 1~32개; 고유 `name`, `position`, 유한 `go`, witness의 `legal_moves`, `terminal` |
| `timeout_ms`, `run_wall_limit_ms` | process 100~60000ms, 전체 run 100~3600000ms |
| `max_rss_mib`, `max_run_bytes`, `max_throttled_usec` | 관측 RSS 16~65536MiB, 산출물 1MiB~1GiB, throttle 증가 허용치 |
| `cpu_affinity`, `runner_cpu_affinity` | engine CPU 목록; 선택적인 runner CPU 목록과 겹치지 않음 |
| `resource_notes` | host·clock·장치·창 독립성 등 available/unavailable/unknown 전제 |

CPU mock argv는 `['--cpu-mock']`와 선택적인 `--mock-delay-ms=0..1000`뿐이다.
native는 해당 provider argv와 선정 model/export/ORT/CUDA bundle 인자만 허용하며
`expected_profile`의 model/encoding/backend digest를 요구한다. 자동 수집한 startup·
termination receipt의 PID/binary, revision 0.1, FP32/B1/full-step 1/worker 1/thread 1,
model·encoding·backend, 성공 종료·physical drain을 대조한다. CUDA는 verified mapping과
실제 실행 node 증거도 필요하다. 실제 inference가 관측되지 않으면 그 값을 유지한다.
receipt의 fixture 검사는 실제 native inference 인수를 대신하지 않는다.

`go nodes 1..128` 또는 `go movetime 1..10000`을 허용한다. 프로세스 실행 전 binary hash를
다시 검사하며 caller는 실행 중 binary·manifest·출력 루트를 독점 관리해야 한다.
RSS는 Linux `VmHWM`을 경계에서 샘플링한 값이며 OS의 강제 메모리 제한은 아니다.
미지원 platform의 RSS/cgroup/PSI는 unknown이다. 필요한 강제 제한·장비 통제는 외부에서
설정한다. 각 stream은 2MiB, 한 protocol line은 64KiB, reader queue는 64개다.

```sh
PYTHONDONTWRITEBYTECODE=1 python benches/runtime/paired_search.py \
  --manifest "$RZ_OPT_OUTPUT_ROOT/comparison.json" \
  --output "$RZ_OPT_OUTPUT_ROOT/smoke" --phase smoke --session 1
PYTHONDONTWRITEBYTECODE=1 python benches/runtime/paired_search.py \
  --manifest "$RZ_OPT_OUTPUT_ROOT/comparison.json" \
  --output "$RZ_OPT_OUTPUT_ROOT/pilot" --phase pilot --session 1
PYTHONDONTWRITEBYTECODE=1 python benches/runtime/paired_search.py \
  --manifest "$RZ_OPT_OUTPUT_ROOT/comparison.json" \
  --output "$RZ_OPT_OUTPUT_ROOT/confirm-1" --phase confirm --session 1
```

caller가 witness/빌드 metadata로 `comparison.json`을 만든다. 중복·미지 JSON key와
잘못된 hash·argv·범위는 거부한다. output은 저장소 밖의 새 경로여야 한다.
smoke는 fixture당 1 block, pilot은 baseline-baseline 3 + baseline-variant 3 block,
confirm은 호출한 창의 fixture당 4 block이다. 한 block은 ABBA 또는 BAAB 4 fresh process이며
seed/session으로 순서를 고정한다. 독립적인 confirm 창 2·3은 해당 `--session`과 새 output으로
별도 호출한다. 세 번 실행했다는 사실만으로 장비나 측정 세션의 독립성을 입증하지 않는다.

1차 지표는 **position 송신부터 bestmove 수신까지**이며 position 준비·ready barrier·go
전송 비용을 포함한다. `position_ready_ns`, `go_bestmove_ns`, startup/quit를 포함한
`process_wall_ns`도 별도 보존한다. 실제 `info nodes`만 `reported_nodes`에 기록한다.
현재 mock UCI는 이를 출력하지 않아 null이며 요청한 nodes를 완료 방문 수로 대입하지 않는다.
완료 방문 수의 정확성은 별도 witness에서 확인하고 이 runner의 throughput으로 옮기지 않는다.

출력은 `manifest.lock.json`, process별 stdout/stderr, `processes.jsonl`,
`execution.json`, `blocks.json`이다. 실패와 오염을 모두 보존하며 제품·인프라 실패 시
중단한다. 느린 표본을 버리거나 자동 재시도하지 않는다. cgroup throttle 또는 OOM/high
증가로 오염을 표시하되 실제 엔진 오류가 있으면 실패를 우선한다. unknown 압력·독점 여부를
청정 실행으로 확정하지 않는다. block 평균 B/A 비율은 진단 자료이며 자동 정식 인수와
신뢰구간 판정은 수행하지 않는다. source observer on/off는 후속 별도 비교다.

검사 명령은 `cargo test -p rz-uci --example optimization_witness --locked`와
`PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s benches/runtime/tests -q`다.
3개 Rust 검사는 실제 Rules/C/D/B 소비·예약 해제·guard와 상한을, 12개 Python 검사는
명시적 가짜 UCI controller로 runner의 실패/timeout/log/receipt 경계를 확인한다.
후자는 실제 engine 성능이나 GPU 검사가 아니다. Ubuntu·Windows CPU CI에서 함께 실행한다.

## OPT-01~12 후속 검사

전체 A 담당 구현의 독립 feature·한도·소비자 계약과 실제 실행 source는
[최적화 기록 11장](../../docs/research/PERFORMANCE-OPTIMIZATION-PLAN.md#11-opt-0112-전체-구현과-검증)에 있다.
OPT-01~08의 독립 옵션은 위 witness와 같은 actual tensor/fixed-visit 기록으로 비교한다.
bounded 상태 cache를 실제로 사용하도록 예제의 Rules wrapper도 retained-byte charge를
전달한다. 새 CPU cohort는 14 비교 × 두 fixture의 2,128 fresh process이며 이전
OPT-00의 584 표본과 합산하지 않는다.

cache/notify/S1 batch의 실제 A/C/D/B 연결·한 번의 backup·취소·drain은
`cargo test --workspace --all-targets --all-features --locked`의
`experimental_pipeline`·runtime/native bridge·batch wakeup 검사에서 확인한다.
이는 합성 raw head/clock을 사용한 CPU 정확성 검사이며 NN·GPU나 성능 표본이 아니다.
S1의 virtual reservation과 완료 순서는 S0와 달라질 수 있다.

위 `paired_search.py` native schema는 기존 Computed-only V1 baseline을 위한 것이다.
새 raw cache/I/O/batch 실험의 native argv·report를 이 V1 runner에 끼워 넣지 않는다.
buffer/binding/fixed CUDA B1 Graph의 수치 대조는 `maia_check`의 명시적인 실행 옵션과
별도 fresh report로 수행한다. 장치 성능과 S1 품질 대조는 새로운 mode/profile을 잠근
별도 cohort이며 기존 NN/대국 인수를 재사용하지 않는다.

native manifest는 asset의 flag/value pair를 유지한다. runner는 실제 Popen 경계에서
각 pair를 단일 `--flag=value` argv로 변환하며, 경로의 공백·등호를 split하거나 shell에
넘기지 않는다. 필수 asset flag는 한 번씩 있어야 한다. PR #17·#18 통합에서 실제 CPU
프로세스의 시작·종료 영수증까지 확인했으며
[최적화 기록 12장](../../docs/research/PERFORMANCE-OPTIMIZATION-PLAN.md)에 해당 근거를 남겼다.
