# rz-eval — C 평가 구현

> **현재 CUDA 실행 경계:** 아래의 bundle 기반 실행·복사 설명은 과거 기록이다.
> 신규 bundle 기반 제품 실행과 bundle 복사 API는 명시적으로 거절하며,
> V1/V5 기록·metadata 읽기는 보존한다. 시스템 CUDA는 별도 V3/V6 경로로
> 준비 중이고 실제 GPU 인수는 별도로 확인해야 한다.

담당: TASK-C01/C02/C03. 기준 규약은 저장소의
[구현 지시서](../../docs/IMPLEMENTATION-DIRECTIVES.md),
[계약](../../docs/CONTRACTS.md), [선정 기록](../../docs/WEIGHT-SELECTION.md)이다.

## 실제 backend

PR #7의 병합된 계약 0.1(`18339c30754f4f38d957d96aeab481b018b98718`)을 확인했다.
`contracts` feature는 이 SHA의 공통 crate를 소비한다. workspace 통합 때 총괄이
모든 consumer를 같은 path dependency로 바꾸어야 한다. 공통 타입·루트 workspace는
수정하지 않았다. Linux 검증 toolchain은 PR #7 CI와 같은 Rust 1.96.0이다.

`asset::MaiaAsset`는 선정 gzip/protobuf와 변환 ONNX의 크기·SHA-256을 검증한다.
`tools/prepare_maia.py`는 외부 LC0 고정 commit의 FP32/opset 17 변환과 provenance
작성을 제공한다. 변환 자산·GPL 도구는 저장소 밖에 보존한다.

`onnx` feature는 `ort=2.0.0-rc.10`/ORT 1.22.0을 명시 동적 로딩한다. 기본 feature는
native runtime/CUDA 설치가 필요 없다. CPU/CUDA 선택, shape/dtype, 최대 batch 16,
1~4 CPU threads, IO staging 예산을 검사한다. CUDA는 TF32 off·CPU fallback 금지와
실제 node placement probe를 요구한다. IO 예산과 CUDA arena cap은 전체 RAM/VRAM
상한이 아니므로 bootstrap/실행 환경에서 activation·workspace·동시 점유를 별도 제한한다.

CUDA arena는 요청 크기만큼 확장하는 SameAsRequested를 명시한다. 기본의
power-of-two 확장으로 다른 상주 엔진의 여유를 줄이는 일을 피하려는 저장 정책이다.
tensor·정밀도·kernel·arena 한도와 CPU backend의 설정/identity는 유지한다.
CUDA digest에는 cuda-arena-extend=same-as-requested-v1을 포함하며 old CUDA의
cache/실행 증거와 혼용하지 않는다. BT4 수치·No/Repeat·dual-resident 메모리·
실패 원인 조사와 source별 인수는
[pilot 기록](../../docs/research/BT4-FINAL-SELECTION-PILOT.md)을 따른다.
이 설정은 물리 VRAM 부족 원인의 확정이나 모든 모델의 GPU 인수를 대신하지 않는다.

선정 원본→ONNX 변환, 원본 protobuf의 **LC0 Eigen** 대비 실제 Rust ORT CPU 검사를
통과했다. 서로 같은 ONNX를 두 언어에서 실행한 대조가 아니다. 12개 국면에서 모든
입력 plane·합법 수 index를 정확히 비교하고 batch 1/2/4/8/16을 검사한다.
최대 오차는 logits `1.0252e-5`, WDL `3.5763e-7`, 합법 정책 `1.6094e-6`이었다.
실제 GPU/RTX 4050·VRAM·강도 인수는 미실행이다.
검증 source SHA·tool/asset digest·국면별 오차·제한·미실행 범위는
[CPU 인수 기록](validation/maia-cpu.json)에 고정했다. Rust 1.96.0에서 eval 27개와
encoding 11개(총 38개) 테스트, fmt·Clippy가 통과했고 Rust 1.85.0 all-feature
check도 통과했다. 기본 feature 검사는 native ORT/CUDA 없이 별도로 통과했다.

## 검증된 native runtime 저장

`runtime_pin::RuntimeCache`가 CPU 단일 library 또는 Linux CUDA 19-file bundle의
검증된 복사본을 공유한다. `RuntimeLibraryPin`의 명시적인 private-copy API는 보존하되
일반 native UCI·`maia_check`·`rules_maia_check`는 공유 캐시를 기본으로 사용한다.
`--runtime-cache-root=<absolute private cache slot>`로 별도 루트를 지정할 수 있다.

기본 위치는 Windows `%APPDATA%/RoveZero/cache/native-runtime-v1/<OS>-<architecture>`,
Linux `${XDG_CACHE_HOME:-$HOME/.cache}/rovezero/native-runtime-v1/<OS>-<architecture>`,
CI `$RUNNER_TEMP/RoveZero/cache/native-runtime-v1/<OS>-<architecture>`다. 비밀·가중치·
모델·로그를 넣지 않는다. root/ancestor의 소유권이 필요하며 link/Git 경로를 거부한다.

cache key는 CPU의 basename+digest 또는 CUDA의 canonical bundle digest다. miss만
private copy→writer close→readonly 검증→원자적 directory publication을 수행한다.
hit는 mutable source를 읽지 않고 cached bytes 전체를 expected digest와 대조해 새
파일 pin을 얻는다. 손상·누락·extra file·쓰기 권한·symlink는 성공으로 바꾸지 않는다.
동시 creator는 bounded 60초 lock을 사용한다. 최대 4 entry/8 GiB이며 full cache·
비정상 종료의 lock/staging은 명시적인 실패와 소유자 확인 대상으로 남긴다. 활성 library를
자동 수리·evict/unload하지 않는다. Unix readonly는 same-UID sandbox가 아니다.

저장 출처 `cache_created/cache_reused`는 별도 metadata이며 raw evaluation의
`Computed/RawEvalHit`나 신경망 실행 수를 변경하지 않는다. 실제 인수 소스 `5558359`의
CPU Maia/CUDA BT4 12-case·batch 1/2/4/8/16, Rules No/Repeat·mapping·physical drain과
일반 UCI의 재사용 결과는 [저장 인수 기록](../../docs/research/PERFORMANCE-OPTIMIZATION-PLAN.md#13-실행별-native-library-복제-제거와-검증된-공유-저장)에 둔다.

## 현재 제공 범위

`mock::ScriptedBackend<K>`는 수동 시계로 움직이는 **물리 backend 시험 도구**다.
`K`에는 호출자가 공통 계약의 요청/실행 문맥을 넣는다. 별도 RequestId, generation,
EvalRequest/Result, scheduler 또는 search 타입을 선언하지 않는다.

- `Step`이 callback 지연·원시 출력/실패·물리 완료·취소 응답을 정한다.
- `advance_to`와 `next_event`를 분리해 같은 tick에서 root 교체·취소·callback을
  원하는 순서로 소비할 수 있다. 지연 polling에도 원래 event 시각을 보존한다.
- 중복 callback, 다른 요청의 역순 완료, missing head, NaN/Inf를 주입할 수 있다.
  seed는 replay metadata이며 숨은 난수 대신 script가 실행을 결정한다.
- 취소는 callback이나 `DeviceCompleted`를 지우지 않는다. 논리 취소 성공은 D의
  finalization, buffer 해제는 실제 물리 완료, accepted backup은 B의 책임이다.
- script 크기·payload·in-flight·대기 event를 제한한다. admission과 cancel 거부는
  다음 script를 소비하거나 기존 physical completion을 잃지 않는다.

이 도구 자체의 성공을 D의 exactly-once finalization 또는 B의 backup 검증으로
보지 않는다. B/D의 실제 consumer 연결 검사는 통합 단계에 남아 있다.

## Maia 출력 검증

`output::validate_maia`는 선정 FP32 export의 1858개 raw policy logits와 이미 확률인
W/D/L head를 검증한다. 호출자가 checked legal move에서 구한 model index 목록을
요청 순서대로 넘기며, 합법 index를 먼저 gather한 뒤 temperature 1.0 softmax를 한다.
WDL에는 softmax를 다시 적용하지 않고 실제 차례 관점을 유지한다.

모든 필수 head의 shape·유한값을 검사한다. legal 목록의 중복·범위·빈 목록, WDL의
범위·합 오류를 명시적으로 거부하며 clipping·자동 renormalization·가짜 무승부로
수선하지 않는다. `ValidatedHeads`는 검증된 **모델 head**만 담고 공통 EvalResult의
요청·세대·관점 metadata를 대체하지 않는다. terminal 상태는 Rules/Search가 선행 처리한다.

첫 코드 profile의 확률 합 허용 오차는 `1e-5`다. 이는 probability admissibility이며
선정 기록의 독립 FP32 parity 기준(raw logit atol=1e-4/rtol=1e-3, 확률 max_abs=1e-4)과
구분한다. 이 수치 설정과 temperature는 C backend identity에 포함한다.
공통 payload는 검증한 f32 head의 반올림을 보존한다. B의 strict f64 합 검사에 넘길 때는
공통 `LegalPolicy::normalized()`와 `Wdl::normalized()`를 명시 호출하는 B adapter 정책을
사용한다. 허용 범위를 벗어난 입력을 그 함수로 복구하지 않는다.

## 계약 0.1과 물리 worker 연결

`contracts::MaiaBinding::for_backend`는 registry가 발급한 model/encoding handle을
로드한 자산·backend에 연결한다. model manifest는 `MaiaAsset::manifest_digest()`,
encoding manifest는 `encoding_manifest(fill)`을 사용한다. full compute는 고정
feed-forward 한 번(`steps=1`), FP32만 지원한다. `input_key`는 encoding handle과
실제 f32 tensor의 little-endian bytes, 알려진/padded 이력 정보를 SHA-256에 넣는다.
이 함수는 C 입력 codec이며 Rules의 상태 digest를 대신하지 않는다.

A/embedding adapter는 **해당 request의 immutable snapshot**에서 `classical::Input`을
투영하고 `input_key`로 요청을 구성한다. `prepare`는 이를 다시 인코딩해 input identity,
차례·history-fill·backend·descriptor·예산·합법 수 순서를 검사한다. 좌표는 a1=0이며
공통 Move의 캐슬링은 king destination(e1g1/e1c1 등)으로 받는다. 실제 king bit를 보고
LC0 king-to-rook index로 변환하며 네 승격을 보존한다. C는 generic P의 합법성이나
Rules metadata를 독립 증명하지 않는다. `HOST_BYTES_PER_ITEM`은 C buffer 예약이며
Rules snapshot과 native workspace 예산은 별도로 더한다.

`spawn_onnx_worker`는 세션 하나를 worker 하나로 옮긴다. D가 발급한 ExecutionId와
`PreparedBatch`를 `submit`하면 nonblocking `PhysicalLease`를 돌려준다. 동시에 물리
batch 하나만 허용하며 자동 재시도·batch 축소는 없다. `poll`의 Ready에는 물리 완료 후
owned EvalOutput 또는 실패가 들어 있다. output은 문맥·legal order·실행 ID를 그대로
보존한다. D는 batch 실패를 각 원 요청에 연결하고 `EvalOutput::validate_for`로 현재
scope/clock을 재검사한 뒤 한 번만 finalization해야 한다. B는 backup 직전에 다시 검사한다.
실패의 `PhysicalFailure`에는 공통 오류와 별도로 원래 C code/stage 및 ORT code·최대
1024-byte UTF-8 원인을 보존한다. 원시 원인 문자열은 local 진단용이며 공통 static detail과
기본 Debug/Display 출력에 복사하지 않는다. 복구/재시도는 수행하지 않는다.

논리 취소는 native Run을 중단하지 않는다. consumer/worker handle을 먼저 drop해도
실행 thread가 입력과 세션을 보유한다. wrapper panic으로 물리 완료가 불확실하면
`Quarantined`이며 backend·입력 pin을 프로세스 종료까지 보존한다. D는 이를 Ready로
바꾸거나 예약을 해제하면 안 된다. 이는 OS/native crash 복구를 보장하는 경계가 아니다.

D PR #4의 `83c2869000482ea1e5c8298b13431f97009c9095` 접점도 확인했다. D의
`Backend::dispatch/poll`을 직접 구현하려면 runtime/embedding 소유 경로에서 이 worker
lease를 감싼다. C→runtime 역의존은 추가하지 않았다. mock Callback도 DeviceCompleted
없이 Ready로 바꾸면 안 된다. 실제 A/D/B 전체 엔진 조합은 아직 검증하지 않았다.

## 개발 검사

루트 workspace 게시 전에는 crate별 manifest로 검사할 수 있다. `[workspace]`를
중첩 선언하지 않아 I01에서 그대로 workspace member로 등록할 수 있다. edition 2021은
이 crate의 현재 선언이며 최종 toolchain/MSRV·workspace 공통 설정은 총괄이 정한다.

```sh
cargo +1.96.0 test --manifest-path crates/rz-eval/Cargo.toml
cargo +1.96.0 test --manifest-path crates/rz-eval/Cargo.toml --all-features
cargo +1.96.0 clippy --manifest-path crates/rz-eval/Cargo.toml --all-targets --all-features -- -D warnings
cargo +1.96.0 fmt --manifest-path crates/rz-eval/Cargo.toml -- --check
```

임시 standalone Cargo.lock은 추적하지 않는다. 직접 의존성은 Cargo.toml에 exact pin했다.
전이 의존성까지 고정하는 root lockfile 도입은 총괄 소유다. build·toolchain·원시 결과는 저장소 밖 작업 전용 output
root에 둔다. 클라우드 작업 동안 보존하고 종료 전 재현 명령·검토된 결과는 PR에,
회수할 원시 근거는 별도 artifact로 인계한다. 자동 삭제는 하지 않는다.

## 실제 자산·독립 참조 검사 재현

저장소 밖에 선정 원본과 LC0 commit `fd71a2d921b689c5f479d3227c3806c8e272d9c5`를
준비한다. submodule은 해당 commit의 `b326b154221a6eb91977bdccf11d6f89e8547875`다.
LC0의 Meson 빌드에서 release, b_lto=false, gtest/ispc/plain_cuda/onnx/openblas=false,
metal=disabled, python_bindings=true를 사용했다. Eigen 3.4.0, Meson 1.8.2,
Ninja 1.11.1.4, GCC 14.2.0, Python 3.12 환경이며 빌드는 jobs=3·timeout 600초였다.
Meson wrapdb 접근 제한은 공식 GitHub release의 **같은 hash** patch를 packagecache에
받아 해결했다. upstream 소스는 수정하지 않았다.

검사용 venv: `onnx==1.18.0`, `onnxruntime==1.22.0`, `numpy==2.2.6`,
`python-chess==1.999`/`chess==1.11.2`. python-chess와 LC0 bindings는 외부 fixture
oracle일 뿐 제품 dependency가 아니다. `LC0_BUILD`, `LC0_SOURCE`, `ASSET_DIR`,
`OUTPUT_DIR`, `ORT_LIBRARY`, `ORT_SHA256`는 사용자가 준비한 절대 경로/해시다.

```sh
python crates/rz-eval/tools/prepare_maia.py \
  --source "$ASSET_DIR/maia-1900.pb.gz" --lc0 "$LC0_BUILD/lc0" \
  --lc0-source "$LC0_SOURCE" --output-dir "$OUTPUT_DIR/export"
PYTHONPATH="$LC0_BUILD" python crates/rz-eval/tools/maia_reference.py \
  --source "$ASSET_DIR/maia-1900.pb.gz" --lc0-source "$LC0_SOURCE" \
  --output "$OUTPUT_DIR/reference.json"
cargo +1.96.0 run --manifest-path crates/rz-eval/Cargo.toml --all-features --example maia_check -- \
  "$ASSET_DIR/maia-1900.pb.gz" "$OUTPUT_DIR/export/maia-1900.onnx" \
  "$OUTPUT_DIR/export/manifest.json" "$ORT_LIBRARY" "$ORT_SHA256" \
  "$OUTPUT_DIR/reference.json" "$OUTPUT_DIR/cpu-report.json" cpu
```

최종 명령을 60초/2 GiB 주소 공간 한도 아래 실행했다. test matrix는 12개 국면,
단일 및 batch 1/2/4/8/16, invalid/empty/oversized 입력, 정책 순열, 실제 CPU worker의
4개 계약 출력과 취소 1개 수락 거부다. 참조 binding의 FEN_ONLY는 각 fixture에서
명시한 No/RepeatOldest와 같음을 plane 전체 비교로 확인한다. 일반 FEN_ONLY profile을
지원한다고 주장하지 않는다.

CUDA 검사 시 마지막 `cpu`를
`cuda "$OUTPUT_DIR/cuda-probe" "$GPU_BUNDLE/bundle.json"`으로 바꾼다. 앞쪽의
`ORT_LIBRARY`·`ORT_SHA256`도 GPU wheel의 core 파일과 SHA로 지정한다. CPU wheel의
core SHA를 재사용하지 않는다. `cuda-probe`는 REPORT 부모의 새 직계 디렉터리여야
하며 예제가 생성한다. 이미 존재하는 profile namespace는 거부한다.

Linux CUDA bootstrap의 `CudaRuntimeBundleSpec` schema 1은 `schema_version`과
`files`를 받는다. 각 파일은 `role`, `filename`, `bytes`, `sha256`이며 role은
`core`, `providers_shared`, `providers_cuda`, `nvidia_dependency`다. ORT 1.22.0의
core/shared/CUDA 세 파일과 지정한 NVIDIA 16개를 모두 요구한다. 임의 파일명,
누락·중복·unknown JSON 필드·잘못된 크기·SHA를 거부한다. JSON 64 KiB, 파일 수 32,
파일당 1 GiB, 합계 4 GiB가 상한이다. 64 KiB buffer로 새 독립 inode에 복사하고
writer를 닫은 뒤 파일 `0400`·디렉터리 `0500`을 적용한다. 전체 19개 파일의 retained
handle과 canonical bundle digest를 보유한다. host driver인 `libcuda`와 OS 기본
라이브러리는 다운로드 bundle과 구별한다. Windows CUDA bundle은 명시 미지원이다.

`rz-native-loader`는 검증된 NVIDIA 절대 경로를 명시 로드하고 resident 이미지의
정체성을 검사하는 별도 작은 FFI 경계다. `rz-eval`의 unsafe 금지는 유지한다.
NVIDIA를 로드하기 전 ORT 3개도 선정한 GPU wheel의 정확한 size·SHA 선언과
대조한다. ORT가 provider를 로드한 실제 probe 뒤와 숫자·batch·C worker 검사 완료
후에는 NVIDIA 16개와 ORT 3개의 mapped 이미지를 재검사한다.
PATH·LD_LIBRARY_PATH 변경, 임의 ambient NVIDIA 또는 CPU fallback을
허용하지 않는다. runtime latch는 전체 bundle을 비교하며 실패와 부분 native 로드의
파일·handle도 프로세스 종료까지 보유한다. CUDA backend identity에 bundle과 loader
정책을 추가하며 기존 CPU 단일 파일 identity와 profile은 유지한다.

CUDA `Run` 오류나 wrapper panic은 물리 완료를 보장하지 않는다. 세션·활성 입력을
격리하고 `PhysicalRun::Quarantined`를 통해 worker에도 원래 bounded typed 원인을
보존한다. 그 lease는 Ready·ActualCompute 또는 예약 해제의 근거가 되지 않는다.
probe panic의 bounded 원문도 별도 local 진단에 보존하며 기본 오류 출력에는 넣지
않는다. 논리 deadline·취소는 이 물리 소유권을 바꾸지 않는다.

이 추가 GPU 경로의 소스 제공은 실제 C03 통과가 아니다. 목표 RTX 4050에서의 실행,
수치·kernel placement·mapped origin 인수는 총괄의 별도 실행 증거로 판정한다.
`maia_check`의 C worker 검사에는 fixture Rules view를 사용하며 실제 A/D GPU 연결을
증명하지 않는다. CUDA arena cap과
외부 sampler의 관측치를 전체 VRAM peak·종단 지연·강도 성과로 승격하지 않는다.
총 VRAM·메모리 peak·GPU 수명·종단 계측은 별도 I02 인수가 필요하다.

## 실제 A 상태의 CPU/CUDA 런타임 연결

`native_runtime_bridge::NativeWorkerOwner::from_onnx`의 기존 CPU guard와 CPU backend
identity는 유지한다. 별도 `from_cuda_onnx(backend, projection, diagnostic_capacity)`는
이미 로드된 CUDA 세션만 소비한다. device 0·batch 1·intra thread 1·FP32 full step 1,
정확한 model/backend identity, 실제 warm probe의 CUDA kernel placement, 현재 NVIDIA
16개와 ORT 3개의 mapping을 확인한 뒤 `NativeWorkerOrigin::CudaOnnx`를 발급한다.
모델 로드나 provider 초기화는 `dispatch` 또는 검색 deadline 안에서 수행하지 않는다.
`owner.cuda_metadata()`는 실제 bundle/placement SHA와 실행 node 수를 제공하며,
warm probe는 검색 요청의 결과 소비 증거와 구분한다.
owner는 binding의 검증된 history/encoding을 보존한다. 수치 인수의 기존 12개 참조는
No 5개·RepeatOldest 7개이며 원래 입력·출력을 재라벨링하지 않는다. 최초 제품 UCI의
HistoryFillNo 고정은 B factory와 별도 CUDA attestation이 소유하는 실행 profile 제약이다.

`NativeAdmissionPolicy::CudaOneGiB`의 `execution_resources()`는 요청의 `ByteBudget`에
추가하는 D 실행 예약(host 4096B·device 1GiB·pinned 0)이다. 현재 요청 자체의 device
budget은 0이며 같은 1GiB를 두 곳에 넣으면 중복 예약된다. 물리 완료 전에는 이 D
예약을 유지한다. `session_resident_admission()`의 device 1GiB는 bootstrap이 별도로
계상하는 상주 선언이며 개별 실행 완료로 해제하지 않는다. 두 선언 모두 실제 VRAM
측정값이나 전체 native 할당 hard cap이 아니다. ORT arena 1GiB 설정도 cuDNN workspace,
driver 등의 총 메모리 상한을 보장하지 않는다. pinned 0은 pinned buffer를 새로 제공했다는
주장이 없음을 뜻한다.

`from_worker_with_admission(..., NativeAdmissionPolicy::CudaOneGiB)`로 GPU 없는 환경에서
device admission·Q·취소 수명을 검사할 수 있다. 이 owner의 출처는 항상 `Injected`이며
CUDA metadata를 발급하지 않는다. CUDA Run 오류 또는 wrapper unwind로 완료가 미확정이면
원본 cause와 session/input을 유지하고 D `Pending`을 반환한다. 논리 취소·deadline이나
mapping audit 성공은 `Ready`, `ActualCompute`, buffer 재사용의 근거가 되지 않는다.
기존 runtime clone도 최초 mapping 실패의 process latch와 원본 typed cause를 소비한다.

`rules_maia_check`는 실제 A의 immutable 상태·이력·합법 수를 투영하여 C 추론과 D
finalization을 독립 Eigen 참조의 12개 사례와 비교한다. 기존 CPU 인자는 유지하며,
CUDA는 마지막 `cpu`를 `cuda PROFILE_DIRECTORY CUDA_BUNDLE.json`으로 바꾼다.
core 경로·SHA는 GPU bundle의 선언과 일치해야 하고, profile directory는 REPORT의
저장소 밖 부모 아래 새 직계 디렉터리여야 한다. 두 history-fill profile은 각각 새
placement prefix를 사용한다. CUDA 인수 보고에는 provider·native origin·bundle/placement
증거·admission 선언·최종 mapping audit·drain과 원본 진단이 함께 들어간다.

이 CUDA A/D 소스 추가와 주입 검사의 작성은 실제 실행 성공이 아니다. 실제 GPU 수치
검사와 B UCI 연결·취소/root 교체는 총괄이 같은 integration SHA에서 별도로 인수한다.
공통 계약 0.1의 fresh `ActualCompute`는 provider를 추가하지 않으며 정확한 backend
identity와 검증된 native 출처를 함께 대조한다. B의 CPU V1 영수증을 GPU 영수증으로
변환하지 않는다. D의 owner 관측 완료 시각·C worker 호출 시각·B의 backup 소비량을
구분하는 D02 계측은 후속 작업이며, 이 예제는 종단 성능·강도 인수를 주장하지 않는다.

직접 Rust dependency의 라이선스는 MIT OR Apache-2.0이며 ONNX Runtime 자체는 MIT와
포함 third-party notices를 따른다. 선정 원본/ONNX는 upstream GPL-3.0 외부 자산이다.
LC0 GPL 구현·테이블·protobuf 생성 코드를 제품에 복사·링크하지 않는다. 변환 산출물의
RIGHTS.txt는 provenance 메모이며 재배포 source/notice 의무 충족 확인을 대신하지 않는다.
# OPT-01 projection 실험

`experimental-bitboards`는 Rules가 유지하는 읽기 전용 12개 bitboard를 사용한다.
`experimental-history-frames`는 최근 8개 frame의 반복 정보를 전체 known prefix의
borrowed 단일 순회에서 구한다. 두 옵션은 기본 off이며 독립 활성화할 수 있다.
원래 square 재구성/owned-history 경로를 대조군으로 유지한다. 이력을 절단하거나
unknown-prefix·raw EP·history fill 의미를 변경하지 않는다.

`experimental-raw-cache`(OPT-06)는 실제 Classical 입력의 모든 f32 bit와
metadata, 모델·encoding·backend·precision·compute·epoch·game을 대조합니다.
`ClassicalProjection::configure_raw_cache`로 유한한 항목/바이트 상한을 지정하며
기본 runtime 설정은 비활성입니다. D 최종 승인 후에만 전체 raw policy/WDL을
재사용 후보로 승격합니다. hit도 새 요청을 정상 admission/완료 검증하며 새
ExecutionId를 만들지 않습니다. CPU mock/native factory의 typed 설정으로 연결하고
ucinewgame에서 초기화합니다. native typed report는 raw-hit와 Computed 집계를
분리하며 기존 Computed-only V1 attestation으로 raw-hit 실행을 게시하지 않습니다.

OPT-09~11은 `experimental-io-buffers`, `experimental-io-binding`,
`experimental-cuda-graph`를 각각 compile한 뒤 `BackendConfig.experiments`에서
명시적으로 선택합니다. native UCI의 같은 이름 `--experimental-*` 플래그로
실제 worker에 연결하며 기본 설정은 모두 off입니다. Graph는 CUDA B1 + binding을
필수로 하고, 고정 device input/output 주소를 유지합니다. host→device와
device→host는 ORT synchronous Identity copy이며 추가 세션 비용도 계측 대상입니다.
Run/전송/fence 오류에서 CUDA completion이 불확실하면 binding·input·session을
격리 보존합니다. `binding_runs()`는 성공한 synchronous 호출 수이지 실제 GPU
capture/replay의 관측 증거가 아닙니다. ORT buffer 실제 재사용·GPU 수치·capture/replay·
지연/VRAM은 장비 검증 항목이며 CPU 계약 테스트로 통과 처리하지 않습니다.

외부 자산을 사용하는 기존 `maia_check` 수치 대조 명령 끝에도 위 세 실행 옵션을
명시할 수 있습니다. 각 옵션의 Cargo feature를 먼저 켜며 중복·미지원 옵션은
거부합니다. Graph는 `cuda`와 `--experimental-io-binding`을 요구하고 batch 1만
실행합니다. 제외한 2/4/8/16은 보고서에 남깁니다. 실험 설정은 32회 B1 반복 대조,
소유 출력 불변성, `last_io_timings()`의 host staging·전송·Run·output fence·출력
소유화 구간을 `experimental_checks`에 기록합니다. 시계는 CPU wall clock이고 Run에
kernel·동기화가 포함됩니다. 성공 호출 수로 capture/replay를 확인했다고 주장하지
않습니다. GPU capture/replay·peak VRAM·quarantine 및 정식 성능 인수는 별도입니다.

OPT-12의 `experimental-batch`는 explicit `NativeWorkerOwner::from_worker_batched`
또는 `from_onnx_batched`/`from_cuda_onnx_batched`와 연결합니다. 한 물리 worker의
실제 PreparedBatch(최대 16개), 공유 ExecutionId, 각 요청의 독립 policy/WDL 변환을
사용합니다. 배치 일부 논리 취소는 나머지 요청을 물리 종료 전 해제하지 않습니다.
native UCI는 `--experimental-batch=N`로 B/D/C의 폭을 함께 설정하고 최대 batch
대기 200µs를 자체 시간에 포함합니다. Graph의 고정 B1 및 기존 V1 attestation과
동시에 켤 수 없습니다. mixed-legal 배치는 C의 전체 raw heads 지원을 전제로만
D에서 명시적으로 활성화합니다. 물리 ID ledger는 root당 최대 1024개로 제한하며
서로 다른 legal 수 때문에 dispatch 순서가 달라도 중복 ID를 거절합니다.
