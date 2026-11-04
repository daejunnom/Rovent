# NVIDIA native loader의 소유 경계

> **현재 CUDA 실행 경계:** 아래의 bundle 기반 실행·복사 설명은 과거 기록이다.
> 신규 bundle 기반 제품 실행과 bundle 복사 API는 명시적으로 거절하며,
> V1/V5 기록·metadata 읽기는 보존한다. 시스템 CUDA는 별도 V3/V6 경로로
> 준비 중이고 실제 GPU 인수는 별도로 확인해야 한다.

이 crate는 GPU bootstrap가 선택한 **Linux CUDA 12/cuDNN 9 profile의 16개
NVIDIA runtime ELF**만 `RTLD_NOW | RTLD_GLOBAL`로 로딩하고, 실제 mapped
path·device·inode를 확인한다. concrete model·tensor·ORT session·search를 소유하지
않고 `rz-contracts`에 의존하지 않는다. `rz-eval`의 `forbid(unsafe_code)`는 유지하고
native constructor를 실행하는 좁은 경계만 이 crate가 소유한다.

`OwnedLibrary`에는 원래 bootstrap가 권리·출처·manifest를 확인하고 전용 위치에
복사한 readonly file capability를 넘긴다. 16개 파일을 모두 받고 지정 순서로
정렬하여 로딩한다. 허용되지 않은 이름·중복·부분 집합은 거부한다. 입력 한도는
파일마다 1 GiB, 총 4 GiB, count 32이며 실제 지원 profile은 정확히 16개다.

bootstrap의 `RuntimeCache`는 같은 bytes를 해시별 공유 디렉터리에 한 번 게시할 수
있다. cache hit마다 전체 파일을 다시 검증하고 프로세스마다 새 readonly descriptor를
넘긴다. loader의 exact admission·단일 link·path/device/inode·실제 mapping·one-shot
latch·process-lifetime pin 검사는 그대로다. cache key나 저장 재사용 표시가 native
실행·신경망 평가의 성공 증거를 대신하지 않는다.

실행 admission은 `TRUSTED_NATIVE_PROFILE`의 exact 19개 이름·크기·SHA-256에
고정한다. 같은 basename 아래 다른 ELF를 놓고 caller가 새로운 hash를 선언해도
거부한다. 다른 native release는 총괄이 provenance와 실제 backend를 검토한 source
변경으로만 추가하며 사용자 설정의 arbitrary hash로 허용 목록을 늘리지 않는다.

```rust,ignore
validate_runtime_profile(&owned_ort_libraries)?;
let admitted = LibrarySet::load(owned_nvidia_libraries)?;
admitted.verify_mappings()?;
// 원래 bootstrap가 ORT core와 provider를 로딩하고 session을 초기화한다.
admitted.verify_mappings()?;
// 준비된 session으로 bounded warm probe를 수행한다.
admitted.verify_runtime_mappings(&owned_ort_libraries)?;
```

첫 native 호출 전에 모든 입력의 absolute path·basename·regular file·symlink
components·단일 link·readonly mode·O_RDONLY descriptor·descriptor/path의
device/inode/metadata·streaming SHA-256·ELF magic를 확인한다. file hash를 검사할
때 큰 파일 전체를 메모리에 올리지 않는다. 상대 경로·`.`·`..`·중복 `/`·newline·
backslash·비 UTF-8·과도한 길이와 깊이는 거부한다. 공백을 포함한 경로는 지원한다.

프로세스의 `/proc/self/maps`는 2 MiB, descriptor의 `/proc/self/fdinfo`는 4 KiB로
제한한다. maps의 고정 필드와 공백을 포함할 수 있는 pathname을 구분하며 device와
inode를 함께 대조한다. 모호한 escaped pathname과 deleted native mapping은 거부한다.
시작 전에 이미 mapped된 NVIDIA runtime 또는 ORT가 있으면 실패한다. GPU bootstrap의
순서는 **NVIDIA admission → ORT bootstrap → session/warm probe**다.

`LD_LIBRARY_PATH`, `LD_PRELOAD`, `LD_AUDIT`, `LD_DEBUG`, `LD_PROFILE`,
`LD_ORIGIN_PATH`, `CUDA_INJECTION64_PATH`가 nonempty면 실패한다. 해당 값은
출력·보고서·error field에 보존하지 않고 환경을 변경하지 않는다. 이 검사는 로딩
전과 mapping 검사마다 수행한다. 기존 process에서 다른 코드가 vendor runtime을
병렬 로딩하거나 loader 환경을 변경하지 않는 bootstrap 소유권이 필요하다.

각 native 호출 이후, 모든 로딩 완료 후, ORT 연결과 warm probe 전후에는
`verify_mappings()`로 NVIDIA 16개를 확인한다. 첫 provider/warm probe 이후에는
`verify_runtime_mappings(&ort_pins)`로 정확히 `libonnxruntime.so.1.22.0`,
`libonnxruntime_providers_shared.so`, `libonnxruntime_providers_cuda.so`까지
19개를 함께 확인한다. 이 3개는 원래 ORT bootstrap가 로딩하고, 이 crate는 전달받은
readonly descriptor를 최초 호출에서 복제해 static pin에 추가한다. combined bytes도
총 4 GiB 제한을 따른다. 다른 ORT bundle로 교체하거나 필요한 mapping이 없거나
다른 NVIDIA/ORT mapping이 추가되거나 path·device·inode가 달라지면 실패한다.
`libcuda.so.1`과 `libnvidia-*`는 host driver이며
이 bundled runtime profile에서 hash를 주장하지 않는다. OS·driver provenance는 별도
인수 증거로 기록한다.

`validate_runtime_profile(&ort_pins)`는 NVIDIA/ORT native 호출 전에 exact ORT3의
선언이 고정 profile과 같은지 검사하는 free function이다. 파일을 로딩하지 않으며
실제 bytes·readonly capability·copied path 검증을 대체하지 않는다. 원래 bootstrap는
전체 19개 파일의 실제 hash/소유권을 먼저 확인한다. loader는 NVIDIA16의 실제 hash를
다시 확인하고 전체 mapping 감사에서 ORT3의 실제 hash도 다시 확인한다. ORT 입력
배열의 순서는 식별자에 영향을 주지 않으며 고정 profile 순서로 canonical화한다.

성공과 실패 모두 **process-global one-shot latch**다. 같은 bundle의 성공 토큰은
복제할 수 있지만 다른 bundle로 재시도할 수 없다. 최초 mapping 검사 실패는 영구
보존되며 환경을 다시 바꿔 같은 토큰을 성공으로 복구하지 않는다. partial native
initialization 뒤의 rollback이나 안전한 unload를 주장하지 않는다. native handles와
file descriptors는 `ManuallyDrop`으로 보관하고 static latch가 프로세스 종료까지
소유한다. GPU worker·driver가 종료됐는지 알 수 없는 상태에서 `dlclose`하거나 pin을
해제하지 않는다. admission 도중 Rust panic이 나도 native handle/FD는 해제하지 않고
poisoned process를 restart하도록 한다. GPU 실패 후 CPU fallback과 다른 provider
재시도도 원래 ORT bootstrap의 공통 latch에서 닫아야 한다.

`LoadError`는 stable cause code와 static detail을 기본 출력한다. 원래 native/I/O
진단은 UTF-8 경계에서 최대 1024 bytes로 보존하며 `diagnostic()`의 명시적 로컬
접근으로만 확인할 수 있다. 진단은 private copied path를 포함할 수 있으므로 기본
Display/Debug·공유 기록에 자동 출력하지 않는다. `diagnostic_truncated()`는 저장한
진단이 원문 전체인지 bounded prefix인지 구분한다. 성공·실패 latch는 최초 원인을
그대로 보존하며 prefix의 hash를 전체 원문의 hash로 표시하지 않는다.

SHA-256은 총괄이 실제 vendor wheel에서 확인하여 고정한 manifest와 실제 파일의
**identity**를 확인한다. safe loader는 caller가 임의로 준 hash를 실행 trust로
받지 않는다. 고정된 vendor 코드와 host OS/driver의 ABI·soundness는 신뢰 전제이며
권리·보안 취약점 부재를 hash 자체로 인증하지 않는다. 이 crate는 arbitrary plugin
loader나 hostile native code sandbox가 아니다. 같은 권한의
공격자가 프로세스나 private directory를 동시에 변경하는 상황까지 차단한다고 주장하지
않는다. 권리·출처 확인, exclusive bootstrap, protected copied directory와 실제
backend의 ABI·숫자·장치 검사는 상위 인수 책임이다.

검사는 maps parser, exact path/profile, descriptor flags, foreign/deleted/missing/inode
mapping, failure latch와 bounded diagnostics를 다룬다. fixture 검사의 성공을 실제
NVIDIA loader·ORT·GPU inference 성공으로 표시하지 않는다. 이 crate만으로 모델의
실제 목표 GPU 수치나 VRAM·시간·buffer 수명 인수가 완료되지 않는다.

실행 profile의 공개 package provenance는 다음과 같다. 아래 SHA-256은 **wheel
archive**의 hash이며 이어지는 ELF hash와 다른 artifact다. vendor binary나 wheel을
이 MIT source crate에 포함하지 않는다. 라이선스·notice와 실제 다운로드 증거는
bootstrap의 별도 인수 자료다.

| package/version | 검증한 wheel SHA-256 | 공개 metadata |
|---|---|---|
| nvidia-cuda-runtime-cu12 12.8.57 | `75342e28567340b7428ce79a5d6bb6ca5ff9d07b69e7ce00d2c7b4dc23eff0be` | [PyPI](https://pypi.org/pypi/nvidia-cuda-runtime-cu12/12.8.57/json) |
| nvidia-cuda-nvrtc-cu12 12.8.93 | `a7756528852ef889772a84c6cd89d41dfa74667e24cca16bb31f8f061e3e9994` | [PyPI](https://pypi.org/pypi/nvidia-cuda-nvrtc-cu12/12.8.93/json) |
| nvidia-cublas-cu12 12.8.4.1 | `8ac4e771d5a348c551b2a426eda6193c19aa630236b418086020df5ba9667142` | [PyPI](https://pypi.org/pypi/nvidia-cublas-cu12/12.8.4.1/json) |
| nvidia-cudnn-cu12 9.8.0.87 | `d6b02cd0e3e24aa31d0193a8c39fec239354360d7d81055edddb69f35d53a4c8` | [PyPI](https://pypi.org/pypi/nvidia-cudnn-cu12/9.8.0.87/json) |
| nvidia-cufft-cu12 11.3.3.83 | `4d2dd21ec0b88cf61b62e6b43564355e5222e4a3fb394cac0db101f2dd0d4f74` | [PyPI](https://pypi.org/pypi/nvidia-cufft-cu12/11.3.3.83/json) |
| nvidia-curand-cu12 10.3.9.90 | `b32331d4f4df5d6eefa0554c565b626c7216f87a06a4f56fab27c3b68a830ec9` | [PyPI](https://pypi.org/pypi/nvidia-curand-cu12/10.3.9.90/json) |
| nvidia-nvjitlink-cu12 12.8.93 | `81ff63371a7ebd6e6451970684f916be2eab07321b73c9d244dc2b4da7f73b88` | [PyPI](https://pypi.org/pypi/nvidia-nvjitlink-cu12/12.8.93/json) |
| onnxruntime-gpu 1.22.0, cp312 Linux x86_64 | `86b064c8f6cbe6da03f51f46351237d985f8fd5eb907d3f9997ea91881131a13` | [PyPI](https://pypi.org/pypi/onnxruntime-gpu/1.22.0/json) |

ORT archive는 `onnxruntime_gpu-1.22.0-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl`,
283199528 bytes다. ORT upstream tag `v1.22.0`의 commit은
`f217402897f40ebba457e2421bc0a4702771968e`다. 다른 platform wheel의 동명 ELF를
위 profile의 입력으로 취급하지 않는다.

| 실행 허용 ELF | bytes | SHA-256 |
|---|---:|---|
| libcudart.so.12 | 728800 | `218eec4c8385a32e258a0235be4d449986844f2d0de3430052f0924e6fe60f71` |
| libnvJitLink.so.12 | 94101392 | `0369e6867d44b800437de4e146d72c65afc6c75adf677a15c2ecd8e6a7ac135f` |
| libnvrtc-builtins.so.12.8 | 6338504 | `eccaa824230ee7858a94a3055cb01f1cb634df05d313880d0bdcf195161fcb4e` |
| libnvrtc.so.12 | 104487248 | `43731e24cd89e3749826304f304e8aa11fbecf1188715271b1f5018d6212b5e6` |
| libcublasLt.so.12 | 751771728 | `10b5e6631cf8115c661eb895ed1533826308b58f7956466f53d236a40c9b622c` |
| libcublas.so.12 | 116388640 | `031ce6c2cbfbb9468f040527cab5c599069ce5609e73e28f87503881063eac21` |
| libcurand.so.10 | 136749240 | `f9bea038a2703b721571fd45a299a898141fd8cb264a5912635c95116f5960fe` |
| libcufft.so.11 | 278925016 | `5c912146449614f9d73ebd1a5cb604242da6b819d94ba5b4a99272a0649f3761` |
| libcudnn_graph.so.9 | 4391696 | `7b2abeb742ad5b737aeb50185481168b325ab443c87b66de3ceece6431e0e771` |
| libcudnn_ops.so.9 | 119547416 | `e822b34d447d2c83f275d2e1e6373a88c19dcadd7a7d289ed27d8ec52a210323` |
| libcudnn_adv.so.9 | 255866448 | `c7f2859225c2fc2992235d3a2504f641fa8ec83a1996a354b07609db8208a746` |
| libcudnn_cnn.so.9 | 6300904 | `3d1643d323d9dba75236d5b9a7741a5b6606ea6ffbcf0a6268c0c6e6d4cb3dd4` |
| libcudnn_engines_precompiled.so.9 | 583023472 | `4ab4bb62c6291d0a3650feb583e62693bdc6a5023352dbc64231187b2ec9cfdc` |
| libcudnn_engines_runtime_compiled.so.9 | 27276224 | `20aab41f9c36e846ba4ccc6e59b67251becfbf992e87d281ffaadbf4f24f32c5` |
| libcudnn_heuristic.so.9 | 56627064 | `6ad33b511a6f9b880c9c13018a669a6d818cddf6842700f5403906fa97eb5279` |
| libcudnn.so.9 | 125136 | `72f74476dcdc074b9d56f8c5cf576ceff6a595989832b9b0798478dc48a4ffcd` |
| libonnxruntime.so.1.22.0 | 19950288 | `09fb71acf9debf4c5755f2c49522bfa80e580223be83d3f4e18744788d4be7db` |
| libonnxruntime_providers_shared.so | 14632 | `ee7bf6d4d32f523dbdebf9dec70a1b3287a1a60d019192490732750dc385f98c` |
| libonnxruntime_providers_cuda.so | 407530104 | `6b9677938c96bf988a0a06958594778dc4b516a2d8ab8d4f54645195e29101d0` |

구현 근거는 [libloading 0.8.9 upstream](https://github.com/nagisa/rust_libloading/tree/0.8.9)
의 Unix loader API와 [Linux proc_pid_maps](https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html)
형식이다. dependency-first 순서는 총괄이 확인한 해당 bundled profile에 고정했다.
