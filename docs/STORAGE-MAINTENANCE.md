# 생성물 수명과 명시적 유지보수

관리 기능은 새로 등록한 소유 영역을 제한한다. Windows C: 전체, WSL VHD의
물리 할당량 또는 과거에 직접 만든 모든 디렉터리를 제한하는 기능은 아니다.
빌드 cache, 보존 결과, 원 runtime 자산과 실제 loader cache는 서로 다른 책임이다.

## 고정 빌드 슬롯

일반 실행은 `python scripts/run_managed.py -- cargo ...`를 사용한다. 별도 진단이
필요하면 `--purpose startup-caller`처럼 **안정적인 목적 이름**을 지정한다.
시각·SHA·PID마다 target 형제 폴더를 만들지 않는다. checkout과 purpose가 함께
슬롯을 식별하며 기본 슬롯도 같은 카탈로그의 8GiB/12GiB/최대 4개 계산에 들어간다.
`CARGO_TARGET_DIR`로 관리 영역 밖을 지정하면 기존처럼 거부한다.

직접 Cargo를 실행하거나 임의 경로에 복사한 과거 target에는 관리 코드의 소유권이
없다. 아래 inventory로 누락을 확인할 수 있지만, 이름이나 오래된 시각을 근거로
자동 채택·삭제하지 않는다. 새 진단은 위 접점으로 옮겨 재발을 막는다.

## 출력 예약과 보존

`managed_outputs.py`는 모델 검증과 followup 실행기의 새 출력에 공통 카탈로그를
연결한다. 모델 검증은 기본 8GiB를 예약하고, followup은 기존 출력 상한을 예약한다.
등록 결과 전체의 기본 상한은 18GiB이며 최대 1,024개 기록을 허용한다. 완료·실패
결과도 합계에 포함한다. 여유가 부족하면 새 실행을 거부한다.

카탈로그 위치는 Windows `%APPDATA%/RoveZero/storage/outputs-v1`, Linux
`${XDG_CACHE_HOME:-$HOME/.cache}/rovezero/storage/outputs-v1`, CI
`$RUNNER_TEMP/RoveZero/storage/outputs-v1`다. 등록한 실제 출력은 원래의
reports/runs 경로를 유지하며, 경로·inode·소유 token을 카탈로그에서 대조한다.
한 카탈로그의 producer는 OS gate로 직렬화한다. Rust가 직접 생성해야 하는
fresh 경로는 deferred 예약으로 남겨 CLI 계약을 보존한다.

- 14일의 검토 기한은 **회수·보존 결정을 검토할 시점**이다. 만료가 모델·PGN·원시
  로그·실패 결과의 삭제 권한이 되지 않는다. 결과가 한도를 채우면 먼저 보존 근거와
  원래 native retention journal을 확인하고 별도 유지보수를 진행한다.
- 카탈로그는 과거 reports/runs를 자동으로 순회·등록하지 않는다. 따라서 초기 도입
  시 기존 미등록 자료까지 18GiB 안으로 줄였다는 주장은 하지 않는다.
- 실행 중 run_managed의 5초 관측과 producer의 검사, 종료 시 최종 크기 검사를
  함께 사용한다. 관측 사이 증가는 가능하며 kernel quota나 물리 사용량 제한은 아니다.
- `tree_gone`이 True가 아니면 active 예약을 유지하여 다음 producer를 차단한다.
  모델 검증의 venv는 바깥 관리 scratch에 남기고, 원 감독자가 실제 자식 트리 종료를
  확인한 뒤에만 임시 자료를 회수하고 같은 owner에 묶인 출력을 확정한다.
- OS 프로세스 종료는 GPU 물리 fence를 대신하지 않는다. 입력·seed·workspace와
  provider 완료 unknown의 정리는 기존 native 수명·quarantine·retention 계약을 따른다.

CPU 모델 검증의 원 ORT DSO는 실행별 native-runtime 대신 digest별 공유 저장소를
기본으로 사용한다. 기본 위치는 관리 루트의 `cache/pals-cpu-runtime`이며 4GiB
합계를 검사한다. 기존 `--runtime-store` 지정도 유지한다. 읽기는 크기 제한과
streaming hash를 사용하고, 손상된 기존 entry는 보존·거부한다. 이는 Rust loader가
관리하는 runtime cache 또는 GPU 버퍼를 합치거나 삭제하는 변경이 아니다.

CI native bundle은 원 FD→private ELF→tar의 custody를 먼저 검증한다. 성공과 모든
command tree 종료가 확인된 뒤 target·cargo-home·tmp·private ELF만 정리한다.
원시 logs, artifact와 build receipt는 남긴다. `compile-cleanup.json`의 최종 상태는
build 성공 영수증과 별도로 확인하며 실패는 `cleanup-failure.json`과 exit에 남긴다.

## 중단 lease 복구

`owner.json`을 오래됐다는 이유로 삭제하지 않는다. 새 실행은 host-local 식별,
boot, process 생성 시점, root inode를 원 lease에 묶는다. 원 supervisor가 tree
종료를 확인하면 정리보다 먼저 token에 묶인 `closure.json`을 저장한다. 일반
실행과 유지보수는 같은 OS gate와 catalog 디렉터리 identity를 확인한다.

`storage_maintenance.py recover-build`는 **원 owner의 raw SHA256**을 요구하고,
다음 둘 중 하나가 증명될 때만 그 소유 슬롯의 재생성 target/tmp를 회수한다.

1. 같은 host의 확인된 재부팅으로 이전 boot의 프로세스가 남을 수 없음.
2. 원 lease와 정확히 같은 종료 증거가 있고 원 owner가 종료됨. PID 재사용은
   process 생성 시점으로 구분하며, 다른 PID를 signal하지 않는다.

hash와 parse는 같은 held FD 바이트에서 얻고 commit 직전 다시 대조한다. root·slot·
gate가 바뀌거나 기록이 불명확하면 보존한다. 구형 v1 owner에는 위 근거가 없으므로
새 복구 명령이 임의로 권한을 만들어 주지 않는다. 별도 원 트리 조사 대상으로 남긴다.

복구 요청 전에는 `inspect-build-recovery`로 현재 owner raw SHA256·원 boot·현재
boot·원 프로세스 생성 시점과 기존 초기 복구 gate를 읽기 전용으로 대조한다.
이 명령은 owner·lease·lock·closure를 만들거나 바꾸지 않고 slot payload와 출력
catalog도 읽지 않는다. 초기 gate 일치는 실제 복구 완료나 출력 종료의 증거가 아니다.
뒤의 slot·catalog·보존 자료 검사는 기존 복구기가 별도로 수행한다.

재시작 후에는 이전에 복구한 owner의 SHA와 현재 owner의 SHA를 구별한다. 원
owner가 없어졌다면 그 복구는 끝난 것이며, 후속 실행이 만든 새 owner를 같은
실패로 보고 다시 재시작을 요구하지 않는다. 새 owner가 남았으면 해당 실행의
실패·종료 경로를 먼저 조사한다. 기존 reader가 받지 못하는 종료 자료는 형식과
범위의 미지원으로 기록하며, 그것만으로 추가 재부팅이 필요하다고 결론 내리지 않는다.

`recover-output`은 같은 host의 재부팅, 원 catalog record의 SHA·경로·inode를
확인하여 interrupted-retained로 옮긴다. 이 명령은 결과나 GPU 입력을 삭제하지
않는다. 생성 중이던 deferred 경로의 identity가 확정되지 않았다면 계속 거부한다.

## 자원 획득 전 검사와 실제 실행의 분리

owner는 실행 중의 소유권과 불명확한 종료를 보호하는 기록이다. 모든 작업에서
owner를 없애는 것이 아니라, 소유 자원이 필요한 구간만 획득하고 원 감독자가
실패 경로까지 종료하도록 한다. 현재 가능한 작업 경계는 다음과 같다.

| 작업 | 소유권·종료 경계 |
|---|---|
| 소스 검토·diff·명시된 파일의 식별·순수 codec/계약 검사 | operational build/output owner를 만들지 않는다. 검사 자체의 외부 I/O·자식 실행은 별도로 확인한다. |
| 작은 CPU/mock 회귀 | 격리된 fixture 영역과 종료 정리를 사용한다. fixture의 모의 owner는 실제 실행·GPU 수명의 증거가 아니다. |
| 빌드·큰 출력·실제 엔진/추론 | 명시한 관리 슬롯·출력 예약과 원 process supervisor가 필요하다. source-only 검사와 명령을 분리한다. |
| Windows가 WSL 실행을 감독하는 경로 | Windows Job 종료와 Linux 원 실행의 종료를 각각 확인한다. Windows client 종료만으로 Linux·GPU 완료를 인정하지 않는다. |

파일명·역할 목록·CLI·shape·고정 설정처럼 자원 없이 검증할 수 있는 조건은
acquire 전에 검사한다. 현재 `run_managed.py`에서 acquire 이후 환경 준비가
실패하더라도 **자식 실행 호출 전**이면 원 manager가 실패 상태로 finish한다.
이 정리의 실패도 원 오류와 함께 보존한다. `run_group` 호출을 시도한 뒤의 예외에
같은 무자식 종료 근거를 적용하지 않는다.

별도 실행기에도 이 경계를 직접 연결해야 한다. 저장된 JSON의 `not_started`나
호출자 PID 부재는 원 실행의 무자식 근거가 아니다. 부분 생성된 continuation/FD의
close 실패를 재호출 성공으로 덮지 않고, 소유권이나 종료가 불명확하면 기록을 남긴다.
새 카탈로그·다른 purpose로 옮겨 중단 owner의 차단을 우회하지 않는다.

기능 결과의 성공 조건과 저장소 writer의 종료 조건을 분리한다. 원래 감독자가
오류 코드로 끝난 작업도 모든 소유 process·FD·cross-OS writer의 종료를 실제로
확인했다면 실패 상태로 마감할 수 있어야 한다. 반대로 결과 decoder·인수 ACK가
실패했다는 이유만으로 하위 kernel 종료 관측을 없던 것으로 만들지 않는다.
원 기록을 보존하고 기능 인수와 자원 종료 각각의 검사·미확인 범위를 남긴다.
이 구분 자체가 기존 기록의 소급 closure 발급이나 recovery gate 우회를 허용하지는 않는다.

재부팅 없이 가능한 복구는 원 종료 근거에 따라 다르다. `recover-build`는 정확한
원 closure와 owner 사망 증거가 있으면 같은 boot에서도 가능하지만, 현재
`recover-output`의 중단 기록 복구는 실제 재부팅을 요구한다. 정상 실행의 원
supervisor가 출력을 마감하는 경로는 이 중단 복구와 구분한다. 재부팅은 누락된
종료 증거를 상상해서 채우는 방법이 아니며, 과거 GPU 물리 완료를 발급하지 않는다.

## 명령 예시와 관측 한계

다음은 저장소 루트에서 실행하는 예시다. 복구 명령은 실제 대조한 원 digest를
전달해야 하며 예시 값을 사용하지 않는다. CLI는 새 lease를 자동으로 훔치지 않는다.

```powershell
python scripts/run_managed.py --purpose startup-caller -- cargo build --locked -p rz-uci
python scripts/storage_maintenance.py inventory --root "$env:APPDATA/RoveZero/build" --root "$env:APPDATA/RoveZero/reports" --root "$env:APPDATA/RoveZero/runs"
python scripts/storage_maintenance.py inspect-build-recovery --root "$env:APPDATA/RoveZero/build/managed-v1" --owner-sha256 <previously-observed-owner-sha256>
python scripts/storage_maintenance.py recover-build --root "$env:APPDATA/RoveZero/build/managed-v1" --owner-sha256 <original-owner-sha256>
```

inventory는 최대 16개의 **명시적·겹치지 않는 root**만 길이 기준으로 조사하고 삭제하지
않는다. root별 비밀 이름·link 경계·entry 수를 검사한다. Linux root 사용자, 일반
사용자, `/var/tmp`와 Windows는 필요한 root를 각각 명시하여 따로 관측한다.
서로 다른 root의 hardlink 중복과 VHD 안팎의 물리 사용량은 합계에서 해결하지 않는다.
파일 삭제와 VHD 축소는 별개이며 전역 WSL 종료·압축을 자동 실행하지 않는다.

등록 카탈로그 밖의 임의 스크립트·수동 복사는 이 한도를 우회할 수 있다. 새 대형
producer는 `OutputLease` 또는 기존 native retention 접점에 연결해야 한다.
기존 미등록 자료의 분류·회수는 실제 소유자·사용 여부·보존 근거 확인 후의 별도 작업이다.

## 연구 ZIP·중복 ELF 검증 자동화

`storage_maintenance.py verify-research`는 원 byte 수와 SHA-256으로 고정한 연구 ZIP의
CRC, `inventory.json`, 모든 member의 크기·SHA-256을 한 번에 검사한다. 목록 밖 파일,
중복 member·JSON key, link/비밀 이름, 손상·과도한 해제 크기는 거부한다. 이 명령은
자료 회수·native 종료·물리 완료를 증명하거나 원 자료를 삭제하지 않는다. 구형 inventory의
다른 필드 형식을 자동 변환하지 않으며, 당시 원문과 검증기를 그대로 보존한다.

`storage_maintenance.py audit-code-cache`는 아래 닫힌 plan의 원 byte 수·SHA-256을
받아 연구 ZIP과 공식 artifact의 nested TAR를 반복 가능한 방식으로 대조한다.
표준 카탈로그의 원 row digest·output inode·marker·종료·owner 사망을 확인하고,
held FD로 읽은 명시적 `selected4/` 또는 `private-elf/`의 제품 ELF만 대상으로 한다.
공식 archive는 추출하지 않고 stream으로 검사한다. 모델·runtime DSO·입력·workspace·
원시 결과를 재생성 코드로 분류하지 않는다.

```powershell
python scripts/storage_maintenance.py verify-research --archive <absolute-research-zip> --archive-bytes <exact-bytes> --archive-sha256 <original-sha256>
python scripts/storage_maintenance.py audit-code-cache --plan <absolute-plan-json> --plan-bytes <exact-bytes> --plan-sha256 <original-sha256>
```

plan의 필드는 `schema=rovezero.code-cache-audit.v1`, `catalog`, `key`,
`record_sha256`, `artifact`, `archive_member`, `code`, `research`다. `artifact`와
`research`는 `{path, bytes, sha256}`의 절대 경로 raw pin이다. `archive_member`는
공식 ZIP 안의 `native-validation-binaries.tar.gz` 같은 정확한 member다. `code`의
각 항목은 `{relative, tar_member, bytes, sha256}`이며 원 output 상대 경로와 공식
TAR의 정확한 member를 구분한다. basename만 같은 파일을 같은 자산으로 취급하지 않는다.

이 자동화는 **읽기 전용 검증**이다. 성공 응답에도 `retirement_authority_issued=false`와
`deletion_performed=false`를 남긴다. audit 결과 파일을 삭제 권한이나 만료 lease의
복구 증거로 사용하지 않는다. 실제 정리 직전의 원 custody 재검증, 현재 exe/FD/maps 사용
조사, 같은 held inode/name 대조와 durable 정리 journal은 기존 소유권별 처리기에 남는다.
기존 interrupted-unbound custody registry가 있으면 자동 채택하지 않고 즉시 거부한다.
따라서 현재 특수 custody 자료의 destructive apply는 이 명령에 자동 연결하지 않았다.

`retire-code-cache`는 같은 닫힌 plan을 받아 **Linux의 종료된 일반 관리 출력에 한해**
중복 ELF 정리를 명시적으로 실행한다. 이전 audit 결과를 권한으로 재사용하지 않고,
원 catalog gate와 행·marker·owner 종료 조건, 연구 ZIP, 공식 archive의 같은 byte/hash,
후보 inode와 현재 프로세스 참조를 실행 시 다시 검사한다. Windows 및 특수 custody,
active/interrupted/완료 불명 출력은 이 경로로 회복하거나 삭제하지 않는다.

```sh
python -I -S -B scripts/storage_maintenance.py retire-code-cache --plan <absolute-plan-json> --plan-bytes <exact-bytes> --plan-sha256 <original-sha256> --seconds 300
```

공식 archive와 연구 자료는 대상 출력 밖에 보존하며, 전체 검증과 사전 검사를 통과한
뒤에만 원 출력의 예약 안에 정리 journal을 남긴다. 삭제할 각 leaf의 held FD·이름·inode를
마지막에 확인하고 디렉터리 FD를 통해 삭제한다. 행·marker·예약량·보존 기간은 변경하지
않는다. 중간 오류에서는 journal과 이미 수행한 삭제를 보존하고 후속 삭제를 중단한다.
이름·크기만 같은 ELF, 다른 byte의 strip본, 모델·runtime·입력·원시 결과는 대상이 아니다.

프로세스 조사는 해당 Linux PID/mount 문맥에서의 유한한 관측이다. 접근 거부나 해석할
수 없는 상태를 미사용으로 처리하지 않으며, 외부 프로세스의 향후 접근까지 막는 전역
권한으로 표현하지 않는다. 삭제한 논리 byte 수와 실제 디스크 공간 반환은 구분한다.
물리 회수량과 VHD 반환량은 별도 관측 없이는 `unknown`이며, 이 명령은 WSL 종료나
가상 디스크 압축을 실행하지 않는다.

입력 수·압축 해제량과 요청한 1~600초의 내부 관측 창을 제한한다. OS I/O 중단까지의
hard timeout과 자식 트리 종료가 필요하면 기존 `run_managed.py` 감독 아래 실행한다.
새 예약을 늘리거나 카탈로그를 수정하지 않고, 보존 결과의 18GiB·14일 검토 정책,
만료가 삭제 권한이 아니라는 규칙, GPU physical unknown의 보존을 유지한다. 반복 실행의
검증 결과도 실제 source·plan·환경과 함께 보관하며 과거 성공을 현재 apply에 재사용하지 않는다.

## 새 출력 한정 preservation session

`OutputLease(..., preservation_session=PreservationSelection(...))`는 별도의 명시적
선택 정책이다. 기본 실행에는 활성화하지 않으며 기존 V1·transition 선택자 및 outer
finalizer와 함께 지정할 수 없다. 이 선택자는 역사 tip raw pin과 현재 source6 raw pin의
설정일 뿐, JSON·과거 영수증·선택자 객체 자체가 실행 권한을 부여하지 않는다.

기존 transition generation 1~8, `MAX_TRANSITIONS=8`, 현재 tip의 boot/source/mount를
요구하는 기존 validator는 그대로 둔다. 새 정책은 9번째 transition을 발행하거나 tip8을
현재 tip으로 바꾸지 않는다. 원 catalog row·원본 복사·snapshot·records1~8·physical
unknown은 읽기 전용으로 보존하며, 원 owner·삭제 권한·물리 완료를 재발급하지 않는다.

새 정책은 같은 canonical catalog의 원 OS gate 아래에서 역사 chain 전체를 검증한
뒤, 현재 boot·mount namespace·source6를 실제 held FD로 다시 관측한다. 역사 anchor와
새 출력에 한정한 `CurrentExclusionSession`은 별도 타입이다. 원 unknown subtree와 그
모든 ancestor/descendant는 새 출력 범위에서 제외한다. 원 reservation charge·registry
metadata·새 출력의 journal/payload를 동일한 18GiB 합계와 새 reserve에 포함한다.

현재 관측은 최대 30초이며 원 관측의 deadline을 늘리지 않는다. 다음 관측은 독립된
새 실제 검증으로 만들고 이전 관측을 닫는다. 이 갱신은 제품 요청·실험의 원 hard
deadline을 연장하지 않는다. boot/source/mount/FD 변경이나 증거 불명확은 거부하며,
기존 출력이나 다른 active row를 현재 producer로 채택하지 않는다.

새 출력은 현재 producer가 fresh 경로로 등록해야 한다. 비deferred 출력은 acquire가
반환되기 전에, deferred 출력은 원 `create_deferred_directory`가 반환되기 전에 필수
관측 증거를 저장한다. unbound deferred acquire는 metadata 예약일 뿐 payload 허용이
아니다. 저장·fsync·readback 실패 시 payload를 시작하지 않고 실패와 부분 자료를 보존한다.

관측 증거는 새 출력의 owner 전용 `.rovezero-preservation-session.jsonl` 한 파일에
append한다. 각 entry는 이전 entry의 digest·새 관측 digest·current source6·역사 tip·
현재 mount/identity·원 snapshot digest·reservation charge를 보존하고 fsync/readback한다.
파일은 원 producer의 실제 FD·inode·token에 묶으며 저장된 journal을 읽어 새 owner로
복원하지 않는다. 상한은 lease당 4,096관측·파일 16MiB·entry 1MiB다. 한도를 채우거나
partial append·파일 교체·다른 writer 흔적이 있으면 실패를 유지하며 자동 truncate,
rollover, 성공 보정, 원 자료 삭제를 하지 않는다.

새 정책으로 묶은 producer가 중단돼도 저장된 journal이나 active row를 다음 producer가
자동 채택하지 않는다. 현재 기존 recovery 경로는 기존 current-tip 조건을 계속 요구하므로,
새 정책의 중단 출력에 대한 별도 복구 인수가 끝나기 전에는 자료와 예약을 보존하고 차단한다.

이 정책 등록과 CPU fixture 성공은 실제 실행 승인·신경망·GPU 수명 인수를 대신하지
않는다. 실제 재관측과 제품 검증은 새 실행 ID·기존 자원/시간/보존 조건으로 별도 기록한다.

## 식별 연속성을 확인하지 못한 자료의 명시적 보존

기존 bound 출력의 기록된 device/inode와 현재 관측이 다르면 기본 경로는 계속
거절한다. inode 일치, 파일 내용 일치 또는 나중에 같은 device 번호가 관측된 사실만으로
과거 소유권이나 물리 객체의 연속성을 인정하지 않는다. 원 행의 identity를 다시 쓰지 않는다.

별도 `UnattributedSelection`은 원 catalog/marker와 독립 연구 ZIP을 명시한 여섯 자료를
미귀속 보존 대상으로만 처리하는 제한된 실험 경로다. 기존 `PreservationSelection`의
source6과 다른 source7 domain이며 자동 변환하지 않는다. 기본값은 계속 선택하지 않은
상태다. 이 타입 자체나 저장된 JSON은 owner·gate·종료·복구·삭제 권한이 아니다.

현재 실행의 실제 catalog gate·FD·boot·mount·원 절대 deadline 아래에서 원 행과 marker의
정확한 bytes, bounded corpus와 독립 ZIP의 모든 파일을 대조한다. ZIP은 중앙 디렉터리를
할당하기 전에 개수·크기를 제한하며, 이번 형식은 ZIP64와 분할 archive를 거절한다.
현재 corpus의 UID나 관측 identity는 원 owner의 증거로 사용하지 않는다. 원 reservation
이상과 회수 ZIP·registry·관측 문서·새 active reservation을 18GiB에서 보수적으로 계수한다.
범위를 맞추기 위해 원 예약을 줄이거나 실패 자료를 삭제하지 않는다.

새 producer는 기존 자료·회수본·control 경로와 분리된 fresh 출력만 등록한다. 관리 빌드의
acquire/prune 전에 cleanup root와 이 자료의 경로 및 현재 bind alias 겹침을 거절한다.
owner 전용 journal의 fsync/readback 이후에만 payload를 허용하며 기존 journal 상한을
유지한다. 원 여섯 행과 기존 transition1~8·MAX8·physical unknown은 그대로 보존한다.

`ManagedBuild(..., output_preservation=...)`와 `run_managed.main(output_preservation=...)`는
명시적으로 등록된 현재 호출자가 사용할 접점이다. 반복 budget observer는 producer가
보유한 gate를 기다리지 않으며 새로운 실행 권한을 발급하지 않는다. 비선택 active 행의
marker 중간쓰기는 budget 관측에 한해 구분하고, 실제 admission과 종료는 엄격한 gate
검사를 유지한다. 원 manager의 자식 종료 확인 뒤에만 별도 gate에서 종료를 처리한다.

소스·CPU fixture 검사는 실제 source7 준비와 원 tip/current-owner 연결 검사를 대신하지
않는다. 기존 source6/18을 현재 변경된 경로로 재열 수 없으면 새 producer를 명시 등록해야
하며, 역사 자료를 현재 소유권으로 재표시하지 않는다. 전체 corpus/ZIP을 다시 해시하는
초기 경로의 비용은 성능 개선 근거로 사용하지 않는다. 14일은 보존 검토 시점이며, 만료나
연구 ZIP 검증 완료만으로 원 자료를 삭제하지 않는다.

원 빌드 감독이 종료 증거를 남긴 뒤 출력 검사에서 멈춘 경우에는
`recover_build(..., output_preservation=...)`로 같은 typed 선택을 명시할 수 있다.
원 build gate·owner raw digest·root/slot identity와 기존 reboot 또는 정확한
closure/owner-dead 검증을 모두 유지한다. 이후 실제 output gate에서 현재 자료를
읽기 전용으로 관측한다. 과거 owner에 정확히 연결된 active supervised 출력이
하나라도 있으면 선택된 자료라도 복구를 거절하며 출력 행을 대신 종료하지 않는다.
현재 정책 소스와 회수본도 cache cleanup 범위 밖에 있어야 한다. 한 번의 30초 관측이
만료되면 갱신하지 않고 실패한다. 이 경로는 기존 재생성 cache/scratch의 복구만
허용하며 과거 source 권한·실행 owner·물리 완료·출력 삭제 권한을 발급하지 않는다.
