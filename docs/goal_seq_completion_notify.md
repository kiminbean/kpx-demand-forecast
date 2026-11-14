# goal-seq 완료 자동 보고 (auto completion notify)

> 2026-11-14 설정·배포. 이 문서는 프로젝트 코드가 아니라 에이전트 운영 설정의 기록이다.
> 원본 스킬(절차·스크립트): 각 기기의 `~/.hermes/skills/software-development/goal-seq-rounds/`

## 목적

goal-seq 다중 라운드 작업이 끝나면 사용자가 물어봐야 알려주는 대신, 부모 세션이 완료를
감지하는 즉시 선제 보고(커밋/푸시 여부, 라운드 문서, 판정 포함)하도록 자동화한다.

## 구성

- `scripts/notify_completion.py` — 감시기. 용법: `python3 notify_completion.py <project_dir> [timeout_s]`
  - 감지 조건(이중화): `.goal-seq/state.json`의 `status == complete` OR `orchestrator.log`의 `ROUNDS COMPLETE`
  - `POLL_INTERVAL=30s`, 기본 timeout 86400s. 완료 시 요약 출력 + exit 0, 실패/타임아웃 exit 1.
  - git 호출은 `_git_cmd()`로 해결: `shutil.which("git")` 우선, 실패 시 `/usr/bin/git`
    (macOS Homebrew git 고장 폴백 — 리눅스/WSL에서도 which 성공 경로를 사용).
- `SKILL.md`의 "완료 자동 보고 (부모 세션)" 섹션 — 부모 세션 절차:
  goal-seq 시작 직후 `terminal(background=true, notify=true)`로 감시기 기동 →
  알림 도착 시 state.json·푸시 여부·라운드 문서를 검증한 뒤 선제 보고(질문 유도 금지).
- `references/completion_notify.md` — 상세 절차 참조.

## 3기기 배포 (2026-11-14, 전 기기 e2e 검증 완료)

| 기기 | 접속 | 배포 방식 | 검증 |
|---|---|---|---|
| 맥미니 (IBui-Macmini, 100.69.69.119) | 키 `~/.ssh/id_ed25519_to_macmini` | rsync 직접 배포 | e2e exit 0 |
| 우분투 (ubuntu-nucbox, 100.110.255.105) | 로컬 직접 키 전부 실패 → **맥미니 경유**(맥미니의 `id_rsa`), 사용자 kiminbean | 맥미니에서 rsync | e2e exit 0 |
| 윈도우 (win-inbeanwin) | cmd.exe에 python/hermes 없음 → **WSL Ubuntu**(/home/kimin, hermes v0.23.1) | scp로 `C:\Users\kimin` 스테이징 → WSL에서 `/mnt/c` 경유 cp | e2e exit 0 |

검증 방법: 각 기기에서 `status: complete` 픽스처 `.goal-seq/`를 만들어
`notify_completion.py`를 실실행 → "goal-seq 완료 감지" 요약 출력 + exit 0.

## 운영 팁 (교훈)

- 우분투·윈도우로 ssh 명령을 보낼 때는 중첩 인용 대신 **stdin 파이프 또는 base64**
  (`printf %s "$CMD" | ssh host "wsl -e bash"`)로 전달한다 — 인용 붕괴 방지.
- macOS에서 `tr -d '\r'`로 윈도우(CP949/CP932) 출력을 정화하면
  `Illegal byte sequence`가 난다 → `LC_ALL=C` 설정 또는 `cat -v` 우회.
- 배포 대상(`~/.hermes/skills/...`)은 repo 외부라 커밋 대상이 아니다. 이 문서는 그 기록만 담는다.
