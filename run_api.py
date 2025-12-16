#!/usr/bin/env python
"""
J-REF API Runner
=================

제주도 신재생에너지 발전량 예측 API 실행 스크립트

Usage:
    python run_api.py                   # 기본 실행 (8000 포트)
    python run_api.py --port 8080       # 포트 변경
    python run_api.py --reload          # 개발 모드 (자동 리로드)
    python run_api.py --debug           # 디버그 모드
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="J-REF API - 제주도 신재생에너지 발전량 예측 API"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="바인드 호스트 (기본: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="포트 번호 (기본: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="자동 리로드 모드 (개발용)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="디버그 모드"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="워커 프로세스 수 (기본: 1)"
    )

    args = parser.parse_args()

    # Set environment variables for debug mode
    if args.debug:
        import os
        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"

    print("=" * 60)
    print("J-REF API - 제주도 신재생에너지 발전량 예측")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"Debug: {args.debug}")
    print(f"Workers: {args.workers}")
    print("=" * 60)
    print(f"API Docs: http://localhost:{args.port}/docs")
    print(f"ReDoc: http://localhost:{args.port}/redoc")
    print("=" * 60)

    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
