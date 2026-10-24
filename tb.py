"""
tb.py

TensorBoardを、オプション無しで起動する。

`tensorboard`コマンドは --logdir が必須で、オプション無しでは起動できない。
学習中は train.py が自動で立ち上げるが、学習が終わった後に過去のランを
見返したい時は、このスクリプトを使う。

  uv run python tb.py                  # logs/ 全体(全実験・全ランをまとめて表示)
  uv run python tb.py wind-hovering    # 特定の実験だけ

終了は Ctrl+C。
"""

import os
import sys

from run_utils import launch_tensorboard


def main():
    logdir = os.path.join("logs", sys.argv[1]) if len(sys.argv) > 1 else "logs"
    if not os.path.isdir(logdir):
        print(f"ログフォルダが見つかりません: {logdir}")
        return
    process, _ = launch_tensorboard(logdir)
    try:
        process.wait()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
