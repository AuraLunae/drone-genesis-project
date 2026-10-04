"""
学習の「実行(ラン)」をタイムスタンプで管理するための共通ユーティリティ。

ディレクトリ構成:
    logs/{exp_name}/{run_id}/
        cfgs.pkl           # そのランで使った env_cfg 等一式(再現性・resume時の整合性のため)
        model_{iter}.pt     # rsl-rl が save_interval ごとに保存するチェックポイント
        # 中断・例外時の緊急保存は行わない。save_intervalの定期保存のみを正とする。

run_id はラン開始時刻から生成するタイムスタンプ("YYYYMMDD_HHMMSS")。
新規学習は必ず新しいrun_idのフォルダに保存されるため、既存のランを
上書き・削除することは構造的に起こらない。
"""

import glob
import os
import re
from datetime import datetime


_RUN_ID_PATTERN = re.compile(r"^\d{8}_\d{6}(?:_\d+)?$")  # 例: 20261004_211853 または 20261004_211853_1


def generate_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def new_run_dir(exp_name: str, logs_root: str = "logs") -> str:
    """新規学習用に、まだ存在しないタイムスタンプ付きディレクトリを作って返す"""
    run_id = generate_run_id()
    run_dir = os.path.join(logs_root, exp_name, run_id)
    # 同一秒内に2回起動される極端なケースの保険(ほぼ起きないが念のため)
    suffix = 0
    base_run_dir = run_dir
    while os.path.exists(run_dir):
        suffix += 1
        run_dir = f"{base_run_dir}_{suffix}"
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def find_latest_run_dir(exp_name: str, logs_root: str = "logs") -> str:
    """logs/{exp_name}/ 配下で、最も新しい(名前順で最大の) run_id ディレクトリを返す。

    rsl-rlが自動生成する"git"フォルダ(gitの差分を保存する場所)など、
    タイムスタンプ形式("YYYYMMDD_HHMMSS")に一致しないディレクトリは
    ランとして扱わず無視する。"""
    base = os.path.join(logs_root, exp_name)
    candidates = sorted(
        d for d in glob.glob(os.path.join(base, "*"))
        if os.path.isdir(d) and _RUN_ID_PATTERN.match(os.path.basename(d))
    )
    if not candidates:
        raise FileNotFoundError(f"'{base}' の下に学習済みのラン(タイムスタンプ形式のフォルダ)が見つかりません。")
    return candidates[-1]


def find_latest_checkpoint(run_dir: str) -> str:
    """run_dir内で最もイテレーション番号の大きい model_*.pt を返す
    (save_intervalごとの定期保存のみが対象。緊急保存は行っていない)。"""
    pattern = re.compile(r"model_(\d+)\.pt$")
    best_iter, best_path = -1, None
    for path in glob.glob(os.path.join(run_dir, "model_*.pt")):
        m = pattern.search(os.path.basename(path))
        if m:
            it = int(m.group(1))
            if it > best_iter:
                best_iter, best_path = it, path
    if best_path is None:
        raise FileNotFoundError(f"'{run_dir}' の中にチェックポイント(model_*.pt)が見つかりません。")
    return best_path


def resolve_resume_target(resume_arg: str, exp_name: str, logs_root: str = "logs"):
    """--resume に渡された値を (run_dir, checkpoint_path) に解決する。

    受け付ける形式:
      - "latest"                       : exp_name内で最新のランの、最新のチェックポイント
      - "logs/exp/20261004_120000"      : そのランディレクトリの、最新のチェックポイント
      - "logs/exp/20261004_120000/model_150.pt" : そのチェックポイントを直接指定
    """
    if resume_arg == "latest":
        run_dir = find_latest_run_dir(exp_name, logs_root)
        ckpt_path = find_latest_checkpoint(run_dir)
        return run_dir, ckpt_path

    if os.path.isdir(resume_arg):
        run_dir = resume_arg
        ckpt_path = find_latest_checkpoint(run_dir)
        return run_dir, ckpt_path

    if os.path.isfile(resume_arg):
        ckpt_path = resume_arg
        run_dir = os.path.dirname(resume_arg)
        return run_dir, ckpt_path

    raise FileNotFoundError(f"--resume に指定されたパスが見つかりません: {resume_arg}")