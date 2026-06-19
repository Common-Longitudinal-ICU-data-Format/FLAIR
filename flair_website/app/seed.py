"""Bootstrap CLI: create the DB and seed one site + its lead user.

  uv run python -m app.seed --key mimic --email lead@mimic.org --password secret \\
      --site-name "Beth Israel Deaconess" --abbrev BIDMC --pi "Dr. Example"

Run once per site. Site keys should match the metric snapshot keys so the
leaderboard can join profile ↔ numbers. With no args it seeds every snapshot
site with a demo login (DEMO ONLY)."""
from __future__ import annotations

import argparse
import string

from . import auth, db
from .data import snapshot_site_keys


def _anon_label(index: int) -> str:
    # 0->A, 1->B, ... 25->Z, 26->AA
    label = ""
    n = index
    while True:
        label = string.ascii_uppercase[n % 26] + label
        n = n // 26 - 1
        if n < 0:
            break
    return label


def seed_site(key: str, email: str, password: str, *, anon_label: str,
              site_name: str = "", abbrev: str = "", pi: str = "",
              opted_in: bool = False) -> None:
    site = db.get_site_by_key(key)
    if site:
        print(f"  site '{key}' already exists (id={site['id']}), skipping")
        return
    site_id = db.create_site(key, anon_label)
    db.create_user(site_id, email, auth.hash_password(password))
    if site_name or abbrev or pi or opted_in:
        s = db.get_site(site_id)
        db.update_profile(
            site_id, site_name=site_name, abbreviation=abbrev, pi_name=pi,
            contact=email, opted_in=opted_in, visibility=s["visibility"],
        )
    print(f"  seeded site '{key}' (label {anon_label}) + lead {email}")


def seed_admin(email: str, password: str) -> None:
    """Committee admin lives on a hidden 'committee' site (never on the board)."""
    if db.get_user_by_email(email):
        print(f"  admin {email} already exists, skipping")
        return
    site = db.get_site_by_key("committee")
    site_id = site["id"] if site else db.create_site("committee", "—")
    db.create_user(site_id, email, auth.hash_password(password), role="admin")
    print(f"  seeded committee admin {email}")


def seed_example_submission() -> None:
    if db.list_submissions():
        return
    db.create_submission(
        model_name="ICU-LSTM v0.3",
        model_type="recurrent neural net (sequence)",
        owner_name="Dr. Ada Researcher",
        owner_email="ada@exampleuniv.edu",
        is_site_lead=False,
        github_url="https://github.com/example/icu-lstm",
        docker_url="docker.io/example/icu-lstm:0.3",
        model_url="https://huggingface.co/example/icu-lstm",
        tasks=["task1_icu_daily_mortality", "task4_sepsis_abx_6h"],
        target_sites="all",
        wants_site_reg=True,
        reg_site_name="Example University Hospital",
        reg_abbrev="EUH",
        reg_pi="Dr. Ada Researcher",
        notes="Pretrained on eICU; requesting external validation across CLIF sites.",
        tested_mimic=True,
    )
    print("  seeded example submission")


def seed_demo_runs() -> None:
    """Post the example submission to sites and seed a few varied responses so the
    committee orchestration matrix is populated (DEMO ONLY)."""
    if db.list_run_requests(None):
        return
    subs = db.list_submissions()
    if not subs:
        return
    sub = subs[0]
    db.push_submission_to_sites(sub)
    db.update_submission_status(sub["id"], "posted",
                                "Audited on MIMIC-CLIF; posted to all sites.")
    reqs = db.list_run_requests(None)
    if not reqs:
        return
    rid = reqs[0]["id"]
    demo_states = ["results-shared", "running", "accepted", "rejected"]
    sites = [s for s in db.list_sites() if s["key"] != "committee"]
    for i, s in enumerate(sites):
        db.set_run_response(rid, s["id"], status=demo_states[i % len(demo_states)],
                            anonymous=(i % 2 == 0))
    print("  seeded demo run + responses")


def main() -> None:
    p = argparse.ArgumentParser(description="Seed FLAIR site + lead user")
    p.add_argument("--key", help="site key (match snapshot key)")
    p.add_argument("--email")
    p.add_argument("--password")
    p.add_argument("--site-name", default="")
    p.add_argument("--abbrev", default="")
    p.add_argument("--pi", default="")
    p.add_argument("--opted-in", action="store_true")
    p.add_argument("--admin", action="store_true", help="create a committee admin user")
    p.add_argument("--demo", action="store_true",
                   help="seed every snapshot site + admin + example submission (DEMO ONLY)")
    args = p.parse_args()

    db.init_db()

    if args.admin and not args.demo:
        if not (args.email and args.password):
            p.error("--admin requires --email and --password")
        seed_admin(args.email, args.password)
        return

    if args.demo or not args.key:
        print("Seeding demo sites (password='demo', emails lead@<key>.flair):")
        for i, key in enumerate(snapshot_site_keys()):
            seed_site(
                key, f"lead@{key}.flair", "demo", anon_label=_anon_label(i),
                opted_in=(key == "mimic"),
                site_name=("MIMIC-CLIF (public)" if key == "mimic" else ""),
                abbrev=("MIMIC" if key == "mimic" else ""),
            )
        seed_admin("admin@flair", "demo")
        seed_example_submission()
        seed_demo_runs()
        return

    # Single explicit site: pick the next free anon label.
    existing = len(db.list_sites())
    seed_site(
        args.key, args.email, args.password, anon_label=_anon_label(existing),
        site_name=args.site_name, abbrev=args.abbrev, pi=args.pi,
        opted_in=args.opted_in,
    )


if __name__ == "__main__":
    main()
