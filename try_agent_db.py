"""Ask the local model questions against the real PostgreSQL data (landmarks, noise, prices, ...).

Usage (Postgres and Ollama running, model pulled):
    .venv/Scripts/python.exe try_agent_db.py "Apartments within 500 m of Wien Westbahnhof for 4 guests"
    .venv/Scripts/python.exe try_agent_db.py            # runs the built-in example questions

The workflow returns tool evidence only (model prose is discarded), so this prints which tool the
model chose, the arguments that reached SQL, and the rows found.

Guards: the input guard and the evidence cleaner are the real ones. Set LLAMA_GUARD=1 to add Llama Guard.
"""

import os
import sys

import psycopg

from data_setup.load_vectors import Masker, clean_text
from guardrails.evidence_guard import remove_hostile_rows
from guardrails.input_guard import InputGuard, bge_m3_embedder, llama_guard
from langchain_agent.workflow import EvidenceWorkflow, local_model, shield_names

QUESTIONS = [
    'Which apartments are within 500 m of Wien Westbahnhof?',
    'Find places for 4 guests within 1000 m of Stephansdom that cost at most 150 euro.',
    'Show listings near Albertina where night noise is below 55 dB.',
    'Show listings near the Oper.',
]


def connect():
    dsn = os.getenv('DATABASE_URL', '')
    defaults = {} if dsn else {'host': os.getenv('PGHOST', '127.0.0.1'), 'port': os.getenv('PGPORT', '5433'),
                               'dbname': os.getenv('PGDATABASE', 'vienna_stay'),
                               'user': os.getenv('PGUSER', 'postgres'), 'password': os.getenv('PGPASSWORD', 'postgres')}
    return psycopg.connect(dsn, **defaults, autocommit=True)


def show(question, outcome):
    print(f'\n=== {question}\nstatus: {outcome["status"]}')
    for result in outcome['results']:
        print(f'- tool: {result["tool"]}')
        print(f'  arguments that reached SQL (last query): {result["evidence"][-1]["parameters"]}')
        print(f'  total_matches: {result.get("total_matches")}', end='')
        if 'landmark_found' in result:
            print(f' | landmark_found: {result["landmark_found"]}', end='')
        if 'missing_noise_exclusions' in result:
            print(f' | missing_noise_exclusions: {result["missing_noise_exclusions"]}', end='')
        print()
        if result.get('landmark_found') is False:
            print(f'  known landmarks: {", ".join(result["known_landmarks"])}')
        for row in result.get('rows', [])[:5]:
            print(f'    listing {row["listing_id"]:>10} {row["neighbourhood"]:<22} price {row["price"]}'
                  f' dist {row.get("distance_m")} m | road {row.get("lnight_road_band")} rail {row.get("lnight_rail_band")}')
        for removed in result.get('rows_removed_by_guard', []):
            print(f'  removed by evidence guard: {removed}')
        for notice in result.get('notices', []):
            print(f'  notice: {notice}')


def main(questions):
    sys.stdout.reconfigure(encoding='utf-8')
    print('Loading privacy models (German/English)...', flush=True)
    masker = Masker()
    input_guard = InputGuard(embed=bge_m3_embedder(), safety=llama_guard() if os.getenv('LLAMA_GUARD') == '1' else None)
    with connect() as connection:
        landmarks = [row[0] for row in connection.execute('SELECT name FROM vienna_stay_assistant.landmarks')]
        workflow = EvidenceWorkflow(
            connection, local_model(timeout=400),  # CPU inference is slow for large tool results
            # Without the shield the masker rewrites many landmark names to <PERSON_n>.
            mask=shield_names(lambda text: masker.mask(clean_text(text)), landmarks),
            input_guard=input_guard,
            evidence_guard=lambda masked_evidence: True,  # the cleaner below removes hostile rows
            clean_evidence=remove_hostile_rows,
        )
        for question in questions:
            show(question, workflow.invoke(question))
            print(f'  input guard: {input_guard.check(question)}')


if __name__ == '__main__':
    main(sys.argv[1:] or QUESTIONS)
