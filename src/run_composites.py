'''
######################################################################
## ARISTOTLE UNIVERSITY OF THESSALONIKI
## PERSLAB
## REMOTE SENSING AND EARTH OBSERVATION TEAM
##
## DATE:             Aug-2025
## SCRIPT:           run_composites.py
## AUTHOR:           Vangelis Fotakidis (fotakidis@topo.auth.gr)
##
## DESCRIPTION:      Script to run CLI subprocesses of the `composites.py` module  >>> (venv) python run_composites.py
##
#######################################################################
'''

from utils.utils import setup_logger, generate_geojson_files_for_composites, mkdir

import datetime, pytz
import gc, os, sys, time
import json

from pathlib import Path

import subprocess

if __name__ == "__main__":   
    # Set up logger.
    mkdir("../logs/compgen")
    log = setup_logger(logger_name='admin_compgen_',
                        logger_path=f'../logs/compgen/admin_compgen_{datetime.datetime.now(pytz.timezone("Europe/Athens")).strftime("%Y%m%dT%H%M%S")}.log', 
                        logger_format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
                        )
    
    # 1) generate/refresh the .geojson tasks
    geojson_path = "../geojsons/compgen"
    generate_geojson_files_for_composites(
        output_dir=geojson_path,
        tile_geojson_filepath='../anciliary/grid_20_v2.geojson',
        start_date=datetime.datetime(2020, 1, 1),
        end_date=datetime.datetime(2025, 9, 1),
    )
    
    # 2) collect tasks
    if os.path.isdir(geojson_path):
        geojson_files = sorted(
            os.path.join(geojson_path, f)
            for f in os.listdir(geojson_path)
            if f.endswith(".geojson")
        )
    else:
        geojson_files = [geojson_path] 
        
    done_file = Path("../logs/compgen/admin_completed_geojsons.txt")
    already_done = set()
    if done_file.exists():
        already_done = set(x for x in done_file.read_text().splitlines() if x)

    # 3) run each in a fresh interpreter, sequentially
    for i, gf in enumerate(geojson_files, 1):
        if gf in already_done:
            log.info(f"Skip already completed: {gf} [{i}/{len(geojson_files)}]")
            continue

        log.info(f"[>] Launching single-shot: {gf} [{i}/{len(geojson_files)}]")
        
        rc = subprocess.run(
            [sys.executable, "composites.py", "--geojson", gf],
            check=False,
        ).returncode

        if rc == 0:
            with done_file.open("a", encoding="utf-8") as df:
                df.write(gf + "\n")
            log.info(f"✔ Processed {gf} | [{i} / {len(geojson_files)}] ({round(100*((i)/len(geojson_files)),2)}%)")
        else:
            log.error(f"✖ Failed {gf} with exit code {rc} | [{i} / {len(geojson_files)}] ({round(100*((i)/len(geojson_files)),2)}%)")
            # optional small backoff to avoid rapid-fire restarts on a flaky machine
            time.sleep(2)
        
    # ---------------- OWS UPDATE ----------------
    try:
        log.info("Triggering OWS update")

        # 1) run datacube-ows-update --views
        rc_views = subprocess.run(
            ["docker", "exec", "drought-drought_ows-1", "bash", "-lc", "datacube-ows-update --views"],
            check=False
        ).returncode

        if rc_views != 0:
            log.error(f"datacube-ows-update --views failed (rc={rc_views})")

        # 2) run datacube-ows-update
        rc_main = subprocess.run(
            ["docker", "exec", "drought-drought_ows-1", "bash", "-lc", "datacube-ows-update"],
            check=False
        ).returncode

        if rc_main != 0:
            log.error(f"datacube-ows-update failed (rc={rc_main})")
        else:
            log.info("OWS update completed successfully.")

    except Exception as e:
        log.exception(f"Error while updating OWS: {e}")
    # ------------------------------------------------