"""Reproduce the bounded 2026-10-19 revision outputs from existing local inputs.
Run from code/: python3 run_paper1_revision.py
No raw-data retrieval, model encoding, or original-result overwrites.
"""
import concurrent.futures
from pathlib import Path
import subprocess
import sys

LOG=Path('../revision_work/2026-10-19');LOG.mkdir(parents=True,exist_ok=True)
def run(job):
 name,args=job
 with (LOG/(name+'.log')).open('w') as f:
  result=subprocess.run([sys.executable,*args],stdout=f,stderr=subprocess.STDOUT)
 if result.returncode:raise RuntimeError(f'{name} failed: see {LOG/(name+".log")}')
 print(name+' completed',flush=True)

def main():
 subprocess.run([sys.executable,'-m','unittest','test_revision_repairs','-v'],check=True)
 jobs=[]
 for case in ['ego_depletion','ptsd_trauma']:
  cfg=f'configs/{case}_revision_20261019.yaml'
  terms='ego depletion,strength model,self control,self regulation' if case=='ego_depletion' else 'trauma,ptsd,posttraumatic stress disorder,trauma informed,secondary traumatic stress,vicarious trauma,complex ptsd'
  jobs += [(case+'_instruments',['28_instrument_extraction.py','--config',cfg]),(case+'_correspondence',['30_correspondence_drift.py','--config',cfg,'--terms',terms])]
  for null in ['venue','journal','degree']:
   args=['27_citation_communities.py','--config',cfg,'--flows-only','--null',null]
   if case=='ptsd_trauma':args+=['--focus','trauma,PTSD,trauma informed,secondary traumatic stress,vicarious trauma,complex PTSD']
   jobs.append((case+'_citation_'+null,args))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:list(ex.map(run,jobs))
 # Sequential per case avoids overwriting shared manifest names during the cross-set run.
 run(('historical_correspondence',['30_correspondence_drift.py','--config','configs/ptsd_trauma_revision_20261019.yaml','--cross','ANC=traumatic neurosis,war neurosis,combat neurosis,combat fatigue,battle fatigue,shell shock,gross stress reaction,postvietnam syndrome,rape trauma syndrome@1965-1979;PTSD80=PTSD@1980-1989;PTSD90=PTSD@1990-1994;BER=bereavement@1965-1979','--compare','ANC:PTSD80<BER:PTSD80','--reps','50','--min-n','30']))
 for case in ['ego_depletion','ptsd_trauma']:
  run((case+'_audit',['32_revision_audit.py','--config',f'configs/{case}_revision_20261019.yaml']))

if __name__=='__main__':main()
