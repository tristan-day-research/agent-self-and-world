"""Human-readable run inventory. Regenerate after pull: python -m core.run_index."""
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import json

PACIFIC=ZoneInfo('America/Los_Angeles')

def pacific_time(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(PACIFIC).strftime('%Y-%m-%d %I:%M:%S %p %Z')

def build_index(root):
    root=Path(root)
    entries=[]
    for path in root.rglob('status.json'):
        try:
            status=json.loads(path.read_text())
            config=json.loads((path.parent/'config.json').read_text())
            if 'started_at' not in status: continue
            synthetic=bool(config.get('smoke')) or status.get('experiment')=='smoke'
            metadata_path=path.parent/'metadata.json'
            if metadata_path.exists():
                meta=json.loads(metadata_path.read_text())
                synthetic=synthetic or meta.get('synthetic',False) or str(meta.get('model','')).startswith(('random tiny','synthetic'))
            kind='SMOKE / synthetic — not research results' if synthetic else 'Research run'
            started=pacific_time(status['started_at'])
            finished=pacific_time(status['finished_at']) if status.get('finished_at') else 'Not recorded'
            label=config.get('run_label') or ('attention diagnostics' if config.get('attention') else status.get('experiment','unknown'))
            relative=path.parent.relative_to(root)
            info=f'# {label}\n\n**{kind}** · **{status["state"]}**\n\nStarted: {started}  \nFinished: {finished}\n\nExperiment: `{status.get("experiment")}`  \nStable run ID: `{status.get("run_id",path.parent.name)}`\n\n'
            info+='[Configuration](config.json) · [Status](status.json)'
            if (path.parent/'report.md').exists(): info+=' · [Authored research report](report.md)'
            info+='\n\nOne folder is one execution, not a copy of another folder. Smoke runs use tiny random models; failed runs may contain partial artifacts.\n'
            (path.parent/'RUN_INFO.md').write_text(info)
            entries.append({'started_at':status['started_at'],'start':started,'experiment':status.get('experiment'),
                'state':status['state'],'synthetic':synthetic,'label':label,'relative':relative,
                'report':(path.parent/'report.md').exists()})
        except (ValueError,KeyError,OSError):
            continue
    entries.sort(key=lambda r:r['started_at'],reverse=True)
    text='# Experiment results — Pacific time\n\nStart here after `pull-results`. Regenerate locally with `python -m core.run_index`.\n\n'
    text+='Each folder is one execution. **Research runs and synthetic smoke tests are different.** Timestamps below use America/Los_Angeles (PST/PDT automatically). Older folder names use UTC (`Z`); they remain unchanged to preserve links and avoid duplicate pulls. Future folders have readable Pacific timestamps; new smoke tests go under `_smoke/`.\n\n'
    for title,subset in [('Research runs',[r for r in entries if not r['synthetic']]),('Local smoke tests — no scientific conclusions',[r for r in entries if r['synthetic']])]:
        text+=f'## {title}\n\n| Started (Pacific) | Experiment / label | Status | Files | Report |\n|---|---|---|---|---|\n'
        for r in subset:
            rel=r['relative'].as_posix()
            text+=f'| {r["start"]} | {r["experiment"]} / {r["label"]} | {r["state"]} | [run]({rel}/RUN_INFO.md) | '+(f'[report]({rel}/report.md)' if r['report'] else '—')+' |\n'
        text+='\n'
    root.mkdir(parents=True,exist_ok=True)
    (root/'README.md').write_text(text)
    return entries

if __name__=='__main__':
    from core.runs import PROJECT_ROOT,RESULTS_SUBDIR
    entries=build_index(PROJECT_ROOT/RESULTS_SUBDIR)
    print(f'Indexed {len(entries)} runs in outputs/README.md (Pacific time)')
