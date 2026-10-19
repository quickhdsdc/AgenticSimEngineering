"""Write an offline example AAS blueprint; never contact a plant or repository."""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=ROOT/'components/harness/BaSyxMinimal/aas/festo-cp-lab.json')
    args = parser.parse_args()
    from a4s.aasctl.model import submodel_reference
    from a4s.machines.festo_cp_lab import fault_profile, maintenance, model

    document = model.build()
    # These are fictional local MES references, not deployed resources.
    context = maintenance.production_context('A4S-FESTO', 'A4S-BACKUP',
                                             'A4S-DEMO-PART', 'http://localhost:8092')
    business = fault_profile.models(context, model.SHELL_ID)
    for submodel in business:
        if submodel['idShort'] not in {'ProductionContext', 'MaintenanceInstructions'}:
            continue
        submodel['kind'] = 'Template'
        submodel['qualifiers'] = [{'type':'A4S/Instantiation/KeepConfiguredValues',
                                  'valueType':'xs:boolean', 'value':'true'}]
        document['submodels'].append(submodel)
        document['assetAdministrationShells'][0]['submodels'].append(submodel_reference(submodel['id']))
    # The public example must never embed this workstation's plant bindings.
    encoded = json.dumps(document, ensure_ascii=False, indent=2)
    for spec in model.SERVERS.values():
        for field, example in (('endpoint', '192.0.2.10'), ('internal', '198.51.100.10')):
            host = urlsplit(spec.get(field, '')).hostname
            if host:
                encoded = encoded.replace(host, example)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(encoded, encoding='utf-8')
    print(json.dumps({'status':'written', 'path':str(args.out),
                      'scope':'offline fictional example; configure deployment mappings separately'}))


if __name__ == '__main__':
    main()
