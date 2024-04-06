[private]
default:
    @just --list

vfm ns:
    LOGLEVEL=debug acies-vfm \
    --connect unixsock-stream//tmp/acies-mic.sock \
    --connect unixsock-stream//tmp/acies-geo.sock \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --weight "models/Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt"

push-weight:
    #!/usr/bin/env fish
    for i in rs1 rs2 rs3 rs5 rs6 rs6 rs7 rs8 rs10
        rsync -azvhP models/Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt $i.dev:/ws/acies/vehicle-classifier/models/Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt
    end
