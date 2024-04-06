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
    --weight "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt"

ds ns:
    LOGLEVEL=debug acies-ds \
    --connect unixsock-stream//tmp/acies-mic.sock \
    --connect unixsock-stream//tmp/acies-geo.sock \
    --namespace {{ns}} \
    --proc_name ds \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --model-config "models/demo2024_deepsense_multi.yaml" \
    --weight "models/demo2024_deepsense_train_best.pt"

push-weight:
    #!/usr/bin/env fish
    for i in rs1 rs2 rs3 rs5 rs6 rs6 rs7 rs8 rs10
        rsync -azvhP models/demo2024* $i.dev:/ws/acies/vehicle-classifier/models
    end
