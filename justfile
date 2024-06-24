[private]
default:
    @just --list

# delete all *.log files
clean:
    rm -f *.log

# namespace (hostname) of the node
ns := `hostname -s`

# Zenoh Router endpoint
zrouter := 'tcp/10.8.0.3:7447'

# VibroFM weights
vfm-weight-2 := "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_best.pt"
vfm-weight-geo := "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_seismic_best.pt"
vfm-weight-mic := "models/demo2024_Parkland_TransformerV4_vehicle_classification_1.0_finetune_yizhuoict15_audio_best.pt"

# FreqMAE weights
mae-weight-2 := "models/demo2024_freqmae_Parkland_TransformerV4_vehicle_classification_1.0_finetune_best.pt"

# launch a VFM classifier
vfm:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect {{zrouter}} \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --feature-twin \
    --weight {{vfm-weight-2}}

# start a digital twin of the VFM classifier
twin-vfm twin-model="multimodal" twin-buff-len="2":
    LOGLEVEL=debug rye run acies-vfm \
    --connect {{zrouter}} \
    --namespace twin/{{ns}} \
    --twin-model {{twin-model}} \
    --twin-buff-len {{twin-buff-len}} \
    --proc_name vfm_{{twin-model}}_{{twin-buff-len}} \
    --topic twin/{{ns}}/geo \
    --topic twin/{{ns}}/mic \
    --weight {{ if twin-model == "multimodal" { vfm-weight-2 } else if twin-model == "seismic" { vfm-weight-geo } else if twin-model == "acoustic" { vfm-weight-mic } else { vfm-weight-2 } }}

# start a digital twin of the VFM classifier as background process
nohup-twin-vfm twin-model="multimodal" twin-buff-len="2":
    nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm {{twin-model}} {{twin-buff-len}} > /dev/null 2>&1 &

# launch a VFM classifier in deactivated mode
deactivated-vfm:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect {{zrouter}} \
    --namespace {{ns}} \
    --proc_name vfm \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --deactivated \
    --weight {{vfm-weight-2}}

# launch a backup VFM classifier
backup-vfm ns_primary:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect {{zrouter}} \
    --namespace {{ns}} \
    --proc_name "backup/{{ns_primary}}/vfm" \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight {{vfm-weight-2}}

# launch a backup VFM-geo classifier
backup-vfm-geo ns_primary:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect {{zrouter}} \
    --namespace {{ns}} \
    --proc_name "backup/{{ns_primary}}/vfm_geo" \
    --modality 'seismic' \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight {{vfm-weight-geo}}

# launch a backup VFM-mic classifier
backup-vfm-mic ns_primary:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns_primary}}_acies-geo.sock \
    --connect {{zrouter}} \
    --namespace {{ns}} \
    --proc_name "backup/{{ns_primary}}/vfm_mic" \
    --modality 'audio' \
    --topic {{ns_primary}}/geo \
    --topic {{ns_primary}}/mic \
    --deactivated \
    --weight {{vfm-weight-mic}}

# launch a FreqMAE classifier
mae:
    LOGLEVEL=debug rye run acies-vfm \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect {{zrouter}} \
    --namespace {{ns}} \
    --proc_name mae \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --weight {{mae-weight-2}}

# launch a DeepSense classifier
ds:
    LOGLEVEL=debug rye run acies-ds \
    --connect unixsock-stream//tmp/{{ns}}_acies-mic.sock \
    --connect unixsock-stream//tmp/{{ns}}_acies-geo.sock \
    --connect {{zrouter}} \
    --namespace {{ns}} \
    --proc_name ds \
    --topic {{ns}}/geo \
    --topic {{ns}}/mic \
    --model-config "models/demo2024_deepsense_multi.yaml" \
    --weight "models/demo2024_deepsense_train_best.pt"

# sync weights to the fleet
push-weight:
    #!/usr/bin/env fish
    for i in rs1 rs2 rs3 rs5 rs6 rs6 rs7 rs8 rs10
        rsync -azvhP models/demo2024* $i.dev:/ws/acies/vehicle-classifier/models
    end

# start the primary and backup classifiers on this node
start:
    just ns={{ns}} zrouter={{zrouter}} {{ns}}

# end all classifiers on this node
end:
    pkill -SIGINT -f 'acies-vfm'

# primary and backup services on rs1
rs1:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs2 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs1 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs1 > nohup_vfm_mic.log 2>&1 &

# primary and backup services on rs2
rs2:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs3 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs2 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs2 > nohup_vfm_mic.log 2>&1 &

# primary and backup services on rs3
rs3:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs5 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs3 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs3 > nohup_vfm_mic.log 2>&1 &

# primary and backup services on rs5
rs5:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs6 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs5 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs5 > nohup_vfm_mic.log 2>&1 &

# primary and backup services on rs6
rs6:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs7 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs6 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs6 > nohup_vfm_mic.log 2>&1 &

# primary and backup services on rs7
rs7:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs8 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs7 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs7 > nohup_vfm_mic.log 2>&1 &

# primary and backup services on rs8
rs8:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs10 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs8 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs8 > nohup_vfm_mic.log 2>&1 &

# primary and backup services on rs10
rs10:
    nohup just ns={{ns}} zrouter={{zrouter}} vfm > nohup_vfm.log 2>&1 &
    # nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs1 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm rs6 > nohup_vfm.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-geo rs10 > nohup_vfm_geo.log 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} backup-vfm-mic rs10 > nohup_vfm_mic.log 2>&1 &

# continuously monitor the <log> file for <pat>
tail log filter="detected":
    tail -F {{log}} | grep "{{filter}}" --line-buffered | cut -d']' -f3-

# start digital twins with different buffer lengths
launch-twins-latency:
    nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm multimodal 1 >/dev/null 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm multimodal 2 >/dev/null 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm multimodal 3 >/dev/null 2>&1 &
    # nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm multimodal 4 >/dev/null 2>&1 &
    # nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm multimodal 5 >/dev/null 2>&1 &

# start digital twins with different modalities
launch-twins-modalities:
    nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm multimodal 1 >/dev/null 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm seismic 1 >/dev/null 2>&1 &
    nohup just ns={{ns}} zrouter={{zrouter}} twin-vfm acoustic 1 >/dev/null 2>&1 &

alias k := kill

# kill a process by name
kill pat="acies-vfm":
    pkill -f {{pat}} -SIGINT

# list all processes matching the pattern
ps pat="acies-vfm":
    ps aux | grep {{pat}}
