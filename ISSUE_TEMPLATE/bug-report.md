# 🐛 Bug Report Template

**IMPORTANT:** Replace all `<!-- -->` comments with actual details. Delete sections that don't apply.

## 1. Environment Context
```plaintext
- velonk Version: <!-- git sha or release tag -->
- Deployment Type: [Kubernetes | Docker-Compose | Bare-Metal]
- Cloud Provider: [AWS/GCP/Azure | On-Premise]
- Agent Count: <!-- Number of active agents when bug occurred -->
- Modules Affected: <!-- Orchestrator/RL Engine/Comms Hub/etc. -->
- Kubernetes (if applicable):
  - Version: <!-- kubectl version -->
  - Node OS: <!-- uname -a -->
  - Storage Class: <!-- e.g., ebs-sc -->
```

## 2. Defect Characterization
### Observed Behavior
#### Error Signature:
```
# Paste relevant error stack trace
Traceback (most recent call last):
File "/path/to/module.py", line X, in <module>
```
#### Expected Behavior

## 3. Reproduction Protocol
### Minimal Reproduction Case (MRC)
```
# Step 1: Deployment command
helm install velonk --set agent.profile=minimal_failure_case

# Step 2: Trigger action
curl -X POST http://orchestrator/api/v1/trigger --data '{"test_case": "<!-- JSON payload -->"}'

# Step 3: Error manifestation point
<!-- Describe where failure becomes visible -->
```

### Reproduction Rate
- [ ] 100% Consistent
- [ ] Intermittent (Frequency: ~ )
- [ ] Edge Case (Specific conditions: )

## 4. Diagnostic Evidence
### Log Snippets
```
<!-- From orchestrator -->
2023-10-05 12:34:56 ERROR [AgentRouter] CID=0xFEEDFACE: Failed to route message <!-- include correlation IDs -->

<!-- From individual agent -->
{"level":"ERROR","ts":1696523696.123,"caller":"agent/core.go:227","msg":"Consensus failure","swarm_id":"SW-001","last_valid_state_hash":"a1b2c3d4"}
```

### Metrics Anomalies
```
- CPU Spike: Node-X jumped from 15% → 90% at 12:34:56 UTC
- Network: 500MB/s outbound traffic (baseline: 50MB/s)
- Memory Leak: RSS growth rate 2MB/s in AgentPool
```

## 5. Forensic Artifacts
- [ ] Attached core dump: core.aelion.<PID>
- [ ] Network capture: tcpdump.pcap
- [ ] Profiler output: pprof.svg
- [ ] Jaeger trace: trace-<ID>.json

## 6. Mitigation History
- Tried increasing agent_heartbeat_timeout from 30s → 60s: ❌ No change
- Disabled RL reward shaping: ⚠️ Reduced error rate by 40%
- Rollback to v1.2.3: ✅ Resolved immediately

## 7. Severity Assessment
- Impact: [Critical | High | Medium | Low]
- Priority: [P0 | P1 | P2 | P3]
- Security Risk: [CVE Potential | Data Exposure | None]
