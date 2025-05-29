#!/usr/bin/env bash
set -eo pipefail

# ==============================================
# Agent Service Upgrade Script - Zero Downtime
# ==============================================

# Configuration
DEPLOY_NAMESPACE="biconic-agents"
CLUSTER_ENV="prod"                         # prod|staging
UPGRADE_STRATEGY="rolling"                 # rolling|blue-green|canary
ROLLING_UPDATE_PARAMS="maxSurge=25%,maxUnavailable=0"
TIMEOUT=600                                # seconds
HEALTH_CHECK_INTERVAL=10                   # seconds
MAX_ATTEMPTS=30
VAULT_ADDR="https://vault.biconic.ai:8200"
BACKUP_DIR="/mnt/biconic/backups/$(date +%Y%m%d-%H%M%S)"
LOG_FILE="/var/log/agent-upgrade.log"
LOCK_FILE="/tmp/agent-upgrade.lock"

# Version Management
OLD_VERSION="${1:?Missing current version}"
NEW_VERSION="${2:?Missing target version}"
VERSION_SKEW=2                             # Max allowed version difference

# Dependency Services
MIN_KUBERNETES="1.25"
MIN_ETCD="3.5"
API_COMPAT_WINDOW="+30d"                  # API compatibility window

# Initialize logging
exec > >(tee -a "${LOG_FILE}") 2>&1

# Phase 1: Pre-upgrade Validation
function validate_upgrade() {
    echo "=== STARTING PRE-UPGRADE VALIDATION ==="
    
    # Check version compatibility
    if ! semver-cli compare "${NEW_VERSION}" -r ">${OLD_VERSION}"; then
        echo "ERROR: Invalid version progression"
        exit 1
    fi

    # Verify version skew
    version_diff=$(( ${NEW_VERSION##*.} - ${OLD_VERSION##*.} ))
    if [[ ${version_diff} -gt ${VERSION_SKEW} ]]; then
        echo "ERROR: Version skew too large (${version_diff} > ${VERSION_SKEW})"
        exit 1
    fi

    # Check cluster prerequisites
    verify_k8s_version
    verify_etcd_version
    verify_storage_capacity
    verify_network_policies

    # Validate image signature
    cosign verify --key k8s://${DEPLOY_NAMESPACE}/aelion-signing-key \
        aelion-registry.ai/agent:${NEW_VERSION}

    # Check API compatibility
    if ! diff -u \
        <(curl -sS "https://api-spec.aelion.ai/${OLD_VERSION}/openapi.json") \
        <(curl -sS "https://api-spec.aelion.ai/${NEW_VERSION}/openapi.json") \
        --ignore-matching-lines='.*version.*' >/tmp/api.diff; then
        echo "WARNING: API changes detected:"
        cat /tmp/api.diff
        read -p "Confirm API compatibility (y/n)? " -n 1 -r
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    fi

    # Create upgrade lock
    if [[ -f "${LOCK_FILE}" ]]; then
        echo "ERROR: Upgrade already in progress"
        exit 1
    fi
    touch "${LOCK_FILE}"
}

function verify_k8s_version() {
    current_k8s=$(kubectl version -o json | jq -r .serverVersion.gitVersion)
    if ! semver-cli compare "${current_k8s}" -r ">=${MIN_KUBERNETES}"; then
        echo "ERROR: Kubernetes version ${current_k8s} < ${MIN_KUBERNETES}"
        exit 1
    fi
}

# Phase 2: Pre-upgrade Backup
function create_backups() {
    echo "=== CREATING PRE-UPGRADE BACKUPS ==="
    
    mkdir -p "${BACKUP_DIR}"
    
    # Backup Kubernetes resources
    kubectl get all,configmaps,secrets -n ${DEPLOY_NAMESPACE} -o yaml \
        > "${BACKUP_DIR}/resources.yaml"
    
    # Backup persistent volumes
    for pvc in $(kubectl get pvc -n ${DEPLOY_NAMESPACE} -o name); do
        kubectl exec -n ${DEPLOY_NAMESPACE} ${pvc/-*/} -- \
            tar czf - /data > "${BACKUP_DIR}/$(basename ${pvc}).tgz"
    done

    # ETCD backup
    kubectl exec -n kube-system etcd-$(hostname) -- \
        etcdctl snapshot save /var/lib/etcd/backup.db
    kubectl cp -n kube-system etcd-$(hostname):/var/lib/etcd/backup.db \
        "${BACKUP_DIR}/etcd.db"

    # Encrypt backups
    age -R "${VAULT_ADDR}/v1/aelion/backup-key" \
        -o "${BACKUP_DIR}.age" "${BACKUP_DIR}"
}

# Phase 3: Rolling Upgrade Execution
function perform_upgrade() {
    echo "=== STARTING ROLLING UPGRADE ==="
    
    # Annotate deployment for audit
    kubectl annotate deploy/agent -n ${DEPLOY_NAMESPACE} \
        aelion.io/previous-version="${OLD_VERSION}" \
        aelion.io/upgrade-timestamp="$(date -u +%s)"

    # Update deployment
    kubectl set image -n ${DEPLOY_NAMESPACE} deploy/agent \
        "*=aelion-registry.ai/agent:${NEW_VERSION}" \
        --record
    
    kubectl rollout status -n ${DEPLOY_NAMESPACE} deploy/agent \
        --timeout=${TIMEOUT}s
    
    # Verify quorum
    check_cluster_quorum
    verify_service_mesh
    validate_metrics_ingestion
}

function check_cluster_quorum() {
    echo "--- Verifying Agent Quorum ---"
    expected_replicas=$(kubectl get deploy/agent -n ${DEPLOY_NAMESPACE} \
        -o jsonpath='{.spec.replicas}')
    actual_replicas=$(kubectl get pods -n ${DEPLOY_NAMESPACE} -l app=agent \
        -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' | wc -w)
    
    if [[ ${actual_replicas} -ne ${expected_replicas} ]]; then
        echo "ERROR: Quorum lost (${actual_replicas}/${expected_replicas})"
        rollback_upgrade
    fi
}

# Phase 4: Post-upgrade Verification
function validate_upgrade() {
    echo "=== VALIDATING UPGRADE ==="
    
    # Verify version consistency
    deployed_version=$(kubectl get deploy/agent -n ${DEPLOY_NAMESPACE} \
        -o jsonpath='{.spec.template.spec.containers[0].image}')
    if [[ "${deployed_version}" != *"${NEW_VERSION}" ]]; then
        echo "ERROR: Version mismatch after upgrade"
        rollback_upgrade
    fi

    # Run smoke tests
    perform_api_validation
    validate_data_pipeline
    check_audit_logs
    
    # Check resource utilization
    if ! kubectl top pods -n ${DEPLOY_NAMESPACE} | awk '\$2 > 90'; then
        echo "ERROR: Resource overutilization detected"
        rollback_upgrade
    fi
}

# Phase 5: Rollback Mechanism
function rollback_upgrade() {
    echo "=== INITIATING ROLLBACK ==="
    
    kubectl rollout undo -n ${DEPLOY_NAMESPACE} deploy/agent \
        --to-revision=$(kubectl rollout history deploy/agent -n ${DEPLOY_NAMESPACE} \
        | grep "${OLD_VERSION}" | awk 'NR==1{print \$1}')
    
    kubectl rollout status -n ${DEPLOY_NAMESPACE} deploy/agent \
        --timeout=${TIMEOUT}s
    
    # Restore backups if needed
    if [[ -d "${BACKUP_DIR}" ]]; then
        age -d -i "${VAULT_ADDR}/v1/aelion/backup-key" "${BACKUP_DIR}.age" | tar xz
        kubectl apply -f "${BACKUP_DIR}/resources.yaml"
    fi
    
    exit 1
}

# Phase 6: Cleanup
function cleanup() {
    echo "=== CLEANING UP ==="
    
    # Remove lock file
    rm -f "${LOCK_FILE}"
    
    # Rotate secrets
    kubectl rollout restart -n ${DEPLOY_NAMESPACE} deploy/agent
    
    # Archive logs
    gzip "${LOG_FILE}"
    mv "${LOG_FILE}.gz" "/var/log/archives/agent-upgrade-$(date +%s).log.gz"
    
    # Update service mesh
    kubectl apply -f ${NEW_VERSION}/mesh-policy-overrides.yaml
}

# Main Execution Flow
function main() {
    validate_upgrade
    create_backups
    perform_upgrade
    validate_upgrade
    cleanup
    
    echo "=== AGENT UPGRADE TO ${NEW_VERSION} COMPLETED SUCCESSFULLY ==="
}

trap 'rollback_upgrade; cleanup' ERR INT TERM
main
