#!/usr/bin/env bash
set -eo pipefail

# ==============================================
# Cluster Initialization Script - Production Grade
# ==============================================

# Global Configuration
CLUSTER_NAME="biconic-prod"
CLOUD_PROVIDER="aws"  # aws|azure|gcp
ENVIRONMENT="prod"
REGION="us-west-2"
K8S_VERSION="1.28"
DEPLOY_NAMESPACE="biconic-system"
ADMIN_EMAIL="admin@biconic.ai"
ROOT_DIR="/opt/aelion"
BACKUP_DIR="${ROOT_DIR}/backups"
LOG_FILE="/var/log/cluster-init.log"

# Infrastructure Sizing
CONTROL_PLANE_NODES=3
WORKER_NODES=5
NODE_INSTANCE_TYPE="m6i.4xlarge"
STORAGE_SIZE="500Gi"

# Security Parameters
TLS_VALIDITY_DAYS=3650
ENCRYPTION_KEY="$(openssl rand -base64 32)"
ADMIN_CERT_DAYS=730

# Dependency Versions
HELM_VERSION="3.12.3"
TERRAFORM_VERSION="1.5.7"
KUSTOMIZE_VERSION="5.0.3"

# Initialize logging
exec > >(tee -a "${LOG_FILE}") 2>&1

# Phase 1: Pre-flight Checks
function preflight_checks() {
    echo "=== STARTING PREFLIGHT CHECKS ==="
    
    # Verify execution context
    if [[ $(id -u) -ne 0 ]]; then
        echo "ERROR: Must be run as root"
        exit 1
    fi

    # Check dependencies
    declare -A REQUIRED_CMDS=(
        ["kubectl"]="1.28"
        ["helm"]="${HELM_VERSION}"
        ["terraform"]="${TERRAFORM_VERSION}"
        ["jq"]="1.6"
        ["openssl"]="3.0"
    )

    for cmd in "${!REQUIRED_CMDS[@]}"; do
        if ! command -v "${cmd}" &> /dev/null; then
            echo "ERROR: ${cmd} not found"
            exit 1
        fi
        
        version=$(${cmd} --version 2>&1 | head -n1)
        if [[ ! "${version}" =~ ${REQUIRED_CMDS[$cmd]} ]]; then
            echo "ERROR: ${cmd} version mismatch"
            exit 1
        fi
    done

    # Validate cloud credentials
    case "${CLOUD_PROVIDER}" in
        aws)
            if [[ -z "${AWS_ACCESS_KEY_ID}" || -z "${AWS_SECRET_ACCESS_KEY}" ]]; then
                echo "ERROR: AWS credentials not configured"
                exit 1
            fi
            ;;
        azure)
            if [[ -z "${AZURE_SUBSCRIPTION_ID}" || -z "${AZURE_TENANT_ID}" ]]; then
                echo "ERROR: Azure credentials not configured"
                exit 1
            fi
            ;;
        gcp)
            if [[ -z "${GOOGLE_CREDENTIALS}" ]]; then
                echo "ERROR: GCP credentials not configured"
                exit 1
            fi
            ;;
        *)
            echo "ERROR: Unsupported cloud provider"
            exit 1
            ;;
    esac

    # Check storage availability
    if [[ $(df --output=avail / | tail -1) -lt 52428800 ]]; then
        echo "ERROR: Insufficient disk space"
        exit 1
    fi

    echo "=== PREFLIGHT CHECKS PASSED ==="
}

# Phase 2: Infrastructure Provisioning
function provision_infrastructure() {
    echo "=== PROVISIONING CLUSTER INFRASTRUCTURE ==="
    
    # Generate Terraform configuration
    cat <<EOF > cluster.tf
module "aelion_cluster" {
  source  = "terraform-${CLOUD_PROVIDER}-modules/kubernetes-cluster/${CLOUD_PROVIDER}"
  version = "4.12.0"

  cluster_name      = "${CLUSTER_NAME}"
  region            = "${REGION}"
  k8s_version       = "${K8S_VERSION}"
  node_count        = ${WORKER_NODES}
  control_plane_count = ${CONTROL_PLANE_NODES}
  node_instance_type = "${NODE_INSTANCE_TYPE}"
  storage_size      = "${STORAGE_SIZE}"
  
  enable_autoscaling = true
  min_nodes         = 3
  max_nodes         = 10
  
  enable_encryption = true
  encryption_key    = "${ENCRYPTION_KEY}"
  
  tags = {
    Environment = "${ENVIRONMENT}"
    ManagedBy   = "Aelion AI"
  }
}
EOF

    # Initialize and apply Terraform
    terraform init
    terraform apply -auto-approve
    
    # Configure kubectl context
    case "${CLOUD_PROVIDER}" in
        aws)
            aws eks update-kubeconfig --name "${CLUSTER_NAME}" --region "${REGION}"
            ;;
        azure)
            az aks get-credentials --resource-group "${CLUSTER_NAME}-rg" --name "${CLUSTER_NAME}"
            ;;
        gcp)
            gcloud container clusters get-credentials "${CLUSTER_NAME}" --region "${REGION}"
            ;;
    esac

    # Verify cluster access
    if ! kubectl cluster-info; then
        echo "ERROR: Cluster connection failed"
        exit 1
    fi

    echo "=== INFRASTRUCTURE PROVISIONING COMPLETE ==="
}

# Phase 3: Cluster Bootstrapping
function bootstrap_cluster() {
    echo "=== BOOTSTRAPPING CLUSTER COMPONENTS ==="
    
    # Create namespace
    kubectl create namespace "${DEPLOY_NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy core components
    deploy_cni
    deploy_csi
    deploy_ingress_controller
    deploy_cert_manager
    deploy_metrics_server
    deploy_prometheus_stack
    deploy_efk_logging
    deploy_vault
    deploy_backup_operator
}

function deploy_cni() {
    echo "--- Deploying CNI (Cilium) ---"
    helm repo add cilium https://helm.cilium.io/
    helm upgrade --install cilium cilium/cilium \
        --namespace kube-system \
        --set kubeProxyReplacement=strict \
        --set k8sServiceHost=api-server.${CLUSTER_NAME}.internal \
        --set hubble.relay.enabled=true \
        --set hubble.ui.enabled=true
}

function deploy_cert_manager() {
    echo "--- Deploying Cert Manager ---"
    kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.12.0/cert-manager.yaml
    
    # Wait for readiness
    kubectl wait --for=condition=Available deployment/cert-manager -n cert-manager --timeout=300s
    
    # Create cluster issuer
    cat <<EOF | kubectl apply -f -
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata:
  name: letsencrypt-prod
spec:
  acme:
    server: https://acme-v02.api.letsencrypt.org/directory
    email: ${ADMIN_EMAIL}
    privateKeySecretRef:
      name: letsencrypt-prod
    solvers:
    - http01:
        ingress:
          class: nginx
EOF
}

function deploy_prometheus_stack() {
    echo "--- Deploying Monitoring Stack ---"
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm upgrade --install kube-prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set alertmanager.enabled=true \
        --set grafana.enabled=true \
        --set prometheus.prometheusSpec.retention=30d
}

# Phase 4: Security Hardening
function harden_security() {
    echo "=== APPLYING SECURITY CONFIGURATIONS ==="
    
    # Generate TLS certificates
    openssl req -x509 -newkey rsa:4096 -days ${TLS_VALIDITY_DAYS} -nodes \
        -keyout ${ROOT_DIR}/tls.key -out ${ROOT_DIR}/tls.crt \
        -subj "/CN=aelion.ai/O=Aelion AI"
        
    # Create secret
    kubectl create secret tls aelion-tls \
        --key=${ROOT_DIR}/tls.key \
        --cert=${ROOT_DIR}/tls.crt \
        --namespace=${DEPLOY_NAMESPACE}
        
    # Apply network policies
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: default-deny
  namespace: ${DEPLOY_NAMESPACE}
spec:
  podSelector: {}
  policyTypes:
  - Ingress
  - Egress
EOF

    # Configure RBAC
    kubectl apply -f - <<EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: aelion-admin
rules:
- apiGroups: ["*"]
  resources: ["*"]
  verbs: ["*"]
EOF

    # Create admin certificate
    openssl genrsa -out ${ROOT_DIR}/admin.key 2048
    openssl req -new -key ${ROOT_DIR}/admin.key \
        -out ${ROOT_DIR}/admin.csr \
        -subj "/CN=admin/O=system:masters"
    openssl x509 -req -in ${ROOT_DIR}/admin.csr \
        -CA ${ROOT_DIR}/tls.crt -CAkey ${ROOT_DIR}/tls.key -CAcreateserial \
        -out ${ROOT_DIR}/admin.crt -days ${ADMIN_CERT_DAYS}
        
    # Configure kubectl context
    kubectl config set-credentials admin \
        --client-certificate=${ROOT_DIR}/admin.crt \
        --client-key=${ROOT_DIR}/admin.key
}

# Phase 5: Application Deployment
function deploy_applications() {
    echo "=== DEPLOYING AELION COMPONENTS ==="
    
    # Create persistent volumes
    kubectl apply -f ${ROOT_DIR}/storage/
    
    # Deploy databases
    helm upgrade --install postgresql bitnami/postgresql-ha \
        --namespace=${DEPLOY_NAMESPACE} \
        --values ${ROOT_DIR}/postgresql-values.yaml
    
    # Deploy Kafka
    helm upgrade --install kafka bitnami/kafka \
        --namespace=${DEPLOY_NAMESPACE} \
        --set replicas=3 \
        --set persistence.size=${STORAGE_SIZE}
    
    # Deploy core services
    kubectl apply -k ${ROOT_DIR}/kustomize/overlays/${ENVIRONMENT}
    
    # Wait for readiness
    kubectl rollout status deployment/aelion-orchestrator -n ${DEPLOY_NAMESPACE} --timeout=600s
    kubectl rollout status statefulset/aelion-agents -n ${DEPLOY_NAMESPACE} --timeout=600s
}

# Phase 6: Validation & Testing
function validate_deployment() {
    echo "=== VALIDATING DEPLOYMENT ==="
    
    # Verify component status
    declare -A DEPLOYMENTS=(
        ["aelion-orchestrator"]=3
        ["aelion-api-gateway"]=2
        ["aelion-metrics-collector"]=2
    )
    
    for dep in "${!DEPLOYMENTS[@]}"; do
        replicas=$(kubectl get deployment/${dep} -n ${DEPLOY_NAMESPACE} -o jsonpath='{.status.readyReplicas}')
        if [[ ${replicas} -ne ${DEPLOYMENTS[$dep]} ]]; then
            echo "ERROR: ${dep} not ready"
            exit 1
        fi
    done
    
    # Run smoke tests
    API_ENDPOINT="https://api.aelion.ai/health"
    if ! curl -sk ${API_ENDPOINT} | grep "OK"; then
        echo "ERROR: API health check failed"
        exit 1
    fi
    
    # Validate data pipeline
    kubectl apply -f ${ROOT_DIR}/tests/pipeline-test.yaml
    kubectl wait --for=condition=complete job/pipeline-test -n ${DEPLOY_NAMESPACE} --timeout=300s
}

# Phase 7: Backup Configuration
function configure_backups() {
    echo "=== CONFIGURING BACKUP SYSTEMS ==="
    
    # Create backup schedule
    helm upgrade --install velero vmware-tanzu/velero \
        --namespace velero \
        --create-namespace \
        --set configuration.backupStorageLocation[0].name=aws \
        --set configuration.backupStorageLocation[0].provider=aws \
        --set configuration.backupStorageLocation[0].bucket=aelion-backups \
        --set schedules.daily.schedule="0 2 * * *" \
        --set schedules.daily.ttl="720h"
    
    # Initial backup
    velero backup create initial-deployment --include-namespaces=${DEPLOY_NAMESPACE}
}

# Main Execution
function main() {
    preflight_checks
    provision_infrastructure
    bootstrap_cluster
    harden_security
    deploy_applications
    validate_deployment
    configure_backups
    
    echo "=== CLUSTER INITIALIZATION COMPLETE ==="
    echo "Dashboard URL: https://dashboard.aelion.ai"
    echo "Admin credentials stored in: ${ROOT_DIR}/admin.crt"
}

# Execute with error trapping
trap 'echo "ERROR at line ${LINENO}"; exit 1' ERR
main
