#!/bin/bash
# Rossmann Sales Forecasting - Cloud Deployment Script
# Machine Learning Zoomcamp 2025 - Midterm Project

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="rossmann-forecasting"
IMAGE_NAME="rossmann-sales-forecasting"
REGISTRY_PREFIX=""  # Will be set based on cloud provider

echo -e "${BLUE}🚀 Rossmann Sales Forecasting - Cloud Deployment${NC}"
echo -e "${BLUE}Machine Learning Zoomcamp 2025 - Midterm Project${NC}"
echo -e "${BLUE}Enhanced with Prophet & ARIMA Time Series Models${NC}"
echo "=================================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to deploy to Google Cloud Run
deploy_gcp() {
    echo -e "${BLUE}📱 Deploying to Google Cloud Run...${NC}"
    
    # Check if gcloud is installed
    if ! command_exists gcloud; then
        print_error "gcloud CLI not found. Please install Google Cloud SDK."
        exit 1
    fi
    
    # Get project ID
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    if [ -z "$PROJECT_ID" ]; then
        print_error "No GCP project set. Run: gcloud config set project YOUR_PROJECT_ID"
        exit 1
    fi
    
    print_status "Using GCP project: $PROJECT_ID"
    
    # Set registry prefix
    REGISTRY_PREFIX="gcr.io/$PROJECT_ID"
    FULL_IMAGE_NAME="$REGISTRY_PREFIX/$IMAGE_NAME"
    
    # Build and submit to Cloud Build
    print_status "Building container with Cloud Build..."
    gcloud builds submit --tag "$FULL_IMAGE_NAME" .
    
    # Deploy to Cloud Run
    print_status "Deploying to Cloud Run..."
    gcloud run deploy $PROJECT_NAME \
        --image "$FULL_IMAGE_NAME" \
        --platform managed \
        --region us-central1 \
        --allow-unauthenticated \
        --port 5000 \
        --memory 2Gi \
        --cpu 1 \
        --max-instances 10 \
        --timeout 300 \
        --set-env-vars "FLASK_ENV=production"
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $PROJECT_NAME --region=us-central1 --format='value(status.url)')
    
    print_status "Deployment completed!"
    echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
    echo -e "${GREEN}🔍 Test with: curl $SERVICE_URL/health${NC}"
}

# Function to deploy to AWS ECS with Fargate
deploy_aws() {
    echo -e "${BLUE}☁️ Deploying to AWS ECS Fargate...${NC}"
    
    # Check if AWS CLI is installed
    if ! command_exists aws; then
        print_error "AWS CLI not found. Please install AWS CLI v2."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &>/dev/null; then
        print_error "AWS credentials not configured. Run: aws configure"
        exit 1
    fi
    
    # Get AWS account ID and region
    ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    REGION=$(aws configure get region)
    
    if [ -z "$REGION" ]; then
        REGION="us-east-1"
        print_warning "No region set, using default: $REGION"
    fi
    
    print_status "Using AWS Account: $ACCOUNT_ID, Region: $REGION"
    
    # Set registry prefix
    REGISTRY_PREFIX="$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com"
    FULL_IMAGE_NAME="$REGISTRY_PREFIX/$IMAGE_NAME:latest"
    
    # Create ECR repository if it doesn't exist
    print_status "Setting up ECR repository..."
    aws ecr describe-repositories --repository-names $IMAGE_NAME --region $REGION &>/dev/null || \
        aws ecr create-repository --repository-name $IMAGE_NAME --region $REGION
    
    # Get ECR login
    print_status "Logging into ECR..."
    aws ecr get-login-password --region $REGION | \
        docker login --username AWS --password-stdin $REGISTRY_PREFIX
    
    # Build and push image
    print_status "Building and pushing Docker image..."
    docker build -t $IMAGE_NAME .
    docker tag $IMAGE_NAME:latest $FULL_IMAGE_NAME
    docker push $FULL_IMAGE_NAME
    
    print_status "Docker image pushed to ECR!"
    print_warning "Next steps for ECS deployment:"
    echo "  1. Create ECS cluster (if not exists)"
    echo "  2. Create task definition with image: $FULL_IMAGE_NAME"
    echo "  3. Set memory to at least 2GB (Prophet models need more memory)"
    echo "  4. Create ECS service with health checks"
    echo "  5. Configure Application Load Balancer"
    echo -e "${BLUE}📖 Full instructions: https://docs.aws.amazon.com/ecs/latest/userguide/getting-started-fargate.html${NC}"
}

# Function to build and test locally
test_local() {
    echo -e "${BLUE}🧪 Testing local deployment...${NC}"
    
    if ! command_exists docker; then
        print_error "Docker not found. Please install Docker."
        exit 1
    fi
    
    # Build image
    print_status "Building Docker image locally..."
    docker build -t $IMAGE_NAME .
    
    # Run container in background
    print_status "Starting container..."
    CONTAINER_ID=$(docker run -d -p 5000:5000 --name $PROJECT_NAME-test $IMAGE_NAME)
    
    # Wait for service to be ready
    print_status "Waiting for service to start..."
    sleep 10
    
    # Test health endpoint
    if curl -f http://localhost:5000/health &>/dev/null; then
        print_status "Local deployment test passed!"
        echo -e "${GREEN}🌐 Service running at: http://localhost:5000${NC}"
        echo -e "${GREEN}🔍 Test with: curl http://localhost:5000/health${NC}"
    else
        print_error "Local deployment test failed!"
        echo "Container logs:"
        docker logs $CONTAINER_ID
    fi
    
    # Cleanup
    echo -e "${YELLOW}Press Enter to stop the test container...${NC}"
    read
    docker stop $CONTAINER_ID &>/dev/null || true
    docker rm $CONTAINER_ID &>/dev/null || true
    
    print_status "Test container cleaned up."
}

# Main deployment logic
case "$1" in
    "gcp")
        deploy_gcp
        ;;
    "aws")
        deploy_aws
        ;;
    "local")
        test_local
        ;;
    *)
        echo -e "${BLUE}Usage: $0 {gcp|aws|local}${NC}"
        echo ""
        echo -e "${YELLOW}Commands:${NC}"
        echo "  gcp    - Deploy to Google Cloud Run (Recommended for time series)"
        echo "  aws    - Deploy to AWS ECS (ECR push only)"
        echo "  local  - Test deployment locally"
        echo ""
        echo -e "${YELLOW}Examples:${NC}"
        echo "  $0 local    # Test locally with Docker"
        echo "  $0 gcp      # Deploy to Google Cloud Run (2GB memory)"
        echo "  $0 aws      # Push to AWS ECR (configure 2GB+ memory)"
        echo ""
        echo -e "${YELLOW}Prerequisites:${NC}"
        echo "  • Docker installed and running"
        echo "  • Cloud CLI tools (gcloud/aws) installed"
        echo "  • Authentication configured"
        echo "  • Project/account permissions set up"
        echo "  • Extra memory for Prophet/ARIMA models (2GB recommended)"
        exit 1
        ;;
esac

print_status "Deployment script completed!"