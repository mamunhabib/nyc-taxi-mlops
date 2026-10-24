pipeline {
    agent any

    triggers {
        // Poll GitHub every 2 minutes for automated CI/CD triggers on push
        pollSCM('H/2 * * * *')
    }

    parameters {
        booleanParam(
            name: 'FORCE_RETRAIN',
            defaultValue: false,
            description: 'Force Continuous Training (CT) retraining loop regardless of drift detection'
        )
    }

    environment {
        SERVING_IMAGE = 'ghcr.io/mamunhabib/nyc-taxi-serving:latest'
        UI_IMAGE      = 'ghcr.io/mamunhabib/nyc-taxi-ui:latest'
        NAMESPACE     = 'nyc-taxi-mlops'
        RUNNER_IMAGE  = 'nyc-taxi-runner:latest'
    }

    stages {
        stage('Unit Testing') {
            steps {
                echo "Building CI/CD test and training runner image..."
                sh "docker build -t ${RUNNER_IMAGE} -f Dockerfile.runner ."

                echo "Running Unit and Integration tests..."
                sh "docker run --rm -v \$(pwd):/app -w /app ${RUNNER_IMAGE} pytest tests/test_data_prep.py -v"
            }
        }

        stage('Continuous Training (CT) & Drift Check') {
            steps {
                echo "Evaluating production data drift and triggering automated retraining if needed..."
                sh """
                docker run --rm --network host \
                    -v \$(pwd):/app -w /app \
                    -e MLFLOW_TRACKING_URI=http://localhost:8087 \
                    -e MLFLOW_S3_ENDPOINT_URL=http://localhost:9000 \
                    -e AWS_ACCESS_KEY_ID=minioadmin \
                    -e AWS_SECRET_ACCESS_KEY=minioadmin \
                    ${RUNNER_IMAGE} \
                    bash -c '
                        if [ "${params.FORCE_RETRAIN}" = "true" ]; then
                            echo "[CI/CD] Forcing Continuous Training cycle..."
                            python pipelines/training_flow.py --force
                        else
                            echo "[CI/CD] Evaluating production data drift for retraining..."
                            python pipelines/training_flow.py
                        fi
                    '
                """
            }
        }

        stage('Build Docker Images') {
            steps {
                echo "Building backend serving image with latest champion weights: ${SERVING_IMAGE}..."
                sh "docker build -t ${SERVING_IMAGE} -f serving/Dockerfile ."

                echo "Building frontend UI image: ${UI_IMAGE}..."
                sh "docker build -t ${UI_IMAGE} -f ui/Dockerfile ."
            }
        }

        stage('Push to GHCR') {
            steps {
                echo "Logging into GHCR and pushing images..."
                withCredentials([usernamePassword(credentialsId: 'github-pat', passwordVariable: 'GITHUB_PAT', usernameVariable: 'GITHUB_USER')]) {
                    sh """
                    echo \$GITHUB_PAT | docker login ghcr.io -u \$GITHUB_USER --password-stdin
                    docker push ${SERVING_IMAGE}
                    docker push ${UI_IMAGE}
                    """
                }
            }
        }

        stage('Clean Local Images') {
            steps {
                echo "Removing images from local Docker cache to force Kubernetes to pull fresh from GHCR..."
                sh "docker rmi ${SERVING_IMAGE} || true"
                sh "docker rmi ${UI_IMAGE} || true"
            }
        }

        stage('Deploy to Kubernetes') {
            steps {
                echo "Applying Kubernetes applications to namespace ${NAMESPACE}..."
                sh "kubectl apply -f k8s/apps/ -n ${NAMESPACE}"

                echo "Triggering zero-downtime rolling restart..."
                sh "kubectl rollout restart deployment/nyc-taxi-serving -n ${NAMESPACE}"
                sh "kubectl rollout restart deployment/nyc-taxi-ui -n ${NAMESPACE}"

                echo "Awaiting deployment rollout completion..."
                sh "kubectl rollout status deployment/nyc-taxi-serving -n ${NAMESPACE} --timeout=120s"
                sh "kubectl rollout status deployment/nyc-taxi-ui -n ${NAMESPACE} --timeout=120s"
            }
        }

        stage('Smoke Test & Verification') {
            steps {
                echo "Verifying serving health endpoint on port 8085..."
                sh """
                sleep 5
                curl -sf http://localhost:8085/health || kubectl exec -n ${NAMESPACE} deploy/nyc-taxi-serving -- curl -sf http://localhost:8000/health
                """

                echo "============================================================"
                echo "NYC TAXI MLOPS PLATFORM IS LIVE & HEALTHY"
                echo "Streamlit Web UI:        http://localhost:8089"
                echo "FastAPI Serving Swagger: http://localhost:8085/docs"
                echo "Evidently Drift UI:      http://localhost:8086"
                echo "MLflow Model Registry:   http://localhost:8087"
                echo "MinIO Object Storage:    http://localhost:8088"
                echo "============================================================"
            }
        }
    }

    post {
        success {
            echo "CI/CD & Continuous Training (CT) Pipeline executed successfully!"
        }
        failure {
            echo "Pipeline run failed. Check stage logs for troubleshooting."
        }
    }
}
