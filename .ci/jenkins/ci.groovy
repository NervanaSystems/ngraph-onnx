// INTEL CONFIDENTIAL
// Copyright 2018-2019 Intel Corporation All Rights Reserved.
// The source code contained or described herein and all documents related to the
// source code ("Material") are owned by Intel Corporation or its suppliers or
// licensors. Title to the Material remains with Intel Corporation or its
// suppliers and licensors. The Material may contain trade secrets and proprietary
// and confidential information of Intel Corporation and its suppliers and
// licensors, and is protected by worldwide copyright and trade secret laws and
// treaty provisions. No part of the Material may be used, copied, reproduced,
// modified, published, uploaded, posted, transmitted, distributed, or disclosed
// in any way without Intel's prior express written permission.
// No license under any patent, copyright, trade secret or other intellectual
// property right is granted to or conferred upon you by disclosure or delivery of
// the Materials, either expressly, by implication, inducement, estoppel or
// otherwise. Any license under such intellectual property rights must be express
// and approved by Intel in writing.

if(DOCKER_REGISTRY.trim() == "") {throw new Exception("Missing Docker registry url!");}

// Map of SKUs and backend configurations to run CI on
try {if(BACKEND_SKU_CONFIGURATION.trim() == "") {throw new Exception()}}
catch (Exception e) {
    BACKEND_SKU_CONFIGURATIONS = [
        [ sku : "skx", backend : "cpu" ],
        [ sku : "clx", backend : "cpu" ],
        [ sku : "bdw", backend : "cpu" ]
        // [ sku: "iris", backend : "igpu" ]
    ]
}
echo "BACKEND_SKU_CONFIGURATIONS=${BACKEND_SKU_CONFIGURATIONS}"

// --- CI constants ---
NGRAPH_ONNX_REPO_ADDRESS="https://github.com/NervanaSystems/ngraph-onnx.git"
NGRAPH_REPO_ADDRESS="https://github.com/NervanaSystems/ngraph.git"
CI_LABELS = "ngraph_onnx && ci"
CI_DIR = "ngraph-onnx/.ci/jenkins"
DOCKER_CONTAINER_NAME = "jenkins_ngraph-onnx_ci"
JENKINS_GITHUB_CREDENTIAL_ID = "7157091e-bc04-42f0-99fd-dc4da2922a55"
JENKINS_HEADLESS_CREDENTIAL_ID = "19d4cefb-3ef3-4632-9553-10f5b9211bd5"

CONFIGURATION_WORKFLOW = { configuration ->
    node(configuration.label) {
        timeout(activity: true, time: 60) {
            WORKDIR = "${WORKSPACE}/${BUILD_NUMBER}"
            try {
                stage("Clone repositories") {
                    cloneRepository(NGRAPH_ONNX_REPO_ADDRESS, configuration.ngraphOnnxBranch)
                    cloneRepository(NGRAPH_REPO_ADDRESS, configuration.ngraphBranch)
                }
                String imageName = "${DOCKER_REGISTRY}/aibt/aibt/ngraph_cpp/${configuration.os}/base:ci"
                stage("Pull Docker image") {
                    pullDockerImage(imageName)
                }
                stage("Run Docker container") {
                    runDockerContainer(imageName)
                }
                stage("Prepare environment") {
                    prepareEnvironment(configuration.backend)
                }
                stage("Run tests") {
                    runToxTests()
                }
            }
            catch(e) {
                // Set result to ABORTED if exception contains exit code of a process interrupted by SIGTERM
                if ("$e".contains("143")) {
                    currentBuild.result = "ABORTED"
                } else {
                    currentBuild.result = "FAILURE"
                }
            }
            finally {
                stage("Cleanup") {
                    cleanup()
                }
            }
        }
    }
}

def cloneRepository(String address, String branch) {
    repositoryName = address.split("/").last().replace(".git","")
    dir ("${WORKDIR}/${repositoryName}") {
        retry(3) {
            checkout([$class: 'GitSCM',
                branches: [[name: "${branch}"]],
                doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'CloneOption', timeout: 30]], submoduleCfg: [],
                userRemoteConfigs: [[credentialsId: "${JENKINS_GITHUB_CREDENTIAL_ID}",
                url: "${address}"]]])
        }
    }
}

def pullDockerImage(String imageName) {
    withCredentials([usernamePassword(credentialsId: "${JENKINS_HEADLESS_CREDENTIAL_ID}",
                                        usernameVariable: 'dockerUsername',
                                        passwordVariable: 'dockerPassword')]) {
        sh """
            echo "${dockerPassword}" | docker login ${DOCKER_REGISTRY} --username ${dockerUsername} --password-stdin
            docker pull ${imageName}
        """
    }
}

def runDockerContainer(String imageName) {
    sh """
        mkdir -p ${HOME}/ONNX_CI
        docker run -id --privileged \
                --name ${DOCKER_CONTAINER_NAME}  \
                --volume ${WORKDIR}:/logs \
                --volume ${HOME}/ONNX_CI:/home \
                --volume ${HOME}/ONNX_CI/cache:/root/.cache \
                --volume ${WORKDIR}:/root \
                ${imageName} tail -f /dev/null
    """
}

def prepareEnvironment(String backend) {
    sh """
        docker exec ${DOCKER_CONTAINER_NAME} bash -c "/root/${CI_DIR}/prepare_environment.sh --backend=${backend}"
    """
}

def runToxTests() {
    sh """
        NGRAPH_WHL=\$(docker exec ${DOCKER_CONTAINER_NAME} find /root/ngraph/python/dist/ -name 'ngraph*.whl')
        docker exec -e TOX_INSTALL_NGRAPH_FROM=\${NGRAPH_WHL} ${DOCKER_CONTAINER_NAME} tox -c /root/ngraph-onnx
    """
}

def cleanup() {
    sh """
        docker start ${DOCKER_CONTAINER_NAME} || true
        docker exec ${DOCKER_CONTAINER_NAME} bash -c "find /root -maxdepth 1 ! -name ".cache" ! -path /root -exec rm -rf {} \;" || true
        docker exec ${DOCKER_CONTAINER_NAME} bash -c "find /root/cache -maxdepth 1 ! -name "pip" ! -path /root/.cache -exec rm -rf {} \;" || true
        docker stop ${DOCKER_CONTAINER_NAME} || true
        docker rm -f \$(docker ps -a -q) || true
        printf 'y' | docker system prune
    """
    deleteDir()
}

def getConfigurationsMap(String dockerfilesPath, String ngraphOnnxBranch, String ngraphBranch) {
    def configurationsMap = [:]
    def osImages = sh (script: "find ${dockerfilesPath} -name '*.dockerfile' -printf '%f\n'",
                    returnStdout: true).trim().replaceAll(".dockerfile","").split("\n") as List

    for (os in osImages) {
        for (backendSku in BACKEND_SKU_CONFIGURATIONS) {
            def configuration = backendSku.clone()
            configuration.os = os
            configuration.ngraphOnnxBranch = ngraphOnnxBranch
            configuration.ngraphBranch = ngraphBranch
            configuration.label = "${configuration.backend} && ${configuration.sku} && ${CI_LABELS}"
            configuration.name = "${configuration.sku}-${configuration.os}"
            configurationsMap[configuration.name] = {
                stage(configuration.name) { CONFIGURATION_WORKFLOW(configuration) }
            }
        }
    }
    return configurationsMap
}

return this
