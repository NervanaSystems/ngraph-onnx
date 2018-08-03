// INTEL CONFIDENTIAL
// Copyright 2018 Intel Corporation All Rights Reserved.
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

// CI settings
PROJECT_NAME = "ngraph-onnx"
REPOSITORY_GIT_ADDRESS = "git@github.com:NervanaSystems/${PROJECT_NAME}.git"
DOCKER_CONTAINER_NAME = "jenkins_${PROJECT_NAME}_ci"
JENKINS_GITHUB_CREDENTIAL_ID = "f9f1f2ce-47b8-47cb-8fa1-c22d16179dce"

UTILS = load ".ci/jenkins/utils/utils.groovy"
result = 'SUCCESS'


def CloneRepository(String jenkins_github_credential_id, String repository_git_address) {
    configurationMaps = []
    configurationMaps.add([
        "name": "clone_repositories",
        "repository_git_address": repository_git_address,
        "jenkins_github_credential_id": jenkins_github_credential_id
    ])

    Closure cloneRepositories = {
        dir("${PROJECT_NAME}/") {
            checkout([$class: 'GitSCM',
            branches: [[name: "$ghprbActualCommit"]],
            doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'CloneOption', timeout: 30]], submoduleCfg: [],
            userRemoteConfigs: [[credentialsId: "${configMap["jenkins_github_credential_id"]}",
            refspec: '+refs/pull/*:refs/remotes/origin/pr/*', url: "${configMap["repository_git_address"]}"]]])
        }
    }
    UTILS.CreateStage("Clone_repository", cloneRepositories, configurationMaps)
}

def BuildImage(configurationMaps) {
    Closure buildMethod = { configMap ->
        UTILS.PropagateStatus("Clone_repository", "clone_repositories")
        sh """
            .ci/jenkins/utils/docker.sh build \
                                --name=${configMap["projectName"]} \
                                --version=${configMap["name"]} \
                                --dockerfile_path=${configMap["dockerfilePath"]}
        """
    }
    UTILS.CreateStage("Build_Image", buildMethod, configurationMaps)
}

def RunDockerContainers(configurationMaps) {
    Closure runContainerMethod = { configMap ->
        UTILS.PropagateStatus("Build_Image", configMap["name"])
        sh """
            mkdir -p ${HOME}/ONNX_CI
            .ci/jenkins/utils/docker.sh start \
                                --name=${configMap["projectName"]} \
                                --version=${configMap["name"]} \
                                --container_name=${configMap["dockerContainerName"]} \
                                --volumes="-v ${WORKSPACE}:/logs -v ${HOME}/ONNX_CI:/home"
        """
    }
    UTILS.CreateStage("Run_docker_containers", runContainerMethod, configurationMaps)
}

def BuildNgraph(configurationMaps) {
    Closure buildNgraphMethod = { configMap ->
        UTILS.PropagateStatus("Run_docker_containers", configMap["dockerContainerName"])
        sh """
            docker cp .ci/jenkins/update_ngraph.sh ${configMap["dockerContainerName"]}:/home
            docker exec bash /home/update_ngraph.sh
        """
    }
    UTILS.CreateStage("Build_NGraph", buildNgraphMethod, configurationMaps)
}

def RunToxTests(configurationMaps) {
    Closure prepareEnvMethod = { configMap ->
        UTILS.PropagateStatus("Build_NGraph", configMap["dockerContainerName"])
        sh """
            NGRAPH_WHL=$(docker exec ${configMap["dockerContainerName"]} find /~/ngraph/python/dist/ -name 'ngraph*.whl')
            docker exec -e TOX_INSTALL_NGRAPH_FROM=${NGRAPH_WHL} ${configMap["dockerContainerName"]} tox
        """
    }
    UTILS.CreateStage("Run_tox_tests", prepareEnvMethod, configurationMaps)
}

def Cleanup(configurationMaps) {
    Closure cleanupMethod = { configMap ->
        sh """
            ./utils/docker.sh chmod --container_name=${configMap["dockerContainerName"]} --directory="/logs" --options="-R 777" || true
            ./utils/docker.sh stop --container_name=${configMap["dockerContainerName"]} || true
            ./utils/docker.sh remove --container_name=${configMap["dockerContainerName"]} || true
            ./utils/docker.sh clean_up || true
        """
    }
    UTILS.CreateStage("Cleanup", cleanupMethod, configurationMaps)
}

def main(String projectName, String dockerContainerName, String jenkins_github_credential_id, String repository_git_address) {
    timeout(activity: true, time: 60) {
        def configurationMaps = UTILS.GetDockerEnvList(projectName, dockerContainerName)
        CloneRepository(jenkins_github_credential_id, repository_git_address)
        BuildImage(configurationMaps)
        RunDockerContainers(configurationMaps)
        RunTests(configurationMaps)
        Cleanup(configurationMaps)
    }
}

main(PROJECT_NAME, DOCKER_CONTAINER_NAME, JENKINS_GITHUB_CREDENTIAL_ID, REPOSITORY_GIT_ADDRESS)
