PROJECT_NAME = "ngraph-onnx"

node('ci && onnx'){
    WORKDIR = "${WORKSPACE}/${BUILD_NUMBER}/${PROJECT_NAME}"
    CI_ROOT = ".ci/jenkins"
    dir ("${WORKDIR}") {
        stage('Clone Repo') {
            checkout([$class: 'GitSCM',
                     branches: [[name: "$ghprbActualCommit"]],
                     doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'CloneOption', timeout: 30]], submoduleCfg: [],
                     userRemoteConfigs: [[credentialsId: "f9f1f2ce-47b8-47cb-8fa1-c22d16179dce",
                     url: "git@github.com:NervanaSystems/ngraph-onnx.git"]]])
        }
        load "${CI_ROOT}/ci.groovy"
    }
    stage("Cleanup build directory") {
        sh """
            cd $WORKSPACE/$BUILD_NUMBER
            rm -rf ..?* .[!.]* *
        """
    }
}
