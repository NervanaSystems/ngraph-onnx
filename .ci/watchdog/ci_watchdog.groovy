node(LABEL) {
    try {
            BUILD_WORKSPACE="$WORKSPACE/$BUILD_NUMBER"
            stage("Clone repository") {
                dir ("$BUILD_WORKSPACE") {
                    checkout([$class: 'GitSCM', branches: [[name: "*/$BRANCH"]], 
                            doGenerateSubmoduleConfigurations: false, extensions: [[$class: 'CloneOption', timeout: 30]],  submoduleCfg: [],
                            userRemoteConfigs: [[credentialsId: '6887a177-8c4d-4fe9-9c3b-fcd71b22bfba', url: 'git@github.com:NervanaSystems/ngraph-onnx.git']]])
                }
            }
            stage("Prepare environment") {
                dir ("$BUILD_WORKSPACE") {
                    sh """#!/bin/bash
                        virtualenv -p python2 venv
                        source venv/bin/activate
                        pip install python-jenkins
                        pip install pygithub
                        pip install arrow
                        pip install slackclient
                    """
                }
            }
            stage("Run script") {
                dir ("$BUILD_WORKSPACE") {
                    sh """#!/bin/bash
                        source venv/bin/activate
                        python .ci/watchdog/ci_watchdog.py
                    """
                }
            }
            stage("Cleanup") {
                sh """
                    cd $BUILD_WORKSPACE
                    rm -rf ..?* .[!.]* *
                """
            }

    } catch (e) {
        echo "$e"
        currentBuild.result = "FAILURE"
    } finally {
        deleteDir()
    }
}