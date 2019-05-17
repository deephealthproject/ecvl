pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker { 
                            label 'docker'
                            image 'stal12/opencv:3.4.6'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
                                echo 'Building..'
                                cmakeBuild buildDir: 'build', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                             }
                        }
                        stage('Test') {
                            steps {
                                echo 'Testing..'
                                ctest arguments: '-C Debug -VV', installation: 'InSearchPath', workingDir: 'build'
                            }
                        }
                        stage('linux_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('windows') {
                    agent {
                        label 'windows'
                    }
                    stages {
                        stage('Build') {
                            steps {
                                echo 'Building..'
                                cmakeBuild buildDir: 'build', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                            }
                        }
                        stage('Test') {
                            steps {
                                echo 'Testing..'
                                bat 'cd build && ctest -C Debug -VV'
                            }
                        }
                        stage('windows_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
            }
        }
    }
}
