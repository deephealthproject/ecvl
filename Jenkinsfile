pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {                                
                stage('linux') {
                    agent { dockerfile true }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(15) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                                }
                            }
                        }
                        stage('Test') {
                            steps {
                                timeout(15) {
                                    echo 'Testing..'
                                    ctest arguments: '-C Debug -VV', installation: 'InSearchPath', workingDir: 'build'
                                }
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
                        label 'windows && opencv'
                    }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(15) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [[withCmake: true]]
                                }
                            }    
                        }
                        stage('Test') {
                            steps {
                                timeout(15) {
                                    echo 'Testing..'
                                    bat 'cd build && ctest -C Debug -VV'
                                }
                            }
                        }
                        stage('windows_end') {
                            steps {
                                echo 'Success!'
                            }
                        }
                    }
                }
                stage('documentation') {
                    when { 
                        branch 'master' 
                        beforeAgent true
                    }
                    agent {
                        label 'windows && ecvl_doxygen'
                    }
                    stages {
                        stage('Update documentation') {
                            steps {
                                timeout(15) {
                                    bat 'cd doc\\doxygen && doxygen'
                                    bat 'powershell -Command "(gc %ECVL_DOXYGEN_INPUT_COMMANDS%) -replace \'@local_dir\', \'doc\\html\' | Out-File commands_out.txt"'
                                    bat 'winscp /ini:nul /script:commands_out.txt'
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
