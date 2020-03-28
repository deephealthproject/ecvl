pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker { 
                            label 'docker'
                            image 'pritt/ecvl:latest'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(15) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', cmakeArgs: '-DECVL_TESTS=ON -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_WITH_DICOM=ON -DECVL_WITH_OPENSLIDE=ON', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '-j4', withCmake: true]
                                    ]
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
                                    bat 'powershell ../../ecvl_dependencies/ecvl_dependencies.ps1'
                                    cmakeBuild buildDir: 'build', cmakeArgs: '-DECVL_TESTS=ON -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_WITH_DICOM=ON -DECVL_WITH_OPENSLIDE=ON -DOPENSLIDE_LIBRARIES=C:/Library/openslide-win32-20171122/lib/libopenslide.lib', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '-j4', withCmake: true]
                                    ]
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
                                    bat '"../../doxygen_git/update_doc_script.bat"'
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
