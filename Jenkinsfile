pipeline {
    agent none
    stages {
        stage('Parallel Stages') {
            parallel {
                stage('linux') {
                    agent {
                        docker {
                            label 'docker'
                            image 'aimagelab/ecvl:latest'
                        }
                    }
                    stages {
                        stage('Build') {
                            steps {
                                timeout(60) {
                                    echo 'Building..'
                                    cmakeBuild buildDir: 'build', buildType: 'Release', cmakeArgs: '-DECVL_TESTS=ON -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_WITH_DICOM=ON -DECVL_WITH_OPENSLIDE=ON -DECVL_GPU=OFF', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '--parallel 4', withCmake: true]
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
                                timeout(60) {
                                    echo 'Building..'
                                    bat 'powershell ../../ecvl_dependencies/ecvl_dependencies.ps1'
                                    cmakeBuild buildDir: 'build', buildType: 'Release', cmakeArgs: '-DECVL_TESTS=ON -DECVL_BUILD_EDDL=ON -DECVL_DATASET=ON -DECVL_WITH_DICOM=ON -DECVL_WITH_OPENSLIDE=ON -DOPENSLIDE_LIBRARIES=C:/Library/openslide-win32-20171122/lib/libopenslide.lib', installation: 'InSearchPath', sourceDir: '.', cleanBuild: true, steps: [
                                        [args: '--parallel 4', withCmake: true]
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
                        stage('Coverage') {
                            steps {
                                timeout(15) {
                                    echo 'Calculating coverage..'
                                    bat '"C:/Program Files/OpenCppCoverage/OpenCppCoverage.exe" --source %cd% --export_type=cobertura --excluded_sources=3rdparty -- "build/bin/Debug/ECVL_TESTS.exe"'
                                    cobertura coberturaReportFile: 'ECVL_TESTSCoverage.xml'
                                    bat 'codecov -f ECVL_TESTSCoverage.xml -t 7635bd2e-51cf-461e-bb1b-fc7ba9fb26d1'
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
                stage('release-doc') {
                    when { tag "v*" }
                    agent {
                        label 'windows && ecvl_doxygen'
                    }
                    stages {
                        stage('Set Release Documentation') {
                            steps {
                                timeout(15) {
                                    bat 'cd doc\\doxygen && doxygen'
                                    bat '"../../doxygen_git/update_doc_script.bat" y'
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
