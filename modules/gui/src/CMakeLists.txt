target_sources(ECVL_GUI
	PRIVATE
    
    gui.cpp

)

if(${ECVL_WITH_OPENGL})
    target_sources(ECVL_GUI
    PRIVATE

	gui3D.cpp
)
endif(${ECVL_WITH_OPENGL})