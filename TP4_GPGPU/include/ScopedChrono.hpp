#pragma once

#include <string>
#include <stdio.h>

template<typename Chrono>
class ScopedChrono {
    Chrono m_Chrono;
    ScopedChrono();
public:
    ScopedChrono(const std::string& action) {
		printf("%s: ", action.c_str());
        fflush(stdout);
		m_Chrono.start();
    }
    ~ScopedChrono() {
		m_Chrono.stop();
        printf("%f ms\n", m_Chrono.elapsedTime());
    }
};
