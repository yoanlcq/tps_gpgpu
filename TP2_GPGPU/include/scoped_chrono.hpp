/*
 * File: scoped_chrono.hpp
 * Author: Yoan LECOQ
 */

#include <string>
#include <iostream>

template<typename Chrono>
class ScopedChrono {
    Chrono m_Chrono;
    ScopedChrono();
public:
    ScopedChrono(const std::string& action) {
		std::cout << action << ": " << std::flush;
		m_Chrono.start();
    }
    ~ScopedChrono() {
		m_Chrono.stop();
		std::cout << m_Chrono.elapsedTime() << " ms" << std::endl;
    }
};
