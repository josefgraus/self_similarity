#ifndef TIMER_H_
#define TIMER_H_

#include <functional>
#include <chrono>
#include <future>
#include <cstdio>

class Timer {
	public:
		template <class callable, class... arguments>
		Timer(int after, bool async, callable&& f, arguments&&... args) {
			std::function<typename std::result_of<callable(arguments...)>::type()> task(std::bind(std::forward<callable>(f), std::forward<arguments>(args)...));

			if (async) {
				std::thread([after, task]() {
					std::this_thread::sleep_for(std::chrono::milliseconds(after));
					task();
				}).detach();
			} else {
				std::this_thread::sleep_for(std::chrono::milliseconds(after));
				task();
			}
		}
};

#endif