#ifndef UNITS_H_
#define UNITS_H_

class Units {
	public:
		template <typename T>
		static T normalize(T value, T min, T max) {
			if (max < min) {
				std::swap(min, max);
			}

			T norm = (std::max(std::min(max, value), min) - min) / (max - min);

			return norm;
		}
};

#endif