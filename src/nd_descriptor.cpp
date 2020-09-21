#include "gui/control_display.h"

//#include <benchmarks/eigen_decomp.h>

int main() {
	/*std::string benchmark_dir = "D:\\test_models\\";
	std::string output_dir = "D:\\test_output\\";

	EigenDecomp eigen_decomp(benchmark_dir, output_dir, 100);

	eigen_decomp.benchmark();

	system("PAUSE");*/

	ControlDisplay control;
	
	control.run();

	return 0;
}