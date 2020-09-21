#include "model_manifest.h"

#include <algorithm>

#include <windows.h>
#include <shlwapi.h>

void read_directory(const std::string& dir, std::vector<std::string>& v) {
	std::string pattern(dir);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;

	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			std::string name = data.cFileName;

			if (name.compare(".") == 0) {
				continue;
			} else if (name.compare("..") == 0) {
				continue;
			}

			v.push_back(name);
		} while (FindNextFile(hFind, &data) != 0);

		FindClose(hFind);
	}
}

std::vector<std::string> extension_manifest_from_directory(std::string dir_path, std::string ext) {
	std::vector<std::string> files;
	read_directory(dir_path, files);

	std::vector<std::string> exts;

	for (auto filename : files) {

		std::string fullpath = dir_path + filename;

		if (PathIsDirectoryA(fullpath.c_str())) {
			continue;
		}

		// Check extension
		std::string tail = filename.substr(filename.length() - ext.length(), ext.length());

		std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
		std::transform(tail.begin(), tail.end(), tail.begin(), ::tolower);

		if (tail.compare(ext) == 0) {
			exts.push_back(fullpath);
		}
	}

	return exts;
}

std::vector<std::string> obj_manifest_from_directory(std::string dir_path) {
	return extension_manifest_from_directory(dir_path, "obj");
}