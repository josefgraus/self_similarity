#ifndef GEN_ID_H_
#define GEN_ID_H_

#include <string>

#include <Objbase.h>

std::string generate_guid() {
	GUID guid;
	OLECHAR* guidString;
	std::wstring guid_string_w;
	std::string guid_string;

	CoCreateGuid(&guid);
	StringFromCLSID(guid, &guidString);
	guid_string_w = guidString;
	CoTaskMemFree(guidString);
	guid_string = std::string(guid_string_w.begin(), guid_string_w.end());

	// Remove brackets
	guid_string = guid_string.substr(guid_string.find_first_of("{") + 1, guid_string.find_last_of("}") - (guid_string.find_first_of("{") + 1));

	return guid_string;
}

#endif