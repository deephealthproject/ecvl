#include "ecvl/core/filesystem.h"

#include <algorithm>
#include <fstream>
#include <string>

#include <sys/stat.h>
#include <sys/types.h>

using namespace std;

namespace filesystem {
const char path::separator_ =
#ifdef ECVL_WINDOWS
'\\';
#else
'/';
#endif

bool exists(const path& p)
{
    struct stat info;
    string s = p.string();
    const char* path_to_check = s.c_str();
    if (stat(path_to_check, &info) != 0) {
        //printf("cannot access %s\n", pathname);
        return false;
    }
    else if (info.st_mode & S_IFDIR) {
        //printf("%s is a directory\n", pathname);
        return true; // is directory
    }

    //printf("%s is no directory\n", pathname);
    return true; // is file
}

bool exists(const path& p, error_code& ec)
{
    return exists(p);
}

bool create_directories(const path& p)
{
    string s(p.string());
    string parameters = "";
#if defined(ECVL_UNIX) || defined(ECVL_LINUX) || defined(ECVL_APPLE)
    // make it recursive by adding "-p" suffix
    parameters = "-p";
#endif

    if (!exists(path(s))) {
        if (system(("mkdir " + parameters + " \"" + s + "\"").c_str()) != 0) {
            //cerr << "Unable to find/create the output path " + s;
            return false;
        }
    }
    return true;
}

bool create_directories(const path& p, error_code& ec)
{
    return create_directories(p);
}

void path::NormalizePath()
{
#if defined(ECVL_UNIX) || defined(ECVL_LINUX) || defined(ECVL_APPLE)
    std::replace(path_.begin(), path_.end(), '\\', '/');
#elif defined(ECVL_WINDOWS)
    std::replace(path_.begin(), path_.end(), '/', '\\');
#endif

    return;
}

void copy(const path& from, const path& to)
{
    if (!exists(from)) {
        return;
    }

    if (!exists(to)) {
        if (create_directories(to.parent_path())) {
            ifstream src(from.string());
            ofstream dst(to.string());

            dst << src.rdbuf();
        }
    }
}

void copy(const path& from, const path& to, error_code& ec)
{
    copy(from, to);
}

} // namespace filesystem