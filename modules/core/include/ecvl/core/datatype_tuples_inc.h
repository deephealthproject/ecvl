// Existing DataType MUST be included before none DataType, 
// otherwise the Table logic will be broken, as well as the in-place
// Neg function.

#include "datatype_existing_tuples_inc.h"
ECVL_TUPLE(none /**< none type  */,    0, void)