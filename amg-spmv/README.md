AMG with SpMV refactored code: 

Default setup in Makefile.include enables MKL (APL is commented out).

Selective control of library replacement calls is enabled by 
using environment variables:

      export HPE_CSR_MAX_MG_LEVEL=2
      	  Example restricting replacement calls to level 0 and 1 only.
	  Valid values between 0 (all replacements disabled)
	  and actual number of levels.

      export HPE_CSR_BLOCKS_USED=2
      	  Example restricting replacement calls to the offd blocks only.
	  This is a 2-bit mask with valid values of {01,10,11}.
	  The following values are used:
	  01 = 0x1 = diag is replaced
	  10 = 0x2 = offd is replaced
	  11 = 0x3 = diag and offd are replaced
	     
      export HPE_CSR_SKIP_SMOOTHER_A=true
      	     Replacement of call for smoother matrix A is skipped.
	     Actual value of variable is ignored (anything),
	     check is for whether variable is set.
	     
      export HPE_CSR_SKIP_RESTRICTER_R=true
      	     Replacement of call for restricter matrix R is skipped.
	     Actual value of variable is ignored (anything),
	     check is for whether variable is set.
	     
      export HPE_CSR_SKIP_INTERPOLATER_P=true
      	     Replacement of call for interpolater matrix P is skipped.
	     Actual value of variable is ignored (anything),
	     check is for whether variable is set.
