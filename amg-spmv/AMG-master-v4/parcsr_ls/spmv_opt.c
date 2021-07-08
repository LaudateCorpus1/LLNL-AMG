
#include <stdlib.h>

#include "_hypre_parcsr_ls.h"
#ifdef MKL
/* mkl_spblas.h and mkl_service enter through seq_mv.h that is included in _hypre_parcsr_ls.h */
#define SUCCESSFUL_RETURN SPARSE_STATUS_SUCCESS
#endif
#ifdef ARMPL
/* armpl.h enters through seq_mv.h that is included in _hypre_parcsr_ls.h */
#define SUCCESSFUL_RETURN ARMPL_STATUS_SUCCESS
#endif
#include "par_amg.h"

#ifdef HYPRE_USING_CALIPER
#include <caliper/cali.h>
#endif
#define DEBUG 0

void setupCSRHandle( hypre_CSRMatrix *M, int numberOfCalls ) 
{
  interface_int_t        *M_i=NULL;
  interface_int_t        *M_j=NULL;
  HYPRE_Complex          *M_data=NULL;
  interface_int_t        M_num_rows;
  interface_int_t        M_num_cols;
  interface_rtn_status_t status=SUCCESSFUL_RETURN;
#ifdef ARMPL
  armpl_int_t            flags=0;
#endif

  M_i = (interface_int_t *) hypre_CSRMatrixI(M);
  M_j = (interface_int_t *) hypre_CSRMatrixJ(M);
  M_num_rows = (interface_int_t) hypre_CSRMatrixNumRows(M);
  M_num_cols = (interface_int_t) hypre_CSRMatrixNumCols(M);
  M_data =  hypre_CSRMatrixData(M);

  if (M->csrHandle) {
    printf("ERROR: csrHandle exists already!\n");
    exit(1);
  }
  if (M_num_rows && M_num_cols) {
    M->csrHandle = (interface_spmtx_t *) malloc(sizeof(interface_spmtx_t));
#ifdef MKL
    if ((status=mkl_sparse_d_create_csr(M->csrHandle,
					SPARSE_INDEX_BASE_ZERO, M_num_rows, M_num_cols,
					M_i, &(M_i[1]), M_j, M_data )) != SUCCESSFUL_RETURN)
      printf("ERROR: mkl_sparse_d_create_csr returned %d\n", status);
    
    M->csrMtxDescr.type = SPARSE_MATRIX_TYPE_GENERAL;
    M->csrMtxDescr.mode = SPARSE_FILL_MODE_FULL;
    M->csrMtxDescr.diag = SPARSE_DIAG_NON_UNIT;

    if ((status = mkl_sparse_set_mv_hint(*(M->csrHandle),
					 SPARSE_OPERATION_NON_TRANSPOSE, 
					 M->csrMtxDescr, (interface_int_t) numberOfCalls)) != SUCCESSFUL_RETURN)
      printf("ERROR: mkl_sparse_set_mv_hint no-trans returned %d\n", status);

    if ((status = mkl_sparse_optimize(*(M->csrHandle))) != SUCCESSFUL_RETURN)
      printf("ERROR: mkl_sparse_optimize returned %d\n", status);
#endif
#ifdef ARMPL
    if ((status=armpl_spmat_create_csr_d(M->csrHandle,M_num_rows,
					 M_num_cols,M_i,M_j,M_data,
					 flags)) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmat_create_csr_d returned %d\n", status);

    if ((status = armpl_spmat_hint(*(M->csrHandle),ARMPL_SPARSE_HINT_SPMV_OPERATION,
				   ARMPL_SPARSE_OPERATION_NOTRANS)) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmat_hint no-trans returned %d\n", status);

    if ((status = armpl_spmat_hint(*(M->csrHandle), ARMPL_SPARSE_HINT_SPMV_INVOCATIONS,
				   ARMPL_SPARSE_INVOCATIONS_MANY)) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmat_hint invocs returned %d\n", status);

    if ((status = armpl_spmv_optimize(*(M->csrHandle))) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmv_optimize returned %d\n", status);
#endif
  }
}

void setupCSRHandleT( hypre_CSRMatrix *M, int numberOfCalls ) {
  interface_int_t        *M_i=NULL;
  interface_int_t        *M_j=NULL;
  HYPRE_Complex          *M_data=NULL;
  interface_int_t        M_num_rows;
  interface_int_t        M_num_cols;
  interface_rtn_status_t status=SUCCESSFUL_RETURN;
#ifdef ARMPL
  armpl_int_t            flags=0;
#endif

  M_i = (interface_int_t *) hypre_CSRMatrixI(M);
  M_j = (interface_int_t *) hypre_CSRMatrixJ(M);
  M_num_rows = (interface_int_t) hypre_CSRMatrixNumRows(M);
  M_num_cols = (interface_int_t) hypre_CSRMatrixNumCols(M);
  M_data =  hypre_CSRMatrixData(M);

  if (M->csrHandleT) {
    printf("ERROR: csrHandleT  exists  already!\n");
    exit(1);
  }
  if (M_num_rows && M_num_cols) {
    M->csrHandleT = (interface_spmtx_t *) malloc(sizeof(interface_spmtx_t));
#ifdef MKL
    if ((status=mkl_sparse_d_create_csr(M->csrHandleT,
					SPARSE_INDEX_BASE_ZERO, M_num_rows, M_num_cols,
					M_i, &(M_i[1]), M_j, M_data )) != SUCCESSFUL_RETURN)
      printf("ERROR: mkl_sparse_d_create_csr returned %d\n", status);

    M->csrMtxDescr.type = SPARSE_MATRIX_TYPE_GENERAL;
    M->csrMtxDescr.mode = SPARSE_FILL_MODE_FULL;
    M->csrMtxDescr.diag = SPARSE_DIAG_NON_UNIT;

    if ((status = mkl_sparse_set_mv_hint(*(M->csrHandleT),
					 SPARSE_OPERATION_TRANSPOSE, 
					 M->csrMtxDescr, (interface_int_t) numberOfCalls)) != SUCCESSFUL_RETURN)
      printf("ERROR: mkl_sparse_set_mv_hint trans returned %d\n", status);

    if ((status = mkl_sparse_optimize(*(M->csrHandleT))) != SUCCESSFUL_RETURN)
      printf("ERROR: mkl_sparse_optimize returned %d\n", status);
#endif
#ifdef ARMPL
    if ((status=armpl_spmat_create_csr_d(M->csrHandleT,M_num_rows,
					 M_num_cols,M_i,M_j,M_data,
					 flags)) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmat_create_csr_d returned %d\n", status);

    if ((status = armpl_spmat_hint(*(M->csrHandleT),ARMPL_SPARSE_HINT_SPMV_OPERATION,
				   ARMPL_SPARSE_OPERATION_TRANS)) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmat_hint trans returned %d\n", status);

    if ((status = armpl_spmat_hint(*(M->csrHandleT), ARMPL_SPARSE_HINT_SPMV_INVOCATIONS,
				   ARMPL_SPARSE_INVOCATIONS_MANY)) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmat_hint invocs returned %d\n", status);

    if ((status = armpl_spmv_optimize(*(M->csrHandleT))) != SUCCESSFUL_RETURN)
      printf("ERROR: armpl_spmv_optimize returned %d\n", status);
#endif
  }
}

void destroyCSRHandle(interface_spmtx_t *csrHandle) {
  interface_rtn_status_t        status=SUCCESSFUL_RETURN;

#ifdef MKL
  status = mkl_sparse_destroy(*csrHandle);
#endif
#ifdef ARMPL
  status = armpl_spmat_destroy(*csrHandle);
#endif
  free(csrHandle);
}

void generateCSRHandles(MPI_Comm comm, void *amg_vdata) {

  HYPRE_Int           myid;
  hypre_ParAMGData    *amg_data = (hypre_ParAMGData*) amg_vdata;
  hypre_ParCSRMatrix  **A_array;
  hypre_ParCSRMatrix  **P_array;
  hypre_ParCSRMatrix  **R_array;
  HYPRE_Int           num_levels;
  int                 i;
  int                 csrDoDiag=1;
  int                 csrDoOffd=2;
  int                 csrBlocksUsed;      // diag=1, offd=2, both=3 (default)
  int                 csrMaxMGLevels;      // defaults to all levels
  int                 csrDoSmootherA;     // defaults to =1 (true)
  int                 csrDoRestricterR;   // defaults to =1 (true)
  int                 csrDoInterpolaterP; // defaults to =1 (true)
  char                *myEnvString;
  
  A_array           = hypre_ParAMGDataAArray(amg_data);
  P_array           = hypre_ParAMGDataPArray(amg_data);
  R_array           = hypre_ParAMGDataRArray(amg_data);
  num_levels        = hypre_ParAMGDataNumLevels(amg_data);

  /*
   * Allow defaults to be modified by environment variables.
   */
  csrMaxMGLevels = (int) num_levels; // all levels
  csrBlocksUsed = 3;                 // =1+2; both blocks
  csrDoSmootherA=1;                  // do it
  csrDoRestricterR=1;                // do it
  csrDoInterpolaterP=1;              // do it
  if ((myEnvString=getenv("HPE_CSR_MAX_MG_LEVEL"))!=NULL){
    if (!(sscanf(myEnvString, "%d", &csrMaxMGLevels))) {
        printf("Error reading Max MG Levels\n");
    }
    if ( csrMaxMGLevels > (int) num_levels ) 
      csrMaxMGLevels = (int) num_levels;
  }
  if ((myEnvString=getenv("HPE_CSR_BLOCKS_USED"))!=NULL){
    if (!(sscanf(myEnvString, "%d", &csrBlocksUsed))) {
        printf("Error reading CSR Blocks Used\n");
    }
    if ((csrBlocksUsed<1) || (csrBlocksUsed>3)) csrBlocksUsed=3;
  }
  if ((myEnvString=getenv("HPE_CSR_SKIP_SMOOTHER_A"))!=NULL)
    csrDoSmootherA=0; // skip it
  if ((myEnvString=getenv("HPE_CSR_SKIP_RESTRICTER_R"))!=NULL)
    csrDoRestricterR=0; // skip it
  if ((myEnvString=getenv("HPE_CSR_SKIP_INTERPOLATER_P"))!=NULL)
    csrDoInterpolaterP=0; // skip it

  for (i=0;i<csrMaxMGLevels;i++) {
    if ((A_array[i]) && csrDoSmootherA) {
      if (csrBlocksUsed & csrDoDiag)
	setupCSRHandle(hypre_ParCSRMatrixDiag(A_array[i]),150);
      if (csrBlocksUsed & csrDoOffd)
	setupCSRHandle(hypre_ParCSRMatrixOffd(A_array[i]),150);
    }
    if (i != num_levels-1) {
      /* 
       * R_array[i] is an alias for P_array[i], but it is only meant
       * for use in transposed SpMV calls:
       *       y -> alpha*A^T*x + beta*y
       *
       * For alias of P_array[i] see line 385 in parcsr_ls/par_amg_setup.c
       * For use only in transpose see line 431 in parcsr_ls/par_cycle.c
       */

      if ((R_array[i]) && csrDoRestricterR) {
	if (csrBlocksUsed & csrDoDiag)
	  setupCSRHandleT(hypre_ParCSRMatrixDiag(R_array[i]),30);
	if (csrBlocksUsed & csrDoOffd)
	  setupCSRHandleT(hypre_ParCSRMatrixOffd(R_array[i]),30);
      }
    }
    if (i !=0)  {
      if ((P_array[i-1]) && csrDoInterpolaterP) {
	if (csrBlocksUsed & csrDoDiag)
	  setupCSRHandle(hypre_ParCSRMatrixDiag(P_array[i-1]),30);
	if (csrBlocksUsed & csrDoOffd)
	  setupCSRHandle(hypre_ParCSRMatrixOffd(P_array[i-1]),30);
      }
    }
  }
}
