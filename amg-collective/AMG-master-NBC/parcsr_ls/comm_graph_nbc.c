/*BHEADER**********************************************************************
 * Copyright (c) 2017,  Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 * Written by Ulrike Yang (yang11@llnl.gov) et al. CODE-LLNL-738-322.
 * This file is part of AMG.  See files README and COPYRIGHT for details.
 *
 * AMG is free software; you can redistribute it and/or modify it under the
 * terms of the GNU Lesser General Public License (as published by the Free
 * Software Foundation) version 2.1 dated February 1999.
 *
 * This software is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTIBILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the terms and conditions of the
 * GNU General Public License for more details.
 *
 ***********************************************************************EHEADER*/





/******************************************************************************
 *
 * ParAMG cycling routine
 *
 *****************************************************************************/

#include "_hypre_parcsr_ls.h"
#include "par_amg.h"

#ifdef HYPRE_USING_CALIPER
#include <caliper/cali.h>
#endif


static inline unsigned int * _hypreint_to_int(HYPRE_Int *bigi, int nelem)
{
#ifdef HYPRE_BIGINT
        unsigned int* ret = hypre_CTAlloc(unsigned int, nelem);
        if (!ret) return NULL;

        int iter;
        for (iter = 0; iter < (nelem); iter++)
                ret[iter] = (unsigned int)bigi[iter];
        return ret;
#else /* HYPRE_BIGINT */
        return (bigi);
#endif /* HYPRE_BIGINT */
}

static inline void _check_free(int *arr)
{
#ifdef HYPRE_BIGINT
        hypre_TFree(arr);
        return;
#else /* HYPRE_BIGINT */
        return;
#endif /* HYPRE_BIGINT */

}


void generateGraphNBC(MPI_Comm comm, void *amg_vdata)
{
	HYPRE_Int myid;
	hypre_ParAMGData      *amg_data = (hypre_ParAMGData*) amg_vdata;
	hypre_ParCSRMatrix    **A_array;
	hypre_ParCSRMatrix    **P_array;
	hypre_ParCSRMatrix    **R_array;
	HYPRE_Int             num_levels;
	A_array               = hypre_ParAMGDataAArray(amg_data);
	P_array               = hypre_ParAMGDataPArray(amg_data);
	R_array               = hypre_ParAMGDataRArray(amg_data);
	num_levels            = hypre_ParAMGDataNumLevels(amg_data);
	char                  operator;
	char                  *myEnvString;
        int                   MaxMGLevels;

	hypre_ParCSRMatrix    *A_tmp,*P_tmp,*R_tmp;
	
        int                   doSmootherA=1;
        int                   doInterpolatorP=1;
        int                   doRestrictorR=1;

	MaxMGLevels = (int) num_levels; // all levels

//Set the num levels from the environment variable if provided

        if ((myEnvString=getenv("HPE_CSR_MAX_MG_LEVEL"))!=NULL){
                if (!(sscanf(myEnvString, "%d", &MaxMGLevels))) {
                        printf("Error reading Max MG Levels\n");
                }
                if ( MaxMGLevels > (int) num_levels )
                        MaxMGLevels = (int) num_levels;
        }

//Set the Smoother,Restrictor and  Interpolater based on the environment variable
        if ((myEnvString=getenv("HPE_CSR_SKIP_SMOOTHER_A"))!=NULL){
                doSmootherA=0; // skip it
        }
        if ((myEnvString=getenv("HPE_CSR_SKIP_RESTRICTER_R"))!=NULL){
                doRestrictorR=0; // skip it
        }
        if ((myEnvString=getenv("HPE_CSR_SKIP_INTERPOLATER_P"))!=NULL){
                doInterpolatorP=0; // skip it

        }


	hypre_MPI_Comm_rank(comm, &myid );
	for (int i=0;i<MaxMGLevels;i++){
		A_tmp = A_array[i];	
		R_tmp = R_array[i];
                P_tmp = P_array[i];

		if (A_tmp){
//			printf("calling A\n");
			operator='A';
			templateGraph(operator,doSmootherA, A_tmp);
		}

		if (R_tmp) {
//		printf("calling R\n");
			operator='R';
			templateGraph(operator,doRestrictorR, R_tmp);
		}
		if (P_tmp)  {
//		printf("calling P\n");
			operator='P';
			templateGraph(operator,doInterpolatorP, P_tmp);
		}
	}	
}

void templateGraph(char op,int value,hypre_ParCSRMatrix *mat)
{
	HYPRE_Int      	      *recv_procs,*send_procs;
	MPI_Comm  	      graph_comm,comm_local;
	HYPRE_Int             num_sends, num_recvs;
	int 		      reorder=0;
        int 		      retcode=0;
	int myrank=0;
	int i,j;

	hypre_ParCSRCommPkg   *comm_pkg_mat;
	
	comm_local = hypre_ParCSRMatrixComm(mat);
	 hypre_MPI_Comm_rank(comm_local,&myrank);

	comm_pkg_mat = hypre_ParCSRMatrixCommPkg(mat);	

	if(comm_pkg_mat) {

		num_sends = hypre_ParCSRCommPkgNumSends(comm_pkg_mat);
	        send_procs=hypre_ParCSRCommPkgSendProcs(comm_pkg_mat);
        	num_recvs = hypre_ParCSRCommPkgNumRecvs(comm_pkg_mat);
	        recv_procs=hypre_ParCSRCommPkgRecvProcs(comm_pkg_mat);
		unsigned int *dests = _hypreint_to_int(send_procs, num_sends);
	        unsigned int *srcs = _hypreint_to_int(recv_procs, num_recvs);
		if (op=='A'){
                        hypre_ParCSRCommPkgDoSmoother(comm_pkg_mat) = value;
		retcode = MPI_Dist_graph_create_adjacent(comm_local, num_recvs, srcs,
                                                 MPI_UNWEIGHTED, num_sends, dests , MPI_UNWEIGHTED,
                                                 MPI_INFO_NULL, reorder, &graph_comm);
		hypre_ParCSRCommPkgGraphComm(comm_pkg_mat) = graph_comm;
                }
		if (op=='P'){
                        hypre_ParCSRCommPkgDoInterpolator(comm_pkg_mat) = value;
                        retcode = MPI_Dist_graph_create_adjacent(comm_local, num_recvs, srcs,
                                                 MPI_UNWEIGHTED, num_sends, dests , MPI_UNWEIGHTED,
                                                 MPI_INFO_NULL, reorder, &graph_comm);
                        hypre_ParCSRCommPkgGraphComm(comm_pkg_mat) = graph_comm;
                }
//	        printf("Return code %d of a Matrix \n",retcode);
		if (op=='R'){
                        hypre_ParCSRCommPkgDoRestrictor(comm_pkg_mat) = value;
                        retcode = MPI_Dist_graph_create_adjacent(comm_local, num_sends, dests,
                                                 MPI_UNWEIGHTED,  num_recvs, srcs , MPI_UNWEIGHTED,
                                                 MPI_INFO_NULL, reorder, &graph_comm);
                        hypre_ParCSRCommPkgGraphCommT(comm_pkg_mat) = graph_comm;
                }
	}
}
