#include "reax_types.h"
#include "allocate.h"
#include <sys/stat.h>
#include "tool_box.h"
#include <time.h>
#include "lapacke.h"
#include "cblas.h"



/* store the model parameters */
struct ELM_model
{
	// flattened matrices
	double* W_in; // (Window_size) X (hidden layer size)
	double* W_out; // (hidden layer size) X (output size)
	int W_size, hidden_size;
};
typedef struct ELM_model ELM_model;

/* TODO: fix the global parameter later */
static ELM_model ELM_model_top;
static ELM_model ELM_model_bot;

void RELU(double* input, int size)
{
	int i;
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        private(i)
#endif   
	for (i = 0; i < size; i++)
	{
		input[i] = MAX(0.0, input[i]);
	}
}


void input_to_hidden(double * output, double* input, double* W_in, int input_size,  
						int W_size, int hidden_size)
{
	// out = input X W_in
	// out = RELU(out)
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                input_size, hidden_size, W_size, //M N K
        	 	1.0, input, W_size, W_in, hidden_size, 0.0, output, hidden_size);

	RELU(output, input_size * hidden_size);

}

void predict(static_storage * const workspace, ELM_model* model, double* output, double* input, int input_size) 
{
	//double* hidden_out = smalloc( sizeof(double) * input_size * model->hidden_size, 
    //                                                "ELM_regressor:hidden_out" );
    double* hidden_out = workspace->flat_input_hidden;
	
    input_to_hidden(hidden_out, input, model->W_in, input_size, model->W_size, model->hidden_size);
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                input_size, 1, model->hidden_size, //M N K
        	 	1.0, hidden_out, model->hidden_size, model->W_out, 1, 0.0, output, 1);

	//sfree( hidden_out, "batch_predict:hidden_out" ); 
    

}

void batch_predict(ELM_model* model, 
        int atom_start, int atom_end, 
        const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, 
        static_storage * const workspace)
{
    int win_size = control->cm_init_guess_win_size;

    double* obs_flat_in, *predictions;

    int i = 0;

    if (atom_end > system->N_cm-1){
        atom_end = system->N_cm-1;
    }


    int batch_size = atom_end - atom_start; // start inclusive,end exclusive


    //obs_flat_in = smalloc( sizeof(double) * batch_size * win_size, "batch_predict:obs_flat" );
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        private(i)
#endif  
    for ( i = atom_start; i < atom_end; ++i )
    {
        int arr_start = i - atom_start;
        for (int j = 0; j < win_size; ++j ) 
        {
            // it needs to be reversed bc of the way the model trained
            // note that workspace->s is shifted (reason of extra  + 1)
            //first order diff.
            double val1 = workspace->s[win_size - j - 1 + 1][i];
            double val2 = workspace->s[win_size - j + 1][i];
            workspace->flat_input[arr_start * win_size + j] = val1 - val2;     
        }
    }
    //predictions = smalloc( sizeof(double) * batch_size, "batch_predict:predictions" );
    predict(workspace, model, workspace->predictions, workspace->flat_input, batch_size);
#ifdef _OPENMP
    #pragma omp parallel for schedule(static) \
        private(i)
#endif 
    for (i = atom_start; i < atom_end; ++i )
    {
        int arr_start = i - atom_start;
        workspace->s[0][i] = workspace->predictions[arr_start] + workspace->s[1][i];
    }

    //sfree( predictions, "batch_predict:predictions" ); 
    //sfree( obs_flat_in, "batch_predict:obs_flat_in" ); 
}

void train(ELM_model model, double* input) 
{
	//TODO
}

void read_2d_array(char* filename, double** result, int* n_rows, int* n_cols)
{
    FILE *fp = sfopen( filename, "r" );
    int i, size;
    double val;

    if ( fp != NULL )
    {

        fscanf (fp, "%d %d", n_rows, n_cols);
        size = (*n_rows) * (*n_cols);
        *result = smalloc( sizeof(double) * size, "batch_predict:result");
        
        for (i = 0; i < size; i++) {
            fscanf (fp, "%lf", &val);
            (*result)[i] = val;
        } 
        sfclose( fp , "batch_predict:fp");          
    }
    else {
        fprintf(stderr, "File %s couldnt open!\n", filename); 
    }
      
}

void read_model(ELM_model *model, int win_size, char* W_in_file, char* W_out_file)
{   
    int hidden_size;
    int out_size;
	read_2d_array(W_in_file, &(model->W_in), &(model->W_size), &(model->hidden_size));
    read_2d_array(W_out_file, &(model->W_out), &(hidden_size), &(out_size));

    if (hidden_size != model->hidden_size || out_size != 1 || win_size != model->W_size) {
        fprintf(stderr, "read_model failed, matrix sizes dont match!\n");
    }
}



static void Predict_Charges_ELM_EE( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int atom_batch = 10000;
    real time;
    int i = 0;

    if (ELM_model_top.W_in == NULL) {
        //openblas_set_num_threads(8);
        read_model(&ELM_model_top, control->cm_init_guess_win_size,
         "W_in.txt", "W_out.txt");
        fprintf(stdout, "Model loaded!\n");
        fprintf(stdout, "%f\n", ELM_model_top.W_in[0]);
        fprintf(stdout, "%f\n", ELM_model_top.W_out[0]);
    }

    time = Get_Time( );
    //#ifdef _OPENMP
    //    #pragma omp parallel for schedule(static) \
    //        private(i)
    //#endif    
    //for (i = 0; i < system->N_cm - 1; i = i + atom_batch) {
    batch_predict(&ELM_model_top, i, i + system->N_cm-2, system,
    control,data, workspace);      
    //}
     

    data->timing.cm_tensorflow_just_prediction = Get_Timing_Info( time );
}

static void Predict_Charges_ELM_ACKS2( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{
    int atom_batch = 10000;
    real time;
    int i = 0;

    if (ELM_model_bot.W_in == NULL) {
        //openblas_set_num_threads(8);
        read_model(&ELM_model_bot, control->cm_init_guess_win_size,
         "W_in.txt", "W_out.txt");
        fprintf(stdout, "Model loaded!\n");
        fprintf(stdout, "%f\n", ELM_model_bot.W_in[0]);
        fprintf(stdout, "%f\n", ELM_model_bot.W_out[0]);
    }

    time = Get_Time( );
    //#ifdef _OPENMP
    //    #pragma omp parallel for schedule(static) \
    //        private(i)
    //#endif    
    //for (i = (system->N_cm-2)/2; i < system->N_cm-2; i = i + atom_batch) {
    batch_predict(&ELM_model_bot, i, i + system->N_cm-1, system,
    control,data, workspace);      
    //}
     

    data->timing.cm_tensorflow_just_prediction = Get_Timing_Info( time );
}




