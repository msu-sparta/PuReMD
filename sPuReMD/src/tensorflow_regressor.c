#include "tensorflow/c/c_api.h"
#include "reax_types.h"
#include "allocate.h"
#include <sys/stat.h>

#define MY_TENSOR_SHAPE_MAX_DIM (16)
// Defined in reax_types.h
// TODO: provide a single source for this
// it is also defined in "init_md.c" file 
//#define WINDOW_SIZE 401   

// TODO: Move these to reax_types.c (?)
typedef struct TF_model TF_model;
typedef struct tensor_shape tensor_shape;
enum checkpoint_state {SAVE = 1, RESTORE = 2};

typedef float DATA_PRECISION;
#define TF_DATA_TYPE (TF_FLOAT)

/*
To store tensorflow model and operations
*/
struct TF_model
{
    TF_Session* session;
    TF_Graph* graph;
    TF_Status* status;

    TF_Output input, target, output, learning_rate;
    TF_Operation *init_op, *train_op, *save_op, *restore_op, *loss_op;
    TF_Output checkpoint_file;
};


struct tensor_shape
{
    int64_t values[MY_TENSOR_SHAPE_MAX_DIM];
    int dim;

//    int64_t size(){
//        assert(dim>=0);
//        int64_t v=1;
//        for(int i=0;i<dim;i++)
//          v*=values[i];
//        return v;
//    }
};

// TODO: global scope, FIX IT LATER
// one model for the top and other for the bottom
static TF_model model_top;
static TF_model model_bot;
/*
Required to get batch of data without repetition in a randomly fashion
*/
static void create_non_repetetive_random_numbers(int start, int end, int* arr) {
    int size = end - start;
    // init
    for (int i = 0; i < size; i++){
        arr[i] = i;
    }
    // randomly swap
    for (int i = 0; i < size; i++){
        int rand_ind = rand() % size;
        //swap
        int temp = arr[i];
        arr[i] = arr[rand_ind];
        arr[rand_ind] = temp;
    }

}

// print the error if status has an eror msg
static int Okay(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: %s\n", TF_Message(status));
    return 0;
  }
  return 1;
}

static void TF_Tensor_Deallocator( void* data, size_t length, void* arg )
{
    sfree( data, "TF_Tensor_Deallocator::data" );
}


static void TF_free( void* data, size_t length )
{
        sfree( data, "TF_free::data" );
}

/*
Convert string to tensor (needed for filenames etc.)
*/
TF_Tensor* ScalarStringTensor(const char* str, TF_Status* status) {
  size_t nbytes = 8 + TF_StringEncodedSize(strlen(str));
  TF_Tensor* t = TF_AllocateTensor(TF_STRING, NULL, 0, nbytes);
  void* data = TF_TensorData(t);
  memset(data, 0, 8);  // 8-byte offset of first string.
  TF_StringEncode(str, strlen(str), data + 8, nbytes - 8, status);
  return t;
}

int DirectoryExists(const char* dirname) {
  struct stat buf;
  return stat(dirname, &buf) == 0;
}

// min-max normalization of the flattened data (store the min max values in the given arrays after the operation is done)
static void min_max_normalize(double* obs_flat, int win_size, int batch_size, double* min_array, double* max_array) {
    int i;
    // find min and max values
    for ( i = 0; i < batch_size; ++i )
    {
        min_array[i] = obs_flat[i * win_size + 0];
        max_array[i] = obs_flat[i * win_size + 0];
        for (int j = 0; j < win_size; ++j ) 
        {
            if (obs_flat[i * win_size + j] > max_array[i]) {
                max_array[i] = obs_flat[i * win_size + j];
            }  
            if (obs_flat[i * win_size + j] < min_array[i])   {
                min_array[i] = obs_flat[i * win_size + j];
            }  
        }
    }
    // 
    for ( i = 0; i < batch_size; ++i )
    {
        double min_max_diff = max_array[i] - min_array[i];
        for (int j = 0; j < win_size; ++j ) 
        {
             obs_flat[i * win_size + j] = (obs_flat[i * win_size + j] - min_array[i]) / min_max_diff;
        }
    }   
}

// reverse the min-max normalization
static void min_max_reverse(double* predictions, int win_size, int batch_size, double* min_array, double* max_array) {
    int i;
    for (i = 0; i < batch_size; ++i )
    {
            predictions[i] = predictions[i] * ( max_array[i] - min_array[i]);
            predictions[i] = predictions[i] + min_array[i];
    }
}
// standard normalization of the flattened data (store the mean and std values in the given arrays after the operation is done)
static void standard_normalize(DATA_PRECISION* obs_flat, int win_size, int batch_size, 
                              double* mean_array, double* std_array) {
    int i;
    // find mean and std values
    for ( i = 0; i < batch_size; ++i )
    {
        mean_array[i] = 0;
        for (int j = 0; j < win_size; ++j ) 
        {
            mean_array[i] = mean_array[i] + obs_flat[i * win_size + j];
 
        }
        mean_array[i] = mean_array[i] / win_size;
    }
    for ( i = 0; i < batch_size; ++i )
    {
        std_array[i] = 0;
        for (int j = 0; j < win_size; ++j ) 
        {
            std_array[i] = std_array[i] + (obs_flat[i * win_size + j] - mean_array[i]) * (obs_flat[i * win_size + j] - mean_array[i]);
 
        }

        std_array[i] = std_array[i] / (win_size);
        std_array[i] = SQRT(std_array[i]);
        if (std_array[i] == 0){
           std_array[i] = 1;
           //fprintf( stdout, "[ERROR] encountered with 0 std: %d \n", i);

        }

    }

    // 
    for ( i = 0; i < batch_size; ++i )
    {
        for (int j = 0; j < win_size; ++j ) 
        {
             obs_flat[i * win_size + j] = (obs_flat[i * win_size + j] - mean_array[i]) / std_array[i];
        }
    }   
}

// reverse the standard normalization
static void standard_reverse(DATA_PRECISION* predictions, int win_size, int batch_size, 
                            double* mean_array, double* std_array) {
    int i;
    for (i = 0; i < batch_size; ++i )
    {
            predictions[i] = predictions[i] * ( std_array[i]);
            predictions[i] = predictions[i] + mean_array[i];
    }
}



/* Read the entire content of a file and return it as a TF_Buffer.
 *
 * @param file: The file to be loaded.
 * @return
 */
static TF_Buffer* read_file_to_TF_Buffer( const char* file )
{
    FILE *f;
    void *data;
    long fsize;
    TF_Buffer* buf;

    f = fopen( file, "rb" );
    fseek( f, 0, SEEK_END );
    fsize = ftell( f );
    fseek( f, 0, SEEK_SET );  //same as rewind(f);

    data = malloc( fsize );
    fread( data, fsize, 1, f );
    fclose( f );

    buf = TF_NewBuffer();
    buf->data = data;
    buf->length = fsize;
    buf->data_deallocator = &TF_free;

    return buf;
}


/* Load a GraphDef from a provided file.
 *
 * @param filename: The file containing the protobuf encoded GraphDef
 * @param input_name: The name of the input placeholder
 * @param output_name: The name of the output tensor
 * @return
 */
TF_model model_load( const char *filename,
        const char *input_name, const char *output_name,const char *target_name,
        const char *init_name, const char *train_name, 
        const char *loss_name, const char *learning_rate_name, 
        const char *save_name,const char *restore_name, const char *checkpoint_name)
{
    TF_Buffer *graph_def;
    TF_ImportGraphDefOptions *opts;
    TF_SessionOptions *sess_opts = TF_NewSessionOptions( );

    TF_model model;


    graph_def = read_file_to_TF_Buffer( filename );
    model.graph = TF_NewGraph( );

    // Import graph_def into graph
    model.status = TF_NewStatus( );
    opts = TF_NewImportGraphDefOptions( );
    TF_GraphImportGraphDef( model.graph, graph_def, opts, model.status );

    if ( TF_GetCode(model.status) != TF_OK )
    {
        fprintf( stderr, "[ERROR] unable to import graph: %s\n", TF_Message(model.status) );
        exit( INVALID_INPUT );
    }

    model.input.oper = TF_GraphOperationByName( model.graph, input_name );
    model.input.index = 0;
    model.output.oper = TF_GraphOperationByName( model.graph, output_name );
    model.output.index = 0;
    model.target.oper = TF_GraphOperationByName(model.graph, target_name);
    model.target.index = 0; 
    model.learning_rate.oper = TF_GraphOperationByName(model.graph, learning_rate_name);
    model.learning_rate.index = 0; 

    model.init_op = TF_GraphOperationByName(model.graph, init_name);
    model.train_op = TF_GraphOperationByName(model.graph, train_name);
    model.save_op = TF_GraphOperationByName(model.graph, save_name);
    model.restore_op = TF_GraphOperationByName(model.graph, restore_name);
    model.loss_op = TF_GraphOperationByName(model.graph, loss_name);

    model.checkpoint_file.oper = TF_GraphOperationByName(model.graph, checkpoint_name);
    model.checkpoint_file.index = 0;

    //TODO: Create extra parameters for the control file which controls whether to activate training or not
    if ( !model.input.oper || !model.output.oper)
    {
        fprintf( stderr, "[ERROR] !input_op || !output_op\n" );
        exit( INVALID_INPUT );
    }
    
    if (!model.target.oper || !model.train_op || !model.save_op)
    {
        fprintf( stderr, "[ERROR] !target || !train_op || !save_op\n" );
        exit( INVALID_INPUT );
    }
    if (!model.restore_op || !model.checkpoint_file.oper)
    {
        fprintf( stderr, "[ERROR] !restore || !checkpoint_file\n" );
        exit( INVALID_INPUT );
    }
    if (!model.loss_op || !model.learning_rate.oper)
    {
        fprintf( stderr, "[ERROR] !loss_op || !learning_rate_op\n" );
        exit( INVALID_INPUT );
    }
    

    // reverse engineered the TF_SetConfig protocol from python code like:
    // >> config = tf.ConfigProto();config.intra_op_parallelism_threads=7;config.SerializeToString()
    // '\x10\x07'

    sess_opts = TF_NewSessionOptions( );
    uint8_t intra_op_parallelism_threads = 2;
    uint8_t inter_op_parallelism_threads = 14;
    uint8_t buf[]={0x10,intra_op_parallelism_threads,0x28,inter_op_parallelism_threads};
    TF_SetConfig(sess_opts, buf,sizeof(buf),model.status);

    model.session = TF_NewSession( model.graph, sess_opts, model.status );
    if ( TF_GetCode(model.status) != TF_OK )
    {
        fprintf( stderr, "[ERROR] unable to start a sesssion: %s\n", TF_Message(model.status) );
        exit( INVALID_INPUT );
    }


    TF_DeleteImportGraphDefOptions( opts );
    TF_DeleteBuffer( graph_def );
    TF_DeleteSessionOptions( sess_opts );

    return model;
}
/*
Function to use save or restore the weights after the model-graph is initialized
*/
static int model_checkpoint(TF_model* my_model, const char* checkpoint_prefix, enum checkpoint_state type) {
  TF_Tensor* t = ScalarStringTensor(checkpoint_prefix, my_model->status);
  if (!Okay(my_model->status)) {
    TF_DeleteTensor(t);
    return 0;
  }
  TF_Output inputs[1] = {my_model->checkpoint_file};
  TF_Tensor* input_values[1] = {t};
  const TF_Operation* op[1] = {type == SAVE ? my_model->save_op
                                            : my_model->restore_op};
  TF_SessionRun(my_model->session, NULL, inputs, input_values, 1,
                /* No outputs */
                NULL, NULL, 0,
                /* The operation */
                op, 1, NULL, my_model->status);
  TF_DeleteTensor(t);
  return Okay(my_model->status);
}

static int restore_model_parameters(TF_model* my_model, const char* checkpoint_prefix) {
    return model_checkpoint(my_model, checkpoint_prefix, RESTORE);
}

static int save_model_parameters(TF_model* my_model, const char* checkpoint_prefix) {
    return model_checkpoint(my_model, checkpoint_prefix, SAVE);
}

static void get_1diff_training_data(double** saved_data, DATA_PRECISION** training_data,
                                   int start_atom_ind, int end_atom_ind,int data_size, int window_size,
                                  int sample_size){

  // assume training is called in pre-shifted time (right after a solve)
  // history data is reversed (index 0 is the most recent)
  int sample_index = 0;

  for (int i = start_atom_ind; i < end_atom_ind; i++){
    double diff_array[data_size-1];
    for (int j = 0; j < data_size-1; j++){
        double val1 = saved_data[data_size - j - 1][i];
        double val2 = saved_data[data_size - j - 2][i];
        diff_array[j] = val2 - val1;       
    }
    for(int start = 0; start < data_size - window_size - 1; start++){
      for(int s_ind = 0; s_ind < window_size + 1; s_ind++){
          training_data[sample_index][s_ind] = diff_array[start + s_ind];
      }
      sample_index = sample_index + 1;
      if (sample_index >= sample_size){
         fprintf( stdout, "[ERROR] unexpected sample ");
      }
    }
  }
}
static void get_batch(TF_Tensor** inputs_tensor, TF_Tensor** targets_tensor,
                      DATA_PRECISION** one_diff_training_data, int* selected_indices, 
                      int full_size, int batch_start_ind, int batch_size, int window_size){

  DATA_PRECISION* inputs = smalloc( sizeof(DATA_PRECISION) * batch_size * window_size, 
                                                    "Predict_Charges_TF_LSTM:inputs" );
  DATA_PRECISION* targets = smalloc(sizeof(DATA_PRECISION) * batch_size * 1, 
                                                    "Predict_Charges_TF_LSTM:targets" );

  tensor_shape input_shape;
  input_shape.values[0] = batch_size;
  input_shape.values[1] = window_size;
  input_shape.values[2] = 1;
  input_shape.dim = 3;

  tensor_shape output_shape;
  output_shape.values[0] = batch_size;
  output_shape.values[1] = 1;
  output_shape.dim = 2; 
  for (int i = 0; i < batch_size; i++){
    int start = selected_indices[batch_start_ind + i];
    for(int j = 0; j < window_size; j++) {
      inputs[i * window_size + j] = one_diff_training_data[start][j];
      //fprintf( stdout, "%d, ", inputs[i * window_size + j]);
    }
    targets[i] = one_diff_training_data[start][window_size];
    //fprintf( stdout, "%d \n", targets[i]);
  }
  double means[batch_size];
  double stds[batch_size];
  // normalize the data
  //standard_normalize(inputs, window_size, batch_size, means, stds);

  //for (int i = 0; i < batch_size; i++){
  // targets[i] = (targets[i] - means[i]) / stds[i];
  //} 

  size_t nbytes_input = batch_size * window_size * sizeof(DATA_PRECISION);
  size_t nbytes_target = batch_size * sizeof(DATA_PRECISION);

  *inputs_tensor = TF_NewTensor( TF_DATA_TYPE, input_shape.values, input_shape.dim,
            (void *)inputs, sizeof(DATA_PRECISION) * window_size * batch_size,
            &TF_Tensor_Deallocator, NULL );

  *targets_tensor = TF_NewTensor( TF_DATA_TYPE, output_shape.values, output_shape.dim,
            (void *)targets, sizeof(DATA_PRECISION) * 1 * batch_size,
            &TF_Tensor_Deallocator, NULL );

}

static double calculate_loss(TF_model* model, TF_Tensor *x, TF_Tensor* y, int num_samples, int win_size) {
    tensor_shape input_shape;
    TF_Tensor *output_tensor[1];
    TF_Tensor *input_tensor[1];
    TF_Output inputs[1] = {model->input};
    TF_Output outputs[1] = {model->output};

    input_shape.values[0] = num_samples;
    input_shape.values[1] = win_size;
    input_shape.values[2] = 1;
    input_shape.dim = 3;

    output_tensor[0] = NULL;
    input_tensor[0] = x;

    //printf(" before TF_SessionRun\n");
    TF_SessionRun( model->session, NULL,
        &inputs[0], input_tensor, 1,
        &outputs[0], output_tensor, 1,
        NULL, 0, NULL, model->status );

    // error happened in the run
    if (!Okay(model->status)) {
        return -1;
    }

    DATA_PRECISION * predictions = (DATA_PRECISION*) TF_TensorData( output_tensor[0] );
    DATA_PRECISION * true_y = (DATA_PRECISION*) TF_TensorData( y );

    //calculate error
    double total_error = 0;

    for (int i = 0; i < num_samples; i++) {
        double err = predictions[i] - true_y[i];
        total_error = total_error + err * err;
    }

    // get the average
    total_error = total_error / num_samples;

    return total_error;




}

/*
data_size is the number of MD steps so far
*/
static int train_model(TF_model* model, static_storage * const workspace, const reax_system * const system, 
                      int start_atom_ind, int end_atom_ind, int data_size, int window_size, int epoch, int batch_size) {
    
    float learnning_rate_fl = 1e-3;
    float *learning_rate_arr;
    int non_decrease_LR_limit = 2;
    int non_decrease_LR_count = 0;
    tensor_shape LR_shape;
    LR_shape.values[0] = 1;
    LR_shape.dim = 1;
    double prev_loss = 99999;
    fprintf( stdout, "[INFO] Initial learning rate: %f\n",learnning_rate_fl);
    //int num_samples = (system->N_cm-2)/2 * (data_size - window_size);
    // start inclusive, end exlusive 
    int num_samples = (end_atom_ind - start_atom_ind) * (data_size - window_size); 
    fprintf( stdout, "[INFO] Number of samples: %d \n",num_samples ); 
    DATA_PRECISION** one_diff_training_data = scalloc( num_samples, sizeof( DATA_PRECISION* ),
                              "tensorflow_regressor::flattened_array" );
    for (int i = 0; i < num_samples; i++){
        one_diff_training_data[i] = scalloc( window_size + 1, sizeof( DATA_PRECISION ),
                              "tensorflow_regressor::flattened_array" );
    }

    DATA_PRECISION** one_diff_training_data2 = scalloc( num_samples, sizeof( DATA_PRECISION* ),
                              "tensorflow_regressor::flattened_array" );
    for (int i = 0; i < num_samples; i++){
        one_diff_training_data2[i] = scalloc( window_size + 1, sizeof( DATA_PRECISION ),
                              "tensorflow_regressor::flattened_array" );
    }
    get_1diff_training_data(workspace->s, one_diff_training_data, start_atom_ind, end_atom_ind, data_size, window_size, num_samples);
    //get_1diff_training_data(workspace->s_14, one_diff_training_data2, start_atom_ind, end_atom_ind, data_size, window_size, num_samples);

    // We need to make sure we can divide the array into equal parts
    // WINDOW_SIZE - 1 should be a multiple of window size

    TF_Output inputs[3] = {model->input, model->target, model->learning_rate};
    const TF_Operation* train_op[1] = {model->train_op};
    fprintf( stdout, "train_model before epoch\n" );
    for (int i = 0; i < epoch; i++){
      // produce random order for the epoch
      TF_Tensor *x,*x_2, *y;
      TF_Tensor *learning_rate_tensor;
      learning_rate_arr = smalloc( sizeof(float) * 1, 
                                                        "Predict_Charges_TF_LSTM:learning_rate" );
      learning_rate_arr[0] = learnning_rate_fl;

      learning_rate_tensor = TF_NewTensor(TF_FLOAT, NULL, 0, learning_rate_arr, sizeof(float),
                      &TF_Tensor_Deallocator, NULL);

      int selected_order[num_samples];
      create_non_repetetive_random_numbers(0, num_samples, selected_order);
      
      for (int j = 0; j < num_samples - batch_size; j = j + batch_size){
        get_batch(&x, &y, one_diff_training_data, selected_order,  
                        num_samples, j, batch_size, window_size);
        //get_batch(&x_2, &y, one_diff_training_data, selected_order,  
        //           num_samples, j, batch_size, window_size);
        TF_Tensor* input_values[3] = {x, y,learning_rate_tensor};
        TF_SessionRun(model->session, NULL, inputs, input_values, 3,
                /* No outputs */
                NULL, NULL, 0, train_op, 1, NULL, model->status);

        TF_DeleteTensor(x);
        TF_DeleteTensor(x_2);
        TF_DeleteTensor(y);    
        if (!Okay(model->status)) {
            break;
        }
      }

      // get the loss in that batch and print it
      get_batch(&x, &y, one_diff_training_data, selected_order,  
                        num_samples, 0, num_samples, window_size);
      double loss = calculate_loss(model, x, y, num_samples, window_size);

      printf("[INFO] Epoch: %d, Avg. Squared Loss: %.14f\n",i, loss);

      if (prev_loss < loss) {
        non_decrease_LR_count = non_decrease_LR_count + 1; 
        //fprintf(stderr, "non_decrease_LR_count %d\n", non_decrease_LR_count);
      }
      else {
        non_decrease_LR_count = 0;
      }

      if (non_decrease_LR_count == non_decrease_LR_limit) {
         non_decrease_LR_count = 0;
         learnning_rate_fl = learnning_rate_fl/5;
         fprintf( stdout, "[INFO] LR decrease: %f\n",learnning_rate_fl);
      }
      prev_loss = loss;
      TF_DeleteTensor(x);
      TF_DeleteTensor(y); 
      TF_DeleteTensor(learning_rate_tensor);

    }
    for (int i = 0; i < num_samples; i++){
        free(one_diff_training_data[i]);
        free(one_diff_training_data2[i]);
    }
    free(one_diff_training_data);
    free(one_diff_training_data2);
    return 1; // succesful
}

static int train_and_save(static_storage * const workspace, 
  const reax_system * const system, const control_params * const control, int data_size) {
  int epoch = control->cm_init_guess_training_epoch;
  int batch_size = 512;
  int window_size = control->cm_init_guess_win_size;
  // TODO: using the global model FIX IT LATER
  const char* checkpoint_prefix = "./checkpoints/checkpoint";
  printf("data size: %d\n", data_size);
  if (train_model(&model_top, workspace, system, 0, (system->N_cm-2)/2, data_size, window_size, epoch/2, batch_size)) {
      if(save_model_parameters(&model_top, checkpoint_prefix)) {
         fprintf( stdout, "[INFO] Model is trained and saved to ./checkpoints/checkpoint");
         //return 1;
      }
      else {
        fprintf( stdout, "[ERROR] Couldnt save");
         return 0;
      }
  }
  else {
    fprintf( stdout, "[ERROR] Couldnt train");
    return 0;
  }

  if (train_model(&model_bot, workspace, system, (system->N_cm-2)/2, system->N_cm-2, data_size, window_size, epoch, batch_size)) {
      if(save_model_parameters(&model_bot, checkpoint_prefix)) {
         fprintf( stdout, "[INFO] Model is trained and saved to ./checkpoints/checkpoint");
         return 1;
      }
      else {
        fprintf( stdout, "[ERROR] Couldnt save");
         return 0;
      }
  }
  else {
    fprintf( stdout, "[ERROR] Couldnt train");
    return 0;
  }
}

static void batch_prediction(TF_model *my_model, int atom_start, int atom_end, const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace)
{
    int win_size = control->cm_init_guess_win_size;
    tensor_shape input_shape;
    TF_Tensor *input_tensor[1];
    TF_Tensor* output_tensor[1];
    TF_Output inputs[1] = {my_model->input};
    TF_Output outputs[1] = {my_model->output};
    DATA_PRECISION* obs_flat_in;
    int i = 0;

    if (atom_end > system->N_cm-2){
        atom_end = system->N_cm-2;
    }


    int batch_size = atom_end - atom_start; // start inclusive,end exclusive
    input_shape.values[0] = batch_size;
    input_shape.values[1] = win_size;
    input_shape.values[2] = 1;
    input_shape.dim = 3;
    //printf(" before 1diff\n");
    obs_flat_in = smalloc( sizeof(DATA_PRECISION) * batch_size * win_size, "Predict_Charges_TF_LSTM:obs_flat" );
    for ( i = atom_start; i < atom_end; ++i )
    {
        int arr_start = i - atom_start;
        for (int j = 0; j < win_size; ++j ) 
        {
            // it needs to be reversed bc of the way the model trained
            // note that workspace->s is shifted (reason of extra  + 1)
            //first order diff.
            //double first_error = workspace->s[win_size - j - 1 + 1][i] - workspace->t[win_size - j - 1 + 1][i];
            //fprintf( stdout, "[INFO] s %.12f, t %.12f", workspace->s[win_size - j - 1 + 1][i], workspace->t[win_size - j - 1 + 1][i]);
            //double second_error = workspace->s[win_size - j + 1][i] - workspace->t[win_size - j + 1][i];
            //double val1 = roundf(workspace->s[win_size - j - 1 + 1][i] * 100000) / 100000;
            //double val2 = roundf(workspace->s[win_size - j + 1][i] * 100000) / 100000;
            double val1 = workspace->s[win_size - j - 1 + 1][i];
            double val2 = workspace->s[win_size - j + 1][i];
            obs_flat_in[arr_start * win_size + j] = val1 - val2;  
            //obs_flat[arr_start * win_size + j] = workspace->s[win_size - j - 1 + 1][i];      
        }
    }
    //fprintf( stdout, "[INFO] obs_flat %.12f %.12f\n",  obs_flat[0 * win_size + 0],obs_flat[0 * win_size + 1]);

    // std normalizer
    
    //double mean_array[batch_size];
    //double std_array[batch_size];
    //standard_normalize(obs_flat_in, win_size, batch_size, mean_array, std_array);


    //fprintf( stdout, "[INFO] obs_flat after norm %.12f %.12f\n",  obs_flat[0 * win_size + 0],obs_flat[0 * win_size + 1]);
    //fprintf( stdout, "[INFO] mean std %.12f %.12f\n",  mean_array[0],std_array[0]);

    //printf(" before TF_NewTensor\n");
    input_tensor[0] = TF_NewTensor( TF_DATA_TYPE, input_shape.values, input_shape.dim,
            (void *)obs_flat_in, sizeof(DATA_PRECISION) * win_size * batch_size,
            &TF_Tensor_Deallocator, NULL );


    //fprintf( stdout, "after input_tensor\n" );
    output_tensor[0] = NULL;

    //printf(" before TF_SessionRun\n");
    TF_SessionRun( my_model->session, NULL,
        &inputs[0], input_tensor, 1,
        &outputs[0], output_tensor, 1,
        NULL, 0, NULL, my_model->status );


    if (output_tensor[0] == NULL){
        fprintf( stdout, "output_tensor is null\n" );
        fprintf( stderr, "[ERROR] unable to start a sesssion: %s\n", TF_Message(my_model->status) );
    }
    DATA_PRECISION * predictions = (DATA_PRECISION*) TF_TensorData( output_tensor[0] );
    //fprintf( stdout, "[INFO] predictions[0] %.12f\n",  predictions[0]);

    //standard_reverse(predictions, win_size, batch_size, mean_array, std_array);
    //fprintf( stdout, "[INFO] after reverse predictions[0] %.12f\n",  predictions[0]);
    //printf(" before atom_start loop\n");
    for (i = atom_start; i < atom_end; ++i )
    {
        /*
        //cumulative sum to reverse
        double oldest = workspace->s[win_size+1][i]; // +1 because of shifting
        double sum = 0;
        int j = 0;
        int arr_start = i - atom_start;
        for (j = 0; j < win_size; j++){
            sum = sum + obs_flat[arr_start * win_size + j];
        }
        workspace->s[0][i] = oldest + sum + predictions[arr_start];
        */
        int arr_start = i - atom_start;
        // add the guessed error to the last error(true - guess)
        workspace->s[0][i] = predictions[arr_start] + workspace->s[1][i];
    }

    //sfree( obs_flat_in, "Predict_Charges_TF_LSTM:obs_flat_in" );
    TF_DeleteTensor( input_tensor[0] );
    TF_DeleteTensor( output_tensor[0] );
    //TODO: fix memory leak later
    //TF_DeleteSession( s.session, status );
    //TF_DeleteGraph( s.graph );

}

static void Predict_Charges_TF_LSTM( const reax_system * const system,
        const control_params * const control,
        simulation_data * const data, static_storage * const workspace )
{

    //TODO: allocate the resources in the init module
    //static TF_model my_model; // using the global one FIX IT LATER
    int atom_batch = 512; //1500 atom at a time
    real time;
    int i = 0;


    /* load the frozen model from file in GraphDef format
     *
     * note: the input/output tensors names must be provided */
    //TODO: either require standarding model names in GraphDef file
    //      or add control file parameters to all these to be changed
    if (model_top.session == NULL) {

         char graph_file[1000];
        fprintf(stdout, "model load in\n");
        fprintf(stdout, "%s\n", control->cm_init_guess_gd_model );
        snprintf(graph_file, 1000, "%s%s",control->cm_init_guess_gd_model, ".pb" );
        fprintf(stdout, "%s\n", graph_file );
        model_top = model_load( graph_file,
                "input", "output/BiasAdd", "target", "init", "train","loss",
                "learning_rate",
                "save/control_dependency", "save/restore_all", "save/Const" ); 
        model_bot = model_load( graph_file,
                "input", "output/BiasAdd", "target", "init", "train","loss",
                "learning_rate",
                "save/control_dependency", "save/restore_all", "save/Const" ); 
        fprintf( stdout, "model loading\n" );
        restore_model_parameters(&model_top,control->cm_init_guess_gd_model);
        restore_model_parameters(&model_bot,control->cm_init_guess_gd_model);

         fprintf( stdout, "weights loading\n" );


    }


    if ( !model_top.session )
    {
        fprintf( stdout, "[ERROR] failed to load frozen model from GraphDef"
                " file for initial guess prediction using Tensorflow. Terminating...\n" );
        exit( INVALID_INPUT );
    }
    if ( !model_bot.session )
    {
        fprintf( stdout, "[ERROR] failed to load frozen model from GraphDef"
                " file for initial guess prediction using Tensorflow. Terminating...\n" );
        exit( INVALID_INPUT );
    }

    //fprintf( stdout, "before shift\n" );
    /* shift previous solutions down by one time step 
    #ifdef _OPENMP
        #pragma omp parallel for schedule(static) \
            default(none) private(i)
    #endif*/

    /*
    for ( i = 0; i < system->N_cm; ++i )
    {
        // shifting
        int j = 0;
        for (j = WINDOW_SIZE-1;j > 0 ; j--) {
            workspace->s[j][i] = workspace->s[j-1][i];
            workspace->t[j][i] = workspace->t[j-1][i];                   
        }
    }
    */
    
    

    /* run the session to calculate the predictions for the given data */
    time = Get_Time( );
    for (int i = (system->N_cm-2)/2; i < system->N_cm-2; i = i + atom_batch) 
    //for (int i = 0; i < system->N_cm-2; i = i + atom_batch) 
    {
        batch_prediction(&model_bot, i, i + atom_batch, system,
        control,data, workspace);
    }
    
    for (int i = 0; i < (system->N_cm-2)/2; i = i + atom_batch) {
        batch_prediction(&model_top, i, i + atom_batch, system,
        control,data, workspace);      
    }
    
    /*
    for(int i = system->N_cm-2; i < system->N_cm; i = i + 1) {
        workspace->s[0][i] = workspace->s[1][i];
    }
    */

    data->timing.cm_tensorflow_just_prediction = Get_Timing_Info( time );
    //fprintf( stdout, "Successfully run session\n" );
    //float * predictions_f = (float*) TF_TensorData( output_tensor[0] );
    //fprintf( stdout, "predictions before\n" );
    //fprintf( stdout, "predictions after\n" );
}
