//====================================================================
// uniform_ kernel
// 04/16/2020 Jack Weber (jlw422@cornell.edu)
//====================================================================

#include <kernel_common.hpp>
#include <random>

extern "C" {

  __attribute__ ((noinline))  int tensorlib_uniform_(
          hb_tensor_t* _self,
          hb_tensor_t* _seed, 
          float* _from, 
          float* _to){
    
    // Unwrap common seed
    uint32_t seed = *(uint32_t*)((intptr_t)_seed->data);
    float from = *_from;
    float to = *_to;     
    auto self = HBTensor<float>(_self);
    
    // RNG
    std::default_random_engine generator;
    generator.seed(seed + __bsg_id);         
    std::uniform_real_distribution<float> distribution(from, to);
    
    // Start profiling
    bsg_cuda_print_stat_kernel_start();

    hb_parallel_for(self.numel(), [&](size_t i) {
        self(i) = distribution(generator);
    });
    
    //   End profiling
    bsg_cuda_print_stat_kernel_end();
    return 0;
                      
}

HB_EMUL_REG_KERNEL(tensorlib_uniform_, hb_tensor_t*, hb_tensor_t*,
                     float*, float*)
}                                