#include <gtest/gtest.h>

#include "puremd.h"


namespace
{
    class PuReMDTest : public ::testing::Test
    {
        protected:
            void *handle;

            PuReMDTest ( )
            {
            }

            virtual ~PuReMDTest ( )
            {
            }

            virtual void SetUp( )
            {
            }

            virtual void TearDown( )
            {
                if ( handle != NULL )
                {
                    cleanup( handle );
                }
            }
    };


    TEST_F(PuReMDTest, water_6540)
    {
        handle = setup( "../data/benchmarks/water/water_6540.pdb", 
                "../data/benchmarks/water/ffield.water",
                "../environ/param.gpu.water" );

        ASSERT_EQ( simulate( handle ), SPUREMD_SUCCESS );

        //TODO: check energy after evolving system, e.g., 100 steps
    }


    TEST_F(PuReMDTest, silica_6000)
    {
        handle = setup( "../data/benchmarks/silica/silica_6000.pdb", 
                "../data/benchmarks/silica/ffield-bio",
                "../environ/param.gpu.water" );

        ASSERT_EQ( simulate( handle ), SPUREMD_SUCCESS );

        //TODO: check energy after evolving system, e.g., 100 steps
    }
}


int main( int argc, char **argv )
{
    ::testing::InitGoogleTest( &argc, argv );
    return RUN_ALL_TESTS( );
}
