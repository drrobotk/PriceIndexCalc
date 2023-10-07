import pandas as pd
from pyspark.sql.functions import col
import pytest
from PriceIndexCalc.pandas_modules.index_methods import multilateral_methods
from pandas.testing import assert_frame_equal
from numpy.testing import assert_almost_equal


@pytest.fixture
def large_input_df():
    return pd.read_csv('tests/test_data/large_input_df.csv')


def test_time_dummy(large_input_df):
    actual = multilateral_methods(large_input_df, method='tpd', groups=['group']).sort_values('index_value')['index_value'].to_list()
    expected = pd.read_csv('tests/test_data/large_output_tpd_pure.csv').sort_values(by='index_value')['index_value'].to_list()
    assert_almost_equal(actual, expected, decimal=9)

# def test_time_dummy_extension_movement(large_input_df):
#     actual = multilateral_methods(large_input_df, method='tpd', extension_method='movement', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_tpd_movement.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=9)

# def test_time_dummy_extension_wisp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='tpd', extension_method='wisp', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_tpd_wisp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=9)

# def test_time_dummy_extension_hasp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='tpd', extension_method='hasp', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_tpd_hasp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=9)

# def test_time_dummy_extension_mean_pub(large_input_df):
#     actual = multilateral_methods(large_input_df, method='tpd', extension_method='mean_pub', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_tpd_mean_pub.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=9)

    



def test_gk(large_input_df):
    actual = multilateral_methods(large_input_df, method='gk', groups=['group']).sort_values('index_value')['index_value'].to_list()
    expected = pd.read_csv('tests/test_data/large_output_gk_pure.csv').sort_values(by='index_value')['index_value'].to_list()
    assert_almost_equal(actual, expected, decimal=14)

# def test_gk_extension_movement(large_input_df):
#     actual = multilateral_methods(large_input_df, method='gk', extension_method='movement', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_gk_movement.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)   

# def test_gk_extension_wisp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='gk', extension_method='wisp', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_gk_wisp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)   

# def test_gk_extension_hasp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='gk', extension_method='hasp', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_gk_hasp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)   

# def test_gk_extension_mean_pub(large_input_df):
#     actual = multilateral_methods(large_input_df, method='gk', extension_method='mean_pub', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_gk_mean_pub.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)   



 
def test_geks_tornqvist(large_input_df):
    actual = multilateral_methods(large_input_df, method='geks', bilateral_method='tornqvist', groups=['group']).sort_values('index_value')['index_value'].to_list()
    expected = pd.read_csv('tests/test_data/large_output_geks_tornqvist_pure.csv').sort_values(by='index_value')['index_value'].to_list()
    assert_almost_equal(actual, expected, decimal=14)

# def test_geks_tornqvist_extension_movement(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='movement', bilateral_method='tornqvist', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_tornqvist_movement.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_tornqvist_extension_wisp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='wisp', bilateral_method='tornqvist', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_tornqvist_wisp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_tornqvist_extension_hasp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='hasp', bilateral_method='tornqvist', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_tornqvist_hasp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_tornqvist_extension_mean_pub(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='mean_pub', bilateral_method='tornqvist', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_tornqvist_mean_pub.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)




# def test_geks_jevons(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', bilateral_method='jevons', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_jevons_pure.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_jevons_extension_movement(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='movement', bilateral_method='jevons', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_jevons_movement.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_jevons_extension_wisp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='wisp', bilateral_method='jevons', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_jevons_wisp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_jevons_extension_hasp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='hasp', bilateral_method='jevons', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_jevons_hasp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_jevons_extension_mean_pub(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='mean_pub', bilateral_method='jevons', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_jevons_mean_pub.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)




# def test_geks_fisher(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', bilateral_method='fisher', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_fisher_pure.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_fisher_extension_movement(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='movement', bilateral_method='fisher', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_fisher_movement.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_fisher_extension_wisp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='wisp', bilateral_method='fisher', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_fisher_wisp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_fisher_extension_hasp(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='hasp', bilateral_method='fisher', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_fisher_hasp.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)

# def test_geks_fisher_extension_mean_pub(large_input_df):
#     actual = multilateral_methods(large_input_df, method='geks', extension_method='mean_pub', bilateral_method='fisher', groups=['group']).sort_values('index_value')['index_value'].to_list()
#     expected = pd.read_csv('tests/test_data/large_output_geks_fisher_mean_pub.csv').sort_values(by='index_value')['index_value'].to_list()
#     assert_almost_equal(actual, expected, decimal=14)