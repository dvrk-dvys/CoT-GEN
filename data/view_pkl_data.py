import pickle

laptops_test_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/laptops/Laptops_Test_Gold_Implicit_Labeled_preprocess_finetune.pkl'
laptops_train_gold_pkl_file = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/laptops/Laptops_Train_v2_Implicit_Labeled_preprocess_finetune.pkl'

preprocessed_laptops = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/preprocessed/laptops_base_google-flan-t5-base.pkl'
preprocessed_restauraunts = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/preprocessed/restaurants_base_google-flan-t5-base.pkl'
old_preprocessed_laptops = '/Users/jordanharris/Code/PycharmProjects/THOR-ISA-M1/data/preprocessed/old_laptops_base_google-flan-t5-base.pkl'



with open(laptops_test_gold_pkl_file, 'rb') as file:
    data = pickle.load(file)
    print(type(data))
    # print(data.keys())
    print()
