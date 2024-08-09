import pickle

loadedmodel = pickle.load(open('Modify.pkl', 'rb'))


custom_input_DE = 0.064254
custom_input_FE = 0.064254
pred = loadedmodel.predict([[custom_input_DE, custom_input_FE]])
print(pred)