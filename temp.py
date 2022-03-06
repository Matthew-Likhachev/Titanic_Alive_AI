import keras as k
import pandas as pd
import numpy  as np
import matplotlib .pyplot as plt


data_frame = pd.read_csv("titanic.csv")
input_names=["Age", "Sex", "Pclass"]
output_names = ["Survived"]

raw_input_data = data_frame[input_names]
raw_output_data = data_frame[output_names]

#макс возраст на кораьле (на самом деле 80 в датасете)
max_age = 100
#словарь, лямбда выражения, возвращается как списки
encoders = {"Age":lambda age: [age/max_age],                                          #напр 22 года в пределах единицы будет 0.22, можно и больший возраст макс взять
            "Sex": lambda gen: {"male":[0], "female" :[1] }.get(gen),                 #муж = 0, жен = 1
            "Pclass": lambda pclass: {1:[1,0,0], 2 :[0,1,0], 3:[0,0,1] }.get(pclass),
            "Survived": lambda s_value: [s_value]} #класс от 1-3 перекод в 100 010 001


#
# 2 функции переводят  данные в более простой вид   
#
#-----------------------------------------------------
def dataframe_to_dict(df):
    result = dict()
    for column in df.columns:
        values = data_frame[column].values
        result[column]=values
    return result

def make_supervised(df):
    raw_input_data=data_frame[input_names]
    raw_output_data=data_frame[output_names]
    return{"inputs": dataframe_to_dict(raw_input_data),
           "outputs": dataframe_to_dict(raw_output_data)}
#------------------------------------------------------------
    
def encode(data):
    vectors = []
    for data_name, data_values in data.items():
        encoded =list(map(encoders[data_name], data_values))
        vectors.append(encoded)
    formated = []
    for vector_raw in list(zip(*vectors)):
        vector = []
        for element in vector_raw:
            for e in element:
                vector.append(e)
        formated.append(vector)
    return formated
    

supervised = make_supervised(data_frame)
encoded_inputs =np.array(encode(supervised["inputs"]))
encoded_outputs =np.array(encode(supervised["outputs"]))

#print(encoded_inputs)
#print(encoded_outputs)

#----------------------------------------------------------------------------------
#сверху все данные подготовлены в необходимый формат
#Далее делим данные на 3 части - обучение, проверка, доп проверка
#----------------------------------------------------------------------------------
  
train_x = encoded_inputs[:600]
train_y = encoded_outputs[:600]

test_x = encoded_inputs[600:]
test_y = encoded_outputs[600:]

#далее модель

model= k.Sequential()
model.add(k.layers.Dense(units=5, activation = "relu"))#relu = value with minus = 0, other the same
model.add(k.layers.Dense(units = 1, activation = "sigmoid"))
model.compile(loss = "mse", optimizer = "sgd", metrics = ["accuracy"])
#добавления прежде обучченной модели
model.load_weights("save_AI.h5")
#для обучения с 0
#fit_results = model.fit(x= train_x, y = train_y, epochs = 100, validation_split = 0.2)

#
#matplotlib - графики
#
plt.title("Losses train/validation")
plt.plot(fit_results.history["loss"], label = "Train")
plt.plot(fit_results.history["val_loss"], label = "Validation")
plt.legend()
plt.show()


plt.title("Accuracies train/validation")
plt.plot(fit_results.history["accuracy"], label = "Train")
plt.plot(fit_results.history["val_accuracy"], label = "Validation")
plt.legend()
plt.show()

#
#
#
predicted_test = model.predict(test_x)
real_data = data_frame.iloc[600:] [input_names]
real_data ["PSurvived"] = predicted_test
print (real_data)




#
#saving
#

model.save_weights("save_AI.h5")
