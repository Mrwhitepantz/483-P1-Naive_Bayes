import pandas as pd
import math


data = pd.read_csv('zoo.csv')
legs = pd.get_dummies(data.legs, prefix='has_legs', columns=['legs']).astype(int)
data[legs.columns] = legs

train = data.sample(frac=.7)
test = data.drop(train.index)

def calculate_pA(a):
    num_instances = len(train)
    class_probabilities = train['class_type'].value_counts()
    for probability, count in class_probabilities.items():
        class_probabilities[probability] = (count + a) / (num_instances + a * 7)
    return class_probabilities
  
def calculate_B_A(a):
    feature_probabilities_given_class = {}
    for class_val in range(1,8):
        instances_of_class = train[train["class_type"] == class_val].drop("animal_name",axis=1).drop("class_type",axis=1)
        length = len(instances_of_class)
        for col in instances_of_class:
            if col == "legs":
                continue

            count = instances_of_class[col].sum(numeric_only=True)
            key = (col, 1, class_val)
            not_key = (col, 0, class_val)
            feature_probabilities_given_class[key] = (count + a) / (length + a * 7)
            feature_probabilities_given_class[not_key] = (length - count + a) / (length + a * 7)
    
    return feature_probabilities_given_class
        
pA_values = calculate_pA(.01)

pB_A_values = calculate_B_A(.01)

final = pd.DataFrame()

for index, row in test.iterrows():
    classes_raw = []
    for class_val in range(1,8):
        feature_odds = []
        keys_row = row.drop(labels=["animal_name", "class_type"])
        for col_name,col_val in keys_row.items():
            if col_name == "legs":
                continue
            feature_odds.append(math.log2(pB_A_values[(col_name, col_val, class_val)]))
        classes_raw.append(pow(2, math.log2(pA_values[class_val]) + sum(feature_odds)))
    
    norm_total = sum(classes_raw)

    classes_norm = [prob / norm_total for prob in classes_raw]
    prediction = max( (prob, index+1) for index, prob in enumerate(classes_norm))

    prediction_data = pd.DataFrame(data=[[prediction[1], prediction[0], "CORRECT" if row["class_type"] == prediction[1] else "wrong"]], columns=['predicted','probability','correct?'])
    row_frame = row.to_frame().T
    output = pd.concat([row_frame, prediction_data.set_index(row_frame.index)], axis=1)
    final = pd.concat([final, output])

final = final.loc[:,~final.columns.str.startswith('has_legs')]

print(final.to_csv(index=False, lineterminator='\n'))
