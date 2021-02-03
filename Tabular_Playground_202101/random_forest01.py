from sklearn.ensemble import RandomForestRegressor
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

train_data = pd.read_csv("train.csv")
train_data.head()

test_data = pd.read_csv("test.csv")
test_data.head()


y = train_data["target"]

features = ["cont1", "cont2", "cont3", "cont4", "cont5", "cont6", "cont7",
            "cont8", "cont9", "cont10", "cont11", "cont12", "cont13", "cont14"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'id': test_data.id, 'target': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
