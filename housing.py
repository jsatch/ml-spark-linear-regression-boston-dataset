# 
# Housing problem
# 1. Cargar el dataset en un dataframe.
# 2. Dividir el dataset en training y test
# 3. Entrenar el modelo utilizando LinearRegression (data de training)
# 4. Calcular el error (r2)
# 5. Probar el modelo con data de validacion. Ver el error.

from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

'''
Metodo para leer los numeros de un string. Estos deben
de estar separados por uno o varios espacios.
'''
def read_words(cad):
    array = []
    word = ''
    iniciado = False
    for x in cad:
        if x != ' ':
            iniciado = True
            word = word + x
        elif iniciado:
            array.append(float(word))
            word = ''
            iniciado = False
    array.append(float(word))
    return array
            
'''
Inicializa el contexto de spark. Por el momento para trabajar
de manera local.
'''
def init():
    conf = SparkConf().setAppName("Housing").setMaster("local")
    return SparkContext(conf=conf)

'''
Carga el dataset de boston housing en un dataframe poniendo
las cabeceras correspondientes.
'''
def load_dataset(spark):
    rdd = spark.sparkContext.textFile(
        "data/housing.data").map(lambda line: read_words(line))
    cabeceras = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", 
                    "RAD", "TAX", "PTRATIO", "B1000", "LSTAT", "MEDV"]
    return spark.createDataFrame(rdd, cabeceras)

'''
Convierte el dataframe en uno que tenga una columna llamada @output_col
que contenga las columnas @features_cols. Esto dado que se necesita
tener una dataframe de esa forma (Vector[], Label) para la regresion.
'''
def convert_dataframe_regression(data, features_cols, output_col):
    assembler = VectorAssembler(inputCols=features_cols, outputCol=output_col)
    return assembler.transform(data)

'''
Obtenemos un modelo utilizando Regresion Lineal. Tomar en cuenta
que se esta utilizando un parametro de regularizacion de 0.01
'''
def train_linear_regression(data, features_col_name, label_col_name):
    lr = LinearRegression(maxIter=10, 
                            featuresCol=features_col_name, 
                            labelCol=label_col_name, 
                            regParam=0.001)
    return lr.fit(data)

'''
Metodo de reporte de parametros
'''
def report_parameters_lr(lr_model):
    print("Coeficientes: {}".format(str(lr_model.coefficients)))
    print("Intercept: {}".format(str(lr_model.intercept)))

'''
Metodo que realizar una evaluacion utilizando una metrica
definida. Los resultados los pinta en la terminal.
'''
def evaluate_model_regression(data, label_col, prediction_col='prediction', metric='rmse'):
    evaluator = RegressionEvaluator(labelCol=label_col, metricName=metric, predictionCol=prediction_col)
    print( "{}:{}".format(metric,evaluator.evaluate(data)))

def main():
    sc = init()
    spark = SparkSession(sc)
    data = load_dataset(spark)
    # data.show() Para mostrar dataset cargado
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=12345)
    print("Longitud Training : {}".format(train_data.count()))
    print("Longitud Trest : {}".format(test_data.count()))
    cabeceras_features = ["CRIM", "ZN", "INDUS", "CHAS", "RM", "AGE", "DIS", 
                    "RAD", "TAX", "PTRATIO", "B1000", "LSTAT", "MEDV"]
    cabecera_output = "NOX"
    data_to_train = convert_dataframe_regression(train_data, cabeceras_features, 'features')
    #data_to_train.show()
    print("Training model...")
    lr_model = train_linear_regression(data_to_train, 'features', cabecera_output)
    report_parameters_lr(lr_model)

    print("Testing model...")
    test_data_to_validate = convert_dataframe_regression(test_data, cabeceras_features, 'features')
    evaluate_model_regression(
        lr_model.transform(test_data_to_validate).select("features", "NOX", "prediction"),
        'NOX', 'prediction', 'rmse')
    


if __name__ == "__main__":
    main()
