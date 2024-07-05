from funciones_mlflow import argumentos, load_dataset, data_treatment, mlflow_tracking

def main():
  print("Ejecutamos el main")
  args_values = argumentos()
  df = load_dataset()
  x_train, x_test, y_train, y_test = data_treatment(df)
  mlflow_tracking(args_values.nombre_job, x_train, x_test, y_train, y_test, args_values.c_list)

if __name__ == "__main__":
  main()
