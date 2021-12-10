def linear_regression(x_train,y_train,x_test,y_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import  mean_squared_error
    from math import sqrt
    model_LR = LinearRegression()
    model_LR.fit(x_train, y_train) #aquí está ocurriendo el entrenamiento
    # imprimimos los coeficientes
    #model_LR.coef_
    # imprimimos el intercepto
    #model_LR.intercept_
    # score
    train_score_r2_LR = model_LR.score(x_train,y_train)
    test_score_r2_LR = model_LR.score(x_test,y_test)
    # predicciones sobre el conjunto de entrenamiento
    y_pred_train = model_LR.predict(x_train)
    rmse_train = sqrt(mean_squared_error(y_train,y_pred_train))
    # predicciones sobre el conjunto de validacion/test
    y_pred_test = model_LR.predict(x_test)
    rmse_test = sqrt(mean_squared_error(y_test,y_pred_test))

    return train_score_r2_LR,test_score_r2_LR,rmse_train,rmse_test


def random_forest_regressor(x_train,y_train,x_test,y_test):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import  mean_squared_error
    from math import sqrt
    model_RFR = RandomForestRegressor()
    model_RFR.fit(x_train,y_train)
    # score
    train_score_r2_RFR = model_RFR.score(x_train,y_train)
    test_score_r2_RFR = model_RFR.score(x_test,y_test)
    # predictions
    # predicciones sobre el conjunto de entrenamiento
    y_pred_train = model_RFR.predict(x_train)
    rmse_train = sqrt(mean_squared_error(y_train,y_pred_train))
    # predicciones sobre el conjunto de validacion/test
    y_pred_test = model_RFR.predict(x_test)
    rmse_test = sqrt(mean_squared_error(y_test,y_pred_test))
    return train_score_r2_RFR,test_score_r2_RFR,rmse_train,rmse_test

def manual_model(x_train,y_train,x_test,y_test):
    import pandas as pd
    df_manual_model = pd.DataFrame()
    cols = str(list(x_train.columns))
    # Build Linear Rgression
    df_manual_model.loc[0,"Model"] = "Linear Regression"
    df_manual_model.loc[0,"Columns"] = cols
    train_score_r2_LR,test_score_r2_LR,rmse_train,rmse_test = linear_regression(x_train,y_train,x_test,y_test)
    df_manual_model.loc[0,"r2 Train"] = train_score_r2_LR
    df_manual_model.loc[0,"r2 Validate"] = test_score_r2_LR
    df_manual_model.loc[0,"RMSE Train"] = rmse_train
    df_manual_model.loc[0,"RMSE Validate"] = rmse_test
    # Build Random Forest Regressor
    df_manual_model.loc[1,"Model"] = "Random Forest Regressor"
    df_manual_model.loc[1,"Columns"] = cols
    train_score_r2_RFR,test_score_r2_RFR,rmse_train,rmse_test = random_forest_regressor(x_train,y_train,x_test,y_test)
    df_manual_model.loc[1,"r2 Train"] = train_score_r2_RFR
    df_manual_model.loc[1,"r2 Validate"] = test_score_r2_RFR
    df_manual_model.loc[1,"RMSE Train"] = rmse_train
    df_manual_model.loc[1,"RMSE Validate"] = rmse_test
    return df_manual_model

def final_model(x_train,y_train,x_test,y_test):
    import pandas as pd
    df_manual_model = pd.DataFrame()
    cols = str(list(x_train.columns))
    # Build Linear Rgression
    df_manual_model.loc[0,"Model"] = "Linear Regression"
    df_manual_model.loc[0,"Columns"] = cols
    train_score_r2_LR,test_score_r2_LR,rmse_train,rmse_test = linear_regression(x_train,y_train,x_test,y_test)
    df_manual_model.loc[0,"r2 Train"] = train_score_r2_LR
    df_manual_model.loc[0,"r2 Test"] = test_score_r2_LR
    df_manual_model.loc[0,"RMSE Train"] = rmse_train
    df_manual_model.loc[0,"RMSE Test"] = rmse_test

    return df_manual_model
