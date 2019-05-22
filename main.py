import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import Callback


class MyLogger(Callback):
    def __init__(self, n):
        self.n = n

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.n == 0:
            print("epoch:", epoch, logs)


def eda(df, is_train=True):
    df = df.copy()

    # 1 missing
    df["GarageCars"] = df.GarageCars.fillna(df.GarageCars.median())
    # 1 missing
    df["GarageArea"] = df.GarageArea.fillna(df.GarageArea.median())
    # 1 missing
    df["TotalBsmtSF"] = df.TotalBsmtSF.fillna(df.TotalBsmtSF.median())
    # 81 missing
    df["GarageYrBlt"] = df.GarageYrBlt.fillna(df.GarageYrBlt.median())
    # 259 missing
    df["LotFrontage"] = df.LotFrontage.fillna(df.LotFrontage.median())
    # 8 missing
    df["MasVnrArea"] = df.MasVnrArea.fillna(df.MasVnrArea.median())
    # 1 missing
    df["BsmtFinSF1"] = df.BsmtFinSF1.fillna(df.BsmtFinSF1.median())
    # 2 missing
    df["BsmtFullBath"] = df.BsmtFullBath.fillna(df.BsmtFullBath.median())
    # 1 missing
    df["BsmtUnfSF"] = df.BsmtUnfSF.fillna(df.BsmtUnfSF.median())
    # 1 missing
    df["BsmtFinSF2"] = df.BsmtFinSF2.fillna(df.BsmtFinSF2.median())
    # 2 missing
    df["BsmtHalfBath"] = df.BsmtHalfBath.fillna(df.BsmtHalfBath.median())

    sale_condition_df = pd.get_dummies(df.SaleCondition, "SaleCondition")
    df[sale_condition_df.columns] = sale_condition_df

    sale_type_df = pd.get_dummies(df.SaleType, "SaleType")
    df[sale_type_df.columns] = sale_type_df

    paved_drive_df = pd.get_dummies(df.PavedDrive, "PavedDrive")
    df[paved_drive_df.columns] = paved_drive_df

    # 81 missing
    df["GarageCond"] = df.GarageCond.fillna("MISSING")
    garage_cond_df = pd.get_dummies(df.GarageCond, "GarageCond")
    df[garage_cond_df.columns] = garage_cond_df

    # 81 missing
    df["GarageQual"] = df.GarageQual.fillna("MISSING")
    garage_qual_df = pd.get_dummies(df.GarageQual, "GarageQual")
    df[garage_qual_df.columns] = garage_qual_df

    # 81 missing
    df["GarageFinish"] = df.GarageFinish.fillna("MISSING")
    garage_finish_df = pd.get_dummies(df.GarageFinish, "GarageFinish")
    df[garage_finish_df.columns] = garage_finish_df

    # 81 missing
    df["GarageType"] = df.GarageType.fillna("MISSING")
    garage_type_df = pd.get_dummies(df.GarageType, "GarageType")
    df[garage_type_df.columns] = garage_type_df

    ms_zoning_df = pd.get_dummies(df.MSZoning, "MSZoning")
    df[ms_zoning_df.columns] = ms_zoning_df

    street_df = pd.get_dummies(df.Street, "Street")
    df[street_df.columns] = street_df

    lot_shape_df = pd.get_dummies(df.LotShape, "LotShape")
    df[lot_shape_df.columns] = lot_shape_df

    land_contour_df = pd.get_dummies(df.LandContour, "LandContour")
    df[land_contour_df.columns] = land_contour_df

    utilities_df = pd.get_dummies(df.Utilities, "Utilities")
    df[utilities_df.columns] = utilities_df

    lot_config_df = pd.get_dummies(df.LotConfig, "LotConfig")
    df[lot_config_df.columns] = lot_config_df

    land_slope_df = pd.get_dummies(df.LandSlope, "LandSlope")
    df[land_slope_df.columns] = land_slope_df

    neighborhood_df = pd.get_dummies(df.Neighborhood, "Neighborhood")
    df[neighborhood_df.columns] = neighborhood_df

    condition1_df = pd.get_dummies(df.Condition1, "Condition1")
    df[condition1_df.columns] = condition1_df

    condition2_df = pd.get_dummies(df.Condition2, "Condition2")
    df[condition2_df.columns] = condition2_df

    bldg_type_df = pd.get_dummies(df.BldgType, "BldgType")
    df[bldg_type_df.columns] = bldg_type_df

    house_style_df = pd.get_dummies(df.HouseStyle, "HouseStyle")
    df[house_style_df.columns] = house_style_df

    roof_style_df = pd.get_dummies(df.RoofStyle, "RoofStyle")
    df[roof_style_df.columns] = roof_style_df

    roof_mat1_df = pd.get_dummies(df.RoofMatl, "RoofMatl")
    df[roof_mat1_df.columns] = roof_mat1_df

    exterior1_df = pd.get_dummies(df.Exterior1st, "Exterior1st")
    df[exterior1_df.columns] = exterior1_df

    exterior2_df = pd.get_dummies(df.Exterior2nd, "Exterior2nd")
    df[exterior2_df.columns] = exterior2_df

    heating_df = pd.get_dummies(df.Heating, "Heating")
    df[heating_df.columns] = heating_df

    heating_qc_df = pd.get_dummies(df.HeatingQC, "HeatingQC")
    df[heating_qc_df.columns] = heating_qc_df

    central_air_df = pd.get_dummies(df.CentralAir, "CentralAir")
    df[central_air_df.columns] = central_air_df

    kitchen_qual_df = pd.get_dummies(df.KitchenQual, "KitchenQual")
    df[kitchen_qual_df.columns] = kitchen_qual_df

    functional_df = pd.get_dummies(df.Functional, "Functional")
    df[functional_df.columns] = functional_df

    exter_qual_df = pd.get_dummies(df.ExterQual, "ExterQual")
    df[exter_qual_df.columns] = exter_qual_df

    exter_cond_df = pd.get_dummies(df.ExterCond, "ExterCond")
    df[exter_cond_df.columns] = exter_cond_df

    foundation_df = pd.get_dummies(df.Foundation, "Foundation")
    df[foundation_df.columns] = foundation_df

    # 1 missing
    df["Electrical"] = df.Electrical.fillna("MISSING")
    electrical_df = pd.get_dummies(df.Electrical, "Electrical")
    df[electrical_df.columns] = electrical_df

    # 37 missing
    df["BsmtQual"] = df.BsmtQual.fillna("MISSING")
    bsmt_qual_df = pd.get_dummies(df.BsmtQual, "BsmtQual")
    df[bsmt_qual_df.columns] = bsmt_qual_df

    # 37 missing
    df["BsmtCond"] = df.BsmtCond.fillna("MISSING")
    bsmt_cond_df = pd.get_dummies(df.BsmtCond, "BsmtCond")
    df[bsmt_cond_df.columns] = bsmt_cond_df

    # 37 missing
    df["BsmtExposure"] = df.BsmtExposure.fillna("MISSING")
    bsmt_exposure_df = pd.get_dummies(df.BsmtExposure, "BsmtExposure")
    df[bsmt_exposure_df.columns] = bsmt_exposure_df

    # 37 missing
    df["BsmtFinType1"] = df.BsmtFinType1.fillna("MISSING")
    bsmt_fin_type1_df = pd.get_dummies(df.BsmtFinType1, "BsmtFinType1")
    df[bsmt_fin_type1_df.columns] = bsmt_fin_type1_df

    # 37 missing
    df["BsmtFinType2"] = df.BsmtFinType2.fillna("MISSING")
    bsmt_fin_type2_df = pd.get_dummies(df.BsmtFinType2, "BsmtFinType2")
    df[bsmt_fin_type2_df.columns] = bsmt_fin_type2_df

    # 8 missing
    df["MasVnrType"] = df.MasVnrType.fillna("MISSING")
    mas_vnr_type_df = pd.get_dummies(df.MasVnrType, "MasVnrType")
    df[mas_vnr_type_df.columns] = mas_vnr_type_df

    columns = (
        [
            "OverallQual",
            "GrLivArea",
            "GarageCars",
            "GarageArea",
            "TotalBsmtSF",
            "1stFlrSF",
            "FullBath",
            "TotRmsAbvGrd",
            "YearBuilt",
            "YearRemodAdd",
            "GarageYrBlt",
            "MasVnrArea",
            "Fireplaces",
            "BsmtFinSF1",
            "LotFrontage",
            "WoodDeckSF",
            "2ndFlrSF",
            "OpenPorchSF",
            "HalfBath",
            "LotArea",
            "BsmtFullBath",
            "BsmtUnfSF",
            "BedroomAbvGr",
            "ScreenPorch",
            "PoolArea",
            "MoSold",
            "3SsnPorch",
            "BsmtFinSF2",
            "BsmtHalfBath",
            "MiscVal",
            "Id",
            "LowQualFinSF",
            "YrSold",
            "OverallCond",
            "MSSubClass",
            "EnclosedPorch",
            "KitchenAbvGr",
        ]
        + sale_condition_df.columns.tolist()
        + sale_type_df.columns.tolist()
        + paved_drive_df.columns.tolist()
        + garage_cond_df.columns.tolist()
        + garage_qual_df.columns.tolist()
        + garage_finish_df.columns.tolist()
        + garage_type_df.columns.tolist()
        + ms_zoning_df.columns.tolist()
        + street_df.columns.tolist()
        + lot_shape_df.columns.tolist()
        + land_contour_df.columns.tolist()
        + utilities_df.columns.tolist()
        + lot_config_df.columns.tolist()
        + land_slope_df.columns.tolist()
        + neighborhood_df.columns.tolist()
        + condition1_df.columns.tolist()
        + condition2_df.columns.tolist()
        + bldg_type_df.columns.tolist()
        + house_style_df.columns.tolist()
        + roof_style_df.columns.tolist()
        + roof_mat1_df.columns.tolist()
        + exterior1_df.columns.tolist()
        + exterior2_df.columns.tolist()
        + heating_df.columns.tolist()
        + heating_qc_df.columns.tolist()
        + central_air_df.columns.tolist()
        + kitchen_qual_df.columns.tolist()
        + functional_df.columns.tolist()
        + exter_qual_df.columns.tolist()
        + exter_cond_df.columns.tolist()
        + foundation_df.columns.tolist()
        + electrical_df.columns.tolist()
        + bsmt_qual_df.columns.tolist()
        + bsmt_cond_df.columns.tolist()
        + bsmt_exposure_df.columns.tolist()
        + bsmt_fin_type1_df.columns.tolist()
        + bsmt_fin_type2_df.columns.tolist()
        + mas_vnr_type_df.columns.tolist()
    )

    y = df.SalePrice if is_train else None
    X = df[columns]
    if "GarageQual_Ex" in X:
        X = X.drop(
            columns=[
                "GarageQual_Ex",
                "Utilities_NoSeWa",
                "Condition2_RRAe",
                "Condition2_RRAn",
                "Condition2_RRNn",
                "HouseStyle_2.5Fin",
                "RoofMatl_ClyTile",
                "RoofMatl_Membran",
                "RoofMatl_Metal",
                "RoofMatl_Roll",
                "Exterior1st_ImStucc",
                "Exterior1st_Stone",
                "Exterior2nd_Other",
                "Heating_Floor",
                "Heating_OthW",
                "Electrical_MISSING",
                "Electrical_Mix",
            ]
        )

    return X, y


def process(X, y=None):
    # poly = PolynomialFeatures(degree=2).fit(X)
    # X_poly = poly.transform(X)
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    # pca = PCA(n_components=100).fit(X_scaled)
    # X_pca = pca.transform(X_scaled)
    if y is not None:
        y = np.log(y)
    return X, y
