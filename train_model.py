"""Script para entrenar el modelo completo."""

from src.model.trainer import load_model, split_train_test, train_decision_tree
from src.preprocessing.pipeline import (
    fit_preprocessing_pipeline,
    load_data,
    save_preprocessing_pipeline,
    transform_data,
)
from src.utils.logging_config import logger


def main():
    """Función principal para entrenar el modelo."""
    logger.info("=== INICIANDO ENTRENAMIENTO DEL MODELO ===")
    
    # 1. Cargar datos
    logger.info("Paso 1: Cargando datos...")
    X, y = load_data()
    
    # 2. Crear y ajustar pipeline de preprocesamiento
    logger.info("Paso 2: Creando pipeline de preprocesamiento...")
    preprocessor = fit_preprocessing_pipeline(X)
    
    # 3. Transformar datos
    logger.info("Paso 3: Transformando datos...")
    X_transformed = transform_data(X, preprocessor)
    
    # 4. Guardar pipeline de preprocesamiento
    logger.info("Paso 4: Guardando pipeline de preprocesamiento...")
    save_preprocessing_pipeline(preprocessor)
    
    # 5. Dividir en train/test
    logger.info("Paso 5: Dividiendo datos en train/test...")
    X_train, X_test, y_train, y_test = split_train_test(X_transformed, y)
    
    # 6. Entrenar modelo
    logger.info("Paso 6: Entrenando modelo con GridSearchCV...")
    best_model, best_params = train_decision_tree(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        save_model=True
    )
    
    logger.info("=== ENTRENAMIENTO COMPLETADO ===")
    logger.info(f"Mejores parámetros: {best_params}")


if __name__ == "__main__":
    main()

