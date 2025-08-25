def train_and_save_model():
    """Train the fraud detection model and save it."""
    try:
        # Build the dataset
        df = build_dataframe(CSV_PATH)
        
        # Separate features and labels
        X = df.drop(columns=["label"])
        y = df["label"]
        
        logger.info(f"Training with {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Features: {list(X.columns)}")
        
        # Create the model
        model = RandomForestClassifier(
            n_estimators=60, 
            random_state=42,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1
        )
        
        if len(X) > 4:
            # Ensure test set has at least 2 samples (one per class)
            test_size = max(2, int(0.2 * len(X)))
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                y_pred = model.predict(X_test)
                logger.info("Test Set Performance:")
                logger.info(f"\n{classification_report(y_test, y_pred, target_names=['legit', 'scam'])}")
                logger.info(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
            except ValueError as e:
                logger.warning(f"Could not stratify split ({e}). Training on full data instead.")
                model.fit(X, y)
        else:
            logger.warning("Small dataset, training on full data")
            model.fit(X, y)
        
        # Full dataset evaluation
        preds = model.predict(X)
        logger.info("Full Dataset Performance:")
        logger.info(f"\n{classification_report(y, preds, target_names=['legit', 'scam'])}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info(f"Feature Importance:\n{feature_importance}")
        
        # Save the model
        joblib.dump(model, MODEL_PATH)
        logger.info(f"Model saved to {MODEL_PATH}")
        
        # Verify model can be loaded
        loaded_model = joblib.load(MODEL_PATH)
        test_prediction = loaded_model.predict(X.iloc[:1])
        logger.info(f"Model verification successful. Test prediction: {test_prediction}")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise
