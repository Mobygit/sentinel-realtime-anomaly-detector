import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import argparse
import logging
from datetime import datetime
import json
from pathlib import Path

# Import from our sentinel module
from sentinel import LogParser, FeatureExtractor




class ModelTrainer:
    """Advanced model trainer with evaluation and versioning."""
    

    def __init__(self, model_dir='models'):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.parser = LogParser()
        self.featurizer = FeatureExtractor()
        

    def load_training_data(self, log_files, sample_ratio=1.0):
        """Load and parse training data from log files."""
        all_lines = []
        
        if isinstance(log_files, str):
            log_files = [log_files]
        
        for log_file in log_files:
            logging.info(f"📖 Reading {log_file}...")
            with open(log_file, 'r') as f:
                lines = f.readlines()
                
            # Sample if requested
            if sample_ratio < 1.0:
                sample_size = int(len(lines) * sample_ratio)
                lines = np.random.choice(lines, sample_size, replace=False)
                
            all_lines.extend(lines)
        
        logging.info(f"📊 Loaded {len(all_lines)} log lines")
        return all_lines
    

    def prepare_features(self, log_lines):
        """Parse log lines and prepare features."""
        feature_vectors = []
        valid_lines = 0
        
        for i, line in enumerate(log_lines):
            if i % 10000 == 0 and i > 0:
                logging.info(f" Processed {i}/{len(log_lines)} lines...")
            
            parsed = self.parser.parse_line(line)
            if parsed:
                features = self.featurizer.featurize(parsed)
                feature_vectors.append(features)
                valid_lines += 1
            else:
                logging.debug(f"⚠️ Failed to parse line: {line[:100]}...")
        
        logging.info(f"✅ Successfully parsed {valid_lines}/{len(log_lines)} lines")
        return pd.DataFrame(feature_vectors)
    

    def train_model(self, X, contamination=0.05, test_size=0.2):
        """Train and evaluate the anomaly detection model."""
        logging.info("🔧 Training model...")
        
        # Split data (for evaluation)
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            random_state=42,
            n_jobs=-1,  # Use all available cores
        )
        model.fit(X_train_scaled)
        
        # Evaluate on test set
        train_predictions = model.predict(X_train_scaled)
        test_predictions = model.predict(X_test_scaled)
        
        # Since we don't have true labels, we can only report distribution
        train_anomaly_ratio = (train_predictions == -1).mean()
        test_anomaly_ratio = (test_predictions == -1).mean()
        
        logging.info(f"📈 Training set anomaly ratio: {train_anomaly_ratio:.3f}")
        logging.info(f"📈 Test set anomaly ratio: {test_anomaly_ratio:.3f}")
        logging.info(f"📈 Expected contamination: {contamination:.3f}")
        
        return model, scaler, {
            'train_anomaly_ratio': train_anomaly_ratio,
            'test_anomaly_ratio': test_anomaly_ratio,
            'expected_contamination': contamination,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    

    def save_model(self, model, scaler, metadata, version=None):
        """Save model with versioning and metadata."""
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = self.model_dir / f"model_{version}.pkl"
        scaler_path = self.model_dir / f"scaler_{version}.pkl"
        metadata_path = self.model_dir / f"metadata_{version}.json"
        
        # Save model and scaler
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save metadata
        metadata['version'] = version
        metadata['created_at'] = datetime.now().isoformat()
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create symlinks for latest version
        latest_model = self.model_dir / "model_latest.pkl"
        latest_scaler = self.model_dir / "scaler_latest.pkl"
        
        if latest_model.exists():
            latest_model.unlink()
        if latest_scaler.exists():
            latest_scaler.unlink()
            
        # latest_model.symlink_to(model_path.name)
        # latest_scaler.symlink_to(scaler_path.name)

        import shutil
        shutil.copy2(model_path, latest_model)
        shutil.copy2(scaler_path, latest_scaler)
        
        
        logging.info(f"💾 Model saved as version: {version}")
        logging.info(f" Model: {model_path}")
        logging.info(f" Scaler: {scaler_path}")
        logging.info(f" Metadata: {metadata_path}")
        
        return version

    
    def train_from_scratch(self, log_files, contamination=0.05, sample_ratio=1.0):
        """Complete training pipeline."""
        # Load and prepare data
        log_lines = self.load_training_data(log_files, sample_ratio)
        X = self.prepare_features(log_lines)
        
        if len(X) == 0:
            logging.error("❌ No valid features extracted. Check your log format.")
            return None
        
        # Train model
        model, scaler, metadata = self.train_model(X, contamination)
        
        # Save model
        version = self.save_model(model, scaler, metadata)
        
        logging.info("✅ Training completed successfully!")
        return version


def main():
    parser = argparse.ArgumentParser(description='Advanced Model Trainer')
    parser.add_argument('--log-files', nargs='+', required=True, help='Log files for training')
    parser.add_argument('--model-dir', default='models', help='Output directory for models')
    parser.add_argument('--contamination', type=float, default=0.05, help='Expected proportion of anomalies')
    parser.add_argument('--sample-ratio', type=float, default=1.0, help='Fraction of data to use for training')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    trainer = ModelTrainer(model_dir=args.model_dir)
    
    try:
        version = trainer.train_from_scratch(
            log_files=args.log_files,
            contamination=args.contamination,
            sample_ratio=args.sample_ratio
        )
        
        if version:
            logging.info("🎉 Success! Use this model with:")
            logging.info(" --model models/model_latest.pkl")
            logging.info(" --scaler models/scaler_latest.pkl")
            
    except Exception as e:
        logging.error(f"❌ Training failed: {e}")
        raise



if __name__ == "__main__":
    main()