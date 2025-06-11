from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException, status
import logging
from datetime import datetime
import os
import tensorflow as tf
import torch
from transformers import AutoModel
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
import json

from models.models import Project, Model, Dataset, Audit, User, Report
from schemas.ml import ModelCreate, ModelResponse, ModelBase
from schemas.dataset import DatasetCreate, DatasetResponse
from schemas.audit import AuditCreate, AuditResponse
from schemas.report import ReportResponse
from utils.file_handler import save_file
from utils.error_handlers import ProjectNotFoundError, ValidationError
from core.config import settings
from services.database import get_supabase_client, get_project_by_id, get_model_by_id, get_dataset_by_id

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage
import hashlib
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import lime
import lime.lime_tabular
from scipy import stats
import psutil
#import onnx
import gc
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
import sklearn.naive_bayes
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification, make_regression
from tensorflow import keras
from keras import layers
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.graphics.charts.piecharts import Pie
from utils.gcs_handler import GCSHandler
import uuid
import tempfile

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self, db=None):
        self.db = db or get_supabase_client()
        self.supported_frameworks = {
            '.h5': 'tensorflow',
            '.pt': 'pytorch',
            '.pkl': 'sklearn',
            '.onnx': 'onnx'
        }
        self.model_categories = {
            'classification': ['tensorflow', 'pytorch', 'sklearn', 'onnx'],
            'regression': ['tensorflow', 'pytorch', 'sklearn', 'onnx'],
            'clustering': ['sklearn'],
            'nlp': ['tensorflow', 'pytorch']
        }
        
        # Initialize storage handler based on configuration
        if settings.USE_GCS:
            self.storage = GCSHandler(settings.GCS_BUCKET_NAME)
            self.models_dir = "models"
            self.outputs_dir = "outputs"
        else:
            # Create models directory if it doesn't exist
            self.models_dir = os.path.join(os.getcwd(), settings.UPLOAD_DIR)
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"Models directory created at: {self.models_dir}")
        
        # Initialize report styles
        styles = getSampleStyleSheet()
        self.title_style = styles['Title']
        self.heading_style = styles['Heading1']
        self.subheading_style = styles['Heading2']
        self.normal_style = styles['Normal']
        self.right_aligned_small_style = ParagraphStyle(
            'RightAlignedSmall',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_RIGHT
        )
        self.error_style = ParagraphStyle(
            'Error',
            parent=styles['Normal'],
            textColor=colors.red,
            fontSize=10
        )

    # Save metrics to JSON
    def save_metrics(self, metrics_dict, filepath):
        """Save metrics to a JSON file"""
        if settings.USE_GCS:
            # Convert filepath to GCS path parts
            path_parts = filepath.split('/')
            self.storage.save_json(metrics_dict, *path_parts)
        else:
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            
    async def _get_project(self, project_id: int, user) -> Any:
        """Get project and verify access"""
        # Handle both dict-style and attribute-style user objects
        user_id = getattr(user, 'id', None)
        if user_id is None and hasattr(user, '__getitem__'):
            try:
                user_id = user['id']
            except (KeyError, TypeError):
                user_id = None
                
        if user_id is None:
            raise ValidationError(
                message="Invalid user object, could not determine user ID",
                error_code="INVALID_USER"
            )
            
        logger.info(f"Getting project for user ID: {user_id}")
        
        try:
            # Use Supabase table query
            response = self.db.table('projects').select('*').eq('id', project_id).eq('user_id', user_id).execute()
            
            if not response.data or len(response.data) == 0:
                raise ProjectNotFoundError(f"Project {project_id} not found")
                
            project_data = response.data[0]
            # Convert dict to Project-like object
            class ProjectObject:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            
            return ProjectObject(project_data)
        except Exception as e:
            logger.error(f"Error getting project: {str(e)}")
            raise ProjectNotFoundError(f"Error accessing project: {str(e)}")

    async def _get_model(self, model_id: int, project_id: int) -> Any:
        """Get model and verify it belongs to project"""
        try:
            response = self.db.table('models').select('*').eq('id', model_id).eq('project_id', project_id).execute()
            
            if not response.data or len(response.data) == 0:
                raise ValidationError(
                    message=f"Model {model_id} not found in project {project_id}",
                    error_code="MODEL_NOT_FOUND"
                )
                
            model_data = response.data[0]
            # Convert dict to Model-like object
            class ModelObject:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)
            
            return ModelObject(model_data)
        except Exception as e:
            logger.error(f"Error getting model: {str(e)}")
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from fastapi import UploadFile, HTTPException, status
import logging
from datetime import datetime
import os
import tensorflow as tf
import torch
from transformers import AutoModel
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency
import json

from models.models import Project, Model, Dataset, Audit, User, Report
from schemas.ml import ModelCreate, ModelResponse, ModelBase
from schemas.dataset import DatasetCreate, DatasetResponse
from schemas.audit import AuditCreate, AuditResponse
from schemas.report import ReportResponse
from utils.file_handler import save_file
from utils.error_handlers import ProjectNotFoundError, ValidationError
from core.config import settings

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from PIL import Image as PILImage
import hashlib
import matplotlib.pyplot as plt
import io
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve
from sklearn.inspection import permutation_importance
from sklearn.inspection import partial_dependence
from sklearn.ensemble import IsolationForest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import shap
import lime
import lime.lime_tabular
from scipy import stats
import psutil
#import onnx
import gc
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
import sklearn.naive_bayes
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification, make_regression
from tensorflow import keras
from keras import layers
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.graphics.charts.piecharts import Pie
from utils.gcs_handler import GCSHandler
import uuid
import tempfile

logger = logging.getLogger(__name__)

class MLService:
    def __init__(self, db: Session):
        self.db = db
        self.supported_frameworks = {
            '.h5': 'tensorflow',
            '.pt': 'pytorch',
            '.pkl': 'sklearn',
            '.onnx': 'onnx'
        }
        self.model_categories = {
            'classification': ['tensorflow', 'pytorch', 'sklearn', 'onnx'],
            'regression': ['tensorflow', 'pytorch', 'sklearn', 'onnx'],
            'clustering': ['sklearn'],
            'nlp': ['tensorflow', 'pytorch']
        }
        
        # Initialize storage handler based on configuration
        if settings.USE_GCS:
            self.storage = GCSHandler(settings.GCS_BUCKET_NAME)
            self.models_dir = "models"
            self.outputs_dir = "outputs"
        else:
            # Create models directory if it doesn't exist
            self.models_dir = os.path.join(os.getcwd(), settings.UPLOAD_DIR)
            os.makedirs(self.models_dir, exist_ok=True)
            logger.info(f"Models directory created at: {self.models_dir}")
        
        # Initialize report styles
        styles = getSampleStyleSheet()
        self.title_style = styles['Title']
        self.heading_style = styles['Heading1']
        self.subheading_style = styles['Heading2']
        self.normal_style = styles['Normal']
        self.right_aligned_small_style = ParagraphStyle(
            'RightAlignedSmall',
            parent=styles['Normal'],
            fontSize=8,
            alignment=TA_RIGHT
        )
        self.error_style = ParagraphStyle(
            'Error',
            parent=styles['Normal'],
            textColor=colors.red,
            fontSize=10
        )

    # Save metrics to JSON
    def save_metrics(self, metrics_dict, filepath):
        """Save metrics to a JSON file"""
        if settings.USE_GCS:
            # Convert filepath to GCS path parts
            path_parts = filepath.split('/')
            self.storage.save_json(metrics_dict, *path_parts)
        else:
            with open(filepath, 'w') as f:
                json.dump(metrics_dict, f, indent=4)
            
    def _get_project(self, project_id: int, user) -> Project:
        """Get project and verify access"""
        # Handle both dict-style and attribute-style user objects
        user_id = getattr(user, 'id', None)
        if user_id is None and hasattr(user, '__getitem__'):
            try:
                user_id = user['id']
            except (KeyError, TypeError):
                user_id = None
                
        if user_id is None:
            raise ValidationError(
                message="Invalid user object, could not determine user ID",
                error_code="INVALID_USER"
            )
            
        logger.info(f"Getting project for user ID: {user_id}")
        
        try:
            # Use Supabase table query instead of SQLAlchemy query
            response = self.db.table('projects').select('*').eq('id', project_id).eq('user_id', user_id).execute()
            
            if not response.data or len(response.data) == 0:
                raise ProjectNotFoundError(f"Project {project_id} not found")
                
            project_data = response.data[0]
            # Convert dict to Project-like object
            project = type('Project', (), project_data)
            return project
        except Exception as e:
            logger.error(f"Error getting project: {str(e)}")
            raise ProjectNotFoundError(f"Error accessing project: {str(e)}")

    def _get_model(self, model_id: int, project_id: int) -> Model:
        """Get model and verify it belongs to project"""
        try:
            response = self.db.table('models').select('*').eq('id', model_id).eq('project_id', project_id).execute()
            
            if not response.data or len(response.data) == 0:
                raise ValidationError(
                    message=f"Model {model_id} not found in project {project_id}",
                    error_code="MODEL_NOT_FOUND"
                )
                
            model_data = response.data[0]
            # Convert dict to Model-like object
            model = type('Model', (), model_data)
            return model
        except Exception as e:
            logger.error(f"Error getting model: {str(e)}")
            raise ValidationError(
                message=f"Error accessing model: {str(e)}",
                error_code="MODEL_ACCESS_ERROR"
            )

    def _get_dataset(self, dataset_id: int, project_id: int) -> Dataset:
        """Get dataset and verify it belongs to project"""
        try:
            response = self.db.table('datasets').select('*').eq('id', dataset_id).eq('project_id', project_id).execute()
            
            if not response.data or len(response.data) == 0:
                raise ValidationError(
                    message=f"Dataset {dataset_id} not found in project {project_id}",
                    error_code="DATASET_NOT_FOUND"
                )
                
            dataset_data = response.data[0]
            # Convert dict to Dataset-like object
            dataset = type('Dataset', (), dataset_data)
            return dataset
        except Exception as e:
            logger.error(f"Error getting dataset: {str(e)}")
            raise ValidationError(
                message=f"Error accessing dataset: {str(e)}",
                error_code="DATASET_ACCESS_ERROR"
            )

    
    def _detect_model_framework(self, file_path: str) -> str:
        """Detect model framework from file extension"""
        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.supported_frameworks:
            raise ValidationError(
                message=f"Unsupported model file type: {ext}",
                error_code="UNSUPPORTED_MODEL_TYPE"
            )
        return self.supported_frameworks[ext]

    async def _load_model(self, model_path: str, model_type: str) -> Any:
        """Load a model from file"""
        try:
            # Debug logging
            logger.info(f"Loading model from path: {model_path}")
            
            # Detect model framework
            model_framework = self._detect_model_framework(model_path)
            
            # If using GCS and path is a GCS path
            if settings.USE_GCS and model_path.startswith("gs://"):
                # Extract the path from the GCS URL
                logger.info(f"Detected GCS path: {model_path}")
                
                # Parse the GCS URL properly
                parts = model_path.replace("gs://", "").split("/")
                bucket_name = parts[0]
                blob_path = '/'.join(parts[1:])  # This includes the full path after bucket
                
                logger.info(f"Parsed GCS path - Bucket: {bucket_name}, Blob path: {blob_path}")
                
                # Handle both old and new path formats
                # If the blob_path starts with "models/" directly (old format), 
                # try adding "uploads/" prefix (new format)
                if blob_path.startswith("models/") and not blob_path.startswith("uploads/"):
                    original_blob_path = blob_path
                    # First try the new path format
                    new_blob_path = f"uploads/{blob_path}"
                    logger.info(f"Trying new path format: {new_blob_path}")
                    
                    # Check if the blob exists with the new path
                    new_blob = self.storage.bucket.blob(new_blob_path)
                    if new_blob.exists():
                        logger.info(f"Found model at new path: {new_blob_path}")
                        blob_path = new_blob_path
                        blob = new_blob
                    else:
                        # Fall back to the original path
                        logger.info(f"Model not found at new path, trying original path: {original_blob_path}")
                        blob = self.storage.bucket.blob(original_blob_path)
                        if not blob.exists():
                            logger.error(f"Blob does not exist at either path: {original_blob_path} or {new_blob_path}")
                            raise ValueError(f"File not found in GCS at paths: {original_blob_path} or {new_blob_path}")
                else:
                    # Use the path as is
                    blob = self.storage.bucket.blob(blob_path)
                    # Check if blob exists
                    if not blob.exists():
                        logger.error(f"Blob does not exist: {blob_path}")
                        raise ValueError(f"File not found in GCS at path: {blob_path}")
                
                # Download file to a temporary location
                temp_dir = "/tmp"
                os.makedirs(temp_dir, exist_ok=True)
                local_path = os.path.join(temp_dir, os.path.basename(model_path))
                
                logger.info(f"Downloading blob from {blob_path} to {local_path}")
                blob.download_to_filename(local_path)
                
                # Use local path for loading
                model_path = local_path
                logger.info(f"Using local path for model loading: {model_path}")
            
            # Load model based on framework
            if model_framework == 'sklearn':
                return joblib.load(model_path)
            elif model_framework == 'tensorflow':
                return tf.keras.models.load_model(model_path)
            #elif model_framework == 'onnx':
                #return onnx.load(model_path)
            else:
                raise ValueError(f"Unsupported model framework: {model_framework}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise HTTPException(status_code=422, detail=f"Error loading model: {str(e)}")

    async def _load_data(
        self,
        dataset_path: str,
        data_type: str,
        model_type: str,
        target_column: Optional[str] = None,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Load and preprocess data based on type"""
        logger.info(f"Starting data loading process for: {dataset_path}")
        logger.info(f"Data type: {data_type}, Model type: {model_type}")
        
        # Handle GCS paths
        if settings.USE_GCS and dataset_path.startswith("gs://"):
            # Extract the path from the GCS URL
            logger.info(f"Detected GCS path: {dataset_path}")
            
            # Parse the GCS URL properly
            parts = dataset_path.replace("gs://", "").split("/")
            bucket_name = parts[0]
            blob_path = '/'.join(parts[1:])  # Preserves the full path
            
            logger.info(f"Parsed GCS path - Bucket: {bucket_name}, Blob path: {blob_path}")
            
            # Handle both old and new path formats
            # If the blob_path starts with "datasets/" directly (old format), 
            # try adding "uploads/" prefix (new format)
            if blob_path.startswith("datasets/") and not blob_path.startswith("uploads/"):
                original_blob_path = blob_path
                # First try the new path format
                new_blob_path = f"uploads/{blob_path}"
                logger.info(f"Trying new path format: {new_blob_path}")
                
                # Check if the blob exists with the new path
                new_blob = self.storage.bucket.blob(new_blob_path)
                if new_blob.exists():
                    logger.info(f"Found dataset at new path: {new_blob_path}")
                    blob_path = new_blob_path
                    blob = new_blob
                else:
                    # Fall back to the original path
                    logger.info(f"Dataset not found at new path, trying original path: {original_blob_path}")
                    blob = self.storage.bucket.blob(original_blob_path)
                    if not blob.exists():
                        logger.error(f"Blob does not exist at either path: {original_blob_path} or {new_blob_path}")
                        raise ValidationError(
                            message=f"Dataset not found in GCS at paths: {original_blob_path} or {new_blob_path}",
                            error_code="DATASET_FILE_NOT_FOUND"
                        )
            else:
                # Use the path as is
                blob = self.storage.bucket.blob(blob_path)
                # Check if blob exists
                if not blob.exists():
                    logger.error(f"Dataset blob does not exist: {blob_path}")
                    raise ValidationError(
                        message=f"Dataset not found in GCS at path: {blob_path}",
                        error_code="DATASET_FILE_NOT_FOUND"
                    )
            
            # Download file to a temporary location
            temp_dir = "/tmp"
            os.makedirs(temp_dir, exist_ok=True)
            local_path = os.path.join(temp_dir, os.path.basename(dataset_path))
            
            logger.info(f"Downloading dataset from {blob_path} to {local_path}")
            blob.download_to_filename(local_path)
            
            # Use local path for loading
            dataset_path = local_path
            logger.info(f"Using local path for dataset loading: {dataset_path}")
            
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset file not found at: {dataset_path}")
            raise ValidationError(
                message=f"Dataset not found at: {dataset_path}",
                error_code="DATASET_FILE_NOT_FOUND"
            )
        
        try:
            if data_type == "tabular":
                logger.info("Loading tabular data...")
                # Load tabular data
                data = pd.read_csv(dataset_path)
                logger.info(f"Loaded data with shape: {data.shape}")
                logger.info(f"Columns: {data.columns.tolist()}")
                
                # Assign default column names if no headers exist
                if data.columns[0] == 0:
                    logger.info("No headers found, assigning default column names")
                    data.columns = [f"feature_{i}" for i in range(data.shape[1])]
                
                # Determine target column
                if target_column is None:
                    logger.info("Determining target column...")
                    # Try common target column names
                    common_targets = [
                        'target', 'class', 'label', 'y', 'output', 'response',
                        'dependent', 'outcome', 'result', 'prediction',
                        'target_variable', 'dependent_variable'
                    ]
                    
                    # First try exact matches
                    target_column = next((col for col in common_targets if col in data.columns), None)
                    
                    # If no exact match, try case-insensitive matches
                    if target_column is None:
                        logger.info("No exact match found, trying case-insensitive matches")
                        target_column = next(
                            (col for col in data.columns 
                             if any(col.lower() == t.lower() for t in common_targets)),
                            None
                        )
                    
                    # Use statistical checks if no direct match
                    if target_column is None:
                        logger.info("No direct match found, using statistical checks")
                        for col in data.columns:
                            if data[col].nunique() < 10: # Assuming categorical target
                                target_column = col
                                break
                    
                    # If still no match, try to find the last column
                    if target_column is None:
                        target_column = data.columns[-1]
                        logger.info(f"Using last column as target: {target_column}")
                    
                    logger.info(f"Selected target column: {target_column}")
                
                if target_column not in data.columns:
                    logger.error(f"Target column '{target_column}' not found in dataset")
                    raise ValidationError(
                        message=f"Target column '{target_column}' not found in dataset",
                        error_code="INVALID_TARGET_COLUMN"
                    )
                
                # Determine feature columns
                if feature_columns is None:
                    feature_columns = [col for col in data.columns if col != target_column]
                    logger.info(f"Using all columns except {target_column} as features")
                else:
                    # Validate specified feature columns
                    missing_features = [col for col in feature_columns if col not in data.columns]
                    if missing_features:
                        logger.error(f"Missing feature columns: {missing_features}")
                        raise ValidationError(
                            message=f"Feature columns not found: {missing_features}",
                            error_code="INVALID_FEATURE_COLUMNS"
                        )
                    logger.info(f"Using specified feature columns: {feature_columns}")
                
                # Separate features and target
                logger.info("Separating features and target...")
                X = data[feature_columns]
                y = data[target_column]
                
                # Store feature names
                feature_names = feature_columns
                
                # Log data shape
                logger.info(f"Data shape: X={X.shape}, y={y.shape}")
                
                # Scale features
                logger.info("Scaling features...")
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                logger.info("Features scaled successfully")
                
                # Save scaler for future use
                scaler_dir = os.path.join(os.path.dirname(dataset_path), 'scalers')
                os.makedirs(scaler_dir, exist_ok=True)
                scaler_path = os.path.join(scaler_dir, f'scaler_{os.path.basename(dataset_path)}.joblib')
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved scaler to: {scaler_path}")
                
                # Convert target based on model type
                if model_type == "classification":
                    logger.info("Converting target for classification...")
                    if y.dtype in ['int64', 'float64']:
                        # For numerical targets, convert to binary
                        y = (y > 0.5).astype(int)
                        logger.info("Converted numerical target to binary")
                    else:
                        # For categorical targets, convert to numeric
                        y = pd.Categorical(y).codes
                        logger.info("Converted categorical target to numeric")
                elif model_type == "regression":
                    logger.info("Validating target for regression...")
                    # Ensure target is numeric for regression
                    if y.dtype not in ['int64', 'float64']:
                        logger.error("Target column must be numeric for regression")
                        raise ValidationError(
                            message="Target column must be numeric for regression",
                            error_code="INVALID_REGRESSION_TARGET"
                        )
                
                logger.info("Data loading completed successfully")
                return X, y, feature_names
                
            elif data_type == "text":
                logger.info("Loading text data...")
                # Load text data
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    texts = [line.strip() for line in f.readlines()]
                logger.info(f"Loaded {len(texts)} text samples")
                
                # Convert text to numerical features
                logger.info("Converting text to numerical features...")
                vectorizer = TfidfVectorizer(max_features=1000)
                X = vectorizer.fit_transform(texts).toarray()
                logger.info(f"Text converted to features with shape: {X.shape}")
                
                # Save vectorizer for future use
                vectorizer_dir = os.path.join(os.path.dirname(dataset_path), 'vectorizers')
                os.makedirs(vectorizer_dir, exist_ok=True)
                vectorizer_path = os.path.join(vectorizer_dir, f'vectorizer_{os.path.basename(dataset_path)}.joblib')
                joblib.dump(vectorizer, vectorizer_path)
                logger.info(f"Saved vectorizer to: {vectorizer_path}")
                
                # For text data, we'll use a simple binary classification
                y = np.zeros(len(texts))  # Placeholder target
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
                
                logger.info(f"Text data shape: X={X.shape}, y={y.shape}")
                return X, y, feature_names
                
            else:
                logger.error(f"Unsupported data type: {data_type}")
                raise ValidationError(
                    message=f"Unsupported data type: {data_type}",
                    error_code="UNSUPPORTED_DATA_TYPE"
                )
                
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}", exc_info=True)
            raise ValidationError(
                message=f"Error loading data: {str(e)}",
                error_code="DATA_LOAD_ERROR"
            )

    def _get_latest_version(self, project_id: int, model_name: str) -> str:
        """Get the latest version of a model"""
        try:
            # Use Supabase query to get models with the given name
            response = self.db.table('models').select('version').eq('project_id', project_id).eq('name', model_name).execute()
            
            if not response.data or len(response.data) == 0:
                return "1.0.0"  # No existing versions, start with 1.0.0
                
            # Extract versions from response
            versions = [model_data.get('version', '0.0.0') for model_data in response.data if model_data.get('version')]
            
            # Sort versions semantically
            versions.sort(key=lambda v: [int(x) for x in v.split('.')])
            
            if not versions:
                return "1.0.0"
                
            # Get latest version
            latest_version = versions[-1]
            
            # Increment the last part of the version
            version_parts = latest_version.split('.')
            version_parts[-1] = str(int(version_parts[-1]) + 1)
            
            return '.'.join(version_parts)
            
        except Exception as e:
            logger.error(f"Error getting latest version: {str(e)}")
            return "1.0.0"  # Default to 1.0.0 on error

    def _validate_version(self, version: str) -> bool:
        """Validate version string format (MAJOR.MINOR.PATCH)"""
        try:
            major, minor, patch = map(int, version.split('.'))
            return major >= 0 and minor >= 0 and patch >= 0
        except (ValueError, AttributeError):
            return False

    async def upload_model(
        self,
        name: str,
        model_type: str,
        version: str,
        file: UploadFile,
        project_id: int,
        description: Optional[str] = None,
        current_user = None
    ) -> ModelBase:
        """Upload a model file"""
        try:
            logger.info(f"Starting model upload process for project {project_id}")
            logger.info(f"File details: {file.filename if file else 'No file'}")
            logger.info(f"File type: {type(file)}")
            logger.info(f"Model details: name={name}, type={model_type}, version={version}")
            logger.info(f"Current user: {current_user}")
            
            # Extract user ID handling both dictionary and object access
            user_id = getattr(current_user, 'id', None)
            if user_id is None and hasattr(current_user, '__getitem__'):
                try:
                    user_id = current_user['id']
                except (KeyError, TypeError):
                    user_id = None
                    
            logger.info(f"Extracted user ID: {user_id}")
            
            if user_id is None:
                raise ValidationError(
                    message="Could not determine user ID",
                    error_code="INVALID_USER"
                )

            # Validate file object
            if not file:
                logger.error("No file provided")
                raise ValidationError(
                    message="No file provided",
                    error_code="NO_FILE"
                )

            # Check if file is a valid UploadFile object
            if not hasattr(file, 'file') or not hasattr(file, 'filename'):
                logger.error("File object is not properly initialized")
                raise ValidationError(
                    message="File object is not properly initialized",
                    error_code="UNINITIALIZED_FILE"
                )

            if not file.filename:
                logger.error("File has no name")
                raise ValidationError(
                    message="File has no name",
                    error_code="INVALID_FILE_NAME"
                )

            if not file.content_type:
                logger.error("File has no content type")
                raise ValidationError(
                    message="File has no content type",
                    error_code="INVALID_FILE_CONTENT_TYPE"
                )

            # Validate model category
            if model_type not in self.model_categories:
                raise ValidationError(
                    message=f"Unsupported model category: {model_type}. Supported categories: {', '.join(self.model_categories.keys())}",
                    error_code="UNSUPPORTED_MODEL_CATEGORY"
                )

            # Validate file type
            ext = os.path.splitext(file.filename)[1].lower()
            logger.info(f"File extension: {ext}")
            
            if ext not in self.supported_frameworks:
                logger.error(f"Unsupported file type: {ext}")
                raise ValidationError(
                    message=f"Unsupported file type: {ext}. Supported types: {', '.join(self.supported_frameworks.keys())}",
                    error_code="UNSUPPORTED_FILE_TYPE"
                )

            # Validate that the file type is compatible with the model category
            actual_model_type = self.supported_frameworks[ext]
            if actual_model_type not in self.model_categories[model_type]:
                raise ValidationError(
                    message=f"File type {ext} is not compatible with model category {model_type}",
                    error_code="INCOMPATIBLE_MODEL_TYPE"
                )

            # Use project_id only as the subdirectory name, not as a separate directory
            project_subdir = str(project_id)
            
            # Save file using appropriate storage method
            if settings.USE_GCS:
                file_path = await self.storage.save_file(
                    file,
                    "uploads",      # Root "uploads" folder
                    "models",       # Subfolder for all models
                    project_subdir, # Project-specific subfolder
                    file.filename   # Original filename
                )
                logger.info(f"File saved at GCS path: {file_path}")
            else:
                # For local storage, create nested directory under uploads
                physical_dir = os.path.join(settings.UPLOAD_DIR, "models", project_subdir)
                os.makedirs(physical_dir, exist_ok=True)
                file_path = await save_file(file, physical_dir)
                logger.info(f"File saved at local path: {file_path}")
            
            # Create model data dictionary
            model_data = {
                "name": name,
                "description": description or f"Uploaded model file: {file.filename}",
                "model_type": model_type,
                "model_framework": actual_model_type,
                "version": version,
                "file_path": str(file_path),
                "project_id": project_id,
                "user_id": user_id,
                "model_metadata": {
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "file_size": self._get_file_size(file_path),
                    "model_type": model_type,
                    "model_framework": actual_model_type,
                    "version": version
                },
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Log the prepared data for debugging
            logger.info(f"Prepared model data: {model_data}")
            
            try:
                # Use Supabase insert instead of SQLAlchemy add
                logger.info(f"Inserting model into Supabase: {model_data}")
                
                # Try insertion with model_metadata first
                try:
                    response = self.db.table('models').insert(model_data).execute()
                except Exception as e:
                    # If model_metadata fails, try without it
                    if "model_metadata" in str(e):
                        logger.warning("model_metadata field not found, removing from data")
                        model_data.pop("model_metadata", None)
                        response = self.db.table('models').insert(model_data).execute()
                    else:
                        # If it's another error, raise it
                        raise
                
                if not response.data:
                    raise Exception("No data returned from Supabase after model creation")
                
                model_response_data = response.data[0]
                logger.info(f"Model created with ID: {model_response_data['id']}")
                
                # Convert to ModelResponse
                return ModelResponse(
                    id=model_response_data.get('id'),
                    name=model_response_data.get('name'),
                    description=model_response_data.get('description'),
                    model_type=model_response_data.get('model_type'),
                    model_framework=model_response_data.get('model_framework'),
                    version=model_response_data.get('version'),
                    file_path=model_response_data.get('file_path'),
                    model_metadata=model_response_data.get('model_metadata', {}),
                    project_id=model_response_data.get('project_id'),
                    user_id=model_response_data.get('user_id'),
                    created_at=model_response_data.get('created_at'),
                    updated_at=model_response_data.get('updated_at')
                )
            except Exception as e:
                logger.error(f"Database error: {str(e)}")
                raise ValidationError(
                    message=f"Database error: {str(e)}",
                    error_code="DATABASE_ERROR"
                )
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to upload model: {str(e)}"
            )

    def list_models(self, project_id: int, user) -> List[ModelResponse]:
        """List all models in a project"""
        try:
            # Verify project access
            project = self._get_project(project_id, user)
            
            # Use Supabase query instead of SQLAlchemy
            response = self.db.table('models').select('*').eq('project_id', project_id).execute()
            
            # Convert models to response format
            model_responses = []
            for model_data in response.data:
                try:
                    # Create a ModelResponse object from the data
                    model_response = ModelResponse(
                        id=model_data.get('id'),
                        name=model_data.get('name'),
                        description=model_data.get('description'),
                        model_type=model_data.get('model_type'),
                        version=model_data.get('version'),
                        file_path=model_data.get('file_path'),
                        model_metadata=model_data.get('model_metadata', {}),
                        project_id=model_data.get('project_id'),
                        user_id=model_data.get('user_id'),
                        created_at=model_data.get('created_at'),
                        updated_at=model_data.get('updated_at')
                    )
                    model_responses.append(model_response)
                except Exception as e:
                    logger.error(f"Error converting model {model_data.get('id')}: {str(e)}")
                    # Continue with other models even if one fails
                    continue
            
            if not model_responses:
                logger.warning(f"No models found for project {project_id}")
            
            return model_responses
            
        except Exception as e:
            logger.error(f"Error listing models: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error listing models: {str(e)}"
            )

    def list_datasets(self, project_id: int, user: User) -> List[DatasetResponse]:
        """List all datasets for a project"""
        try:
            # Verify project access
            project = self._get_project(project_id, user)
            
            # Use Supabase query instead of SQLAlchemy
            response = self.db.table('datasets').select('*').eq('project_id', project_id).execute()
            
            # Convert datasets to response format
            dataset_responses = []
            for dataset_data in response.data:
                try:
                    # Create a DatasetResponse object from the data
                    dataset_response = DatasetResponse(
                        id=dataset_data.get('id'),
                        name=dataset_data.get('name'),
                        description=dataset_data.get('description'),
                        dataset_type=dataset_data.get('dataset_type'),
                        file_path=dataset_data.get('file_path'),
                        dataset_metadata=dataset_data.get('dataset_metadata'),
                        project_id=dataset_data.get('project_id'),
                        user_id=dataset_data.get('user_id'),
                        created_at=dataset_data.get('created_at'),
                        updated_at=dataset_data.get('updated_at')
                    )
                    dataset_responses.append(dataset_response)
                except Exception as e:
                    logger.error(f"Error converting dataset {dataset_data.get('id')}: {str(e)}")
                    # Continue with other datasets even if one fails
                    continue
            
            if not dataset_responses:
                logger.warning(f"No datasets found for project {project_id}")
            
            return dataset_responses
            
        except Exception as e:
            logger.error(f"Error listing datasets: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error listing datasets: {str(e)}"
            )

    def _detect_dataset_type(self, file_path: str) -> str:
        """Detect dataset type from file extension"""
        logger.info(f"Detecting dataset type from file path: {file_path}")
        
        # Handle GCS paths
        if settings.USE_GCS and file_path.startswith("gs://"):
            # Extract the filename from the GCS path
            filename = file_path.split("/")[-1]
            logger.info(f"Extracted filename from GCS path: {filename}")
            
            # Detect file extension
            ext = os.path.splitext(filename)[1].lower()
        else:
            ext = os.path.splitext(file_path)[1].lower()
            
        logger.info(f"File extension: {ext}")
        
        if ext == '.csv':
            return "tabular"
        elif ext in ['.txt', '.json', '.jsonl']:
            return "text"
        elif ext == '.parquet':
            return "tabular"
        elif ext in ['.jpg', '.jpeg', '.png']:
            return "image"
        else:
            raise ValidationError(
                message=f"Unsupported dataset file type: {ext}",
                error_code="UNSUPPORTED_DATASET_TYPE"
            )


    async def upload_dataset(
        self, 
        project_id: int, 
        file: UploadFile, 
        user,
        dataset_type: Optional[str] = None
    ) -> DatasetResponse:
        """Upload a dataset file"""
        try:
            # Extract user ID handling both dictionary and object access
            user_id = getattr(user, 'id', None)
            if user_id is None and hasattr(user, '__getitem__'):
                try:
                    user_id = user['id']
                except (KeyError, TypeError):
                    user_id = None
                    
            if user_id is None:
                raise ValidationError(
                    message="Could not determine user ID",
                    error_code="INVALID_USER"
                )
            
            # Check project ownership
            project = self._get_project(project_id, user)
            logger.info(f"Project access validated: {project.name}")
            
            # Validate file
            if not file or not file.filename:
                raise ValidationError(
                    message="No file provided or file has no name",
                    error_code="INVALID_FILE"
                )
                
            # Use project_id only as the subdirectory name, not as a separate directory
            project_subdir = str(project_id)
            
            # Save file to directory
            try:
                if settings.USE_GCS:
                    # Save directly to "uploads/datasets/{project_id}/" path
                    file_path = await self.storage.save_file(
                        file,
                        "uploads",      # Root "uploads" folder
                        "datasets",     # Subfolder for all datasets
                        project_subdir, # Project-specific subfolder
                        file.filename   # Original filename
                    )
                    logger.info(f"File saved at GCS path: {file_path}")
                else:
                    # For local storage, create nested directory under uploads
                    physical_dir = os.path.join(settings.UPLOAD_DIR, "datasets", project_subdir)
                    os.makedirs(physical_dir, exist_ok=True)
                    file_path = await save_file(file, physical_dir)
                    logger.info(f"File saved at local path: {file_path}")
                    
            except Exception as e:
                logger.error(f"Error saving file: {str(e)}")
                raise ValidationError(
                    message=f"Error saving file: {str(e)}",
                    error_code="FILE_SAVE_ERROR"
                )
                
            # Detect dataset type if not specified
            if not dataset_type:
                try:
                    dataset_type = self._detect_dataset_type(file.filename)
                    logger.info(f"Auto-detected dataset type: {dataset_type}")
                except Exception as e:
                    logger.error(f"Error detecting dataset type: {str(e)}")
                    dataset_type = "tabular"  # Default to tabular if detection fails
                    logger.info(f"Using default dataset type: {dataset_type}")
                
            # Create dataset data
            dataset_data = {
                "name": file.filename,
                "description": f"Uploaded dataset file: {file.filename}",
                "dataset_type": dataset_type,
                "file_path": str(file_path),
                "project_id": project_id,
                "user_id": user_id,
                # Store additional metadata
                "dataset_metadata": {
                    "original_filename": file.filename,
                    "content_type": file.content_type,
                    "uploaded_at": datetime.utcnow().isoformat(),
                    "file_size": self._get_file_size(file_path),
                    "dataset_type": dataset_type
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()  # Add updated_at field
            }
            
            # Create database record
            try:
                logger.info(f"Creating dataset record with data: {dataset_data}")
                response = self.db.table('datasets').insert(dataset_data).execute()
                
                if not response.data or len(response.data) == 0:
                    logger.error("Failed to insert dataset record")
                    raise ValidationError(
                        message="Failed to create dataset record in database",
                        error_code="DATABASE_INSERT_ERROR"
                    )
                    
                # Extract dataset ID from response
                dataset_id = response.data[0]['id']
                logger.info(f"Dataset record created with ID: {dataset_id}")
                
                # Return dataset response
                return DatasetResponse(
                    id=dataset_id,
                    name=file.filename,
                    description=f"Uploaded dataset file: {file.filename}",
                    dataset_type=dataset_type,
                    file_path=str(file_path),
                    project_id=project_id,
                    user_id=user_id,
                    created_at=datetime.utcnow().isoformat(),
                    updated_at=datetime.utcnow().isoformat(),  # Add updated_at field
                    metadata=dataset_data["dataset_metadata"]
                )
                
            except Exception as e:
                logger.error(f"Error creating dataset record: {str(e)}")
                raise ValidationError(
                    message=f"Error creating dataset record: {str(e)}",
                    error_code="DATABASE_ERROR"
                )
                
        except ValidationError as e:
            # Re-raise validation errors
            raise e
        except Exception as e:
            # Wrap other exceptions
            logger.error(f"Unexpected error in upload_dataset: {str(e)}")
            raise ValidationError(
                message=f"Error uploading dataset: {str(e)}",
                error_code="UPLOAD_ERROR"
            )


    async def generate_synthetic_data(
        self, 
        project_id: int, 
        model_type: str,
        n_samples: int,
        n_features: int,
        n_classes: int,
        imbalance_ratio: float,
        noise: float, 
        user: User,
    ) -> DatasetResponse:
        """Upload a dataset file"""
        logger.info(f"Starting dataset upload for project {project_id}")
        try:
            # Validate project access
            project = self._get_project(project_id, user)
            logger.info(f"Project access validated: {project.name}")
            
            # Create project-specific directory for the dataset
            if settings.USE_GCS:
                # Create consistent path structure with upload_dataset method
                # Will create a GCS path as "uploads/datasets/{project_id}/synthetic_data.csv"
                filepath = f"gs://{settings.GCS_BUCKET_NAME}/uploads/datasets/{project_id}/synthetic_data.csv"
                # Local temp file for generation before uploading to GCS
                temp_filepath = os.path.join("/tmp", "synthetic_data.csv")
            else:
                # Create the same directory structure as in upload_dataset method
                physical_dir = os.path.join(settings.UPLOAD_DIR, "datasets", project_id)
                os.makedirs(physical_dir, exist_ok=True)
                filepath = os.path.join(physical_dir, "synthetic_data.csv")
                temp_filepath = filepath  # For local storage, no temp file needed
            
            logger.info(f"Dataset will be saved at: {filepath}")
            
            use_gan = True
            use_smote = True
            random_state = 42
            np.random.seed(random_state)
    
            if model_type == 'classification':
                weights = [imbalance_ratio, 1 - imbalance_ratio] if n_classes == 2 else None
                X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                                        n_classes=n_classes, weights=weights, flip_y=noise, 
                                        random_state=random_state)
                
                if use_smote:
                    smote = SMOTE(random_state=random_state)
                    X, y = smote.fit_resample(X, y)
            
            elif model_type == 'regression':
                X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=noise * 100, 
                                    random_state=random_state)
            else:
                raise ValueError("Invalid model_type. Choose 'classification' or 'regression'.")
            
            if use_gan:
                def build_generator():
                    model = keras.Sequential([
                        layers.Dense(16, activation="relu", input_shape=(n_features,)),
                        layers.Dense(n_features, activation="linear")
                    ])
                    return model
                
                generator = build_generator()
                noise = np.random.normal(0, 1, (n_samples, n_features))
                X = generator.predict(noise)
                y = np.random.choice(np.unique(y), size=n_samples)  # Random labels for generated data
            
            df = pd.DataFrame(X, columns=[f'feature_{i+1}' for i in range(n_features)])
            df['target'] = y
            
            # Save to temp file first
            df.to_csv(temp_filepath, index=False)
            logger.info(f"Synthetic data generated and saved to temp file: {temp_filepath}")
            
            # If using GCS, upload the file to the correct GCS path
            if settings.USE_GCS:
                blob_path = f"uploads/datasets/{project_id}/synthetic_data.csv"
                blob = self.storage.bucket.blob(blob_path)
                blob.upload_from_filename(temp_filepath)
                logger.info(f"Uploaded synthetic data to GCS: {filepath}")
                
                # Clean up temp file
                os.remove(temp_filepath)
                logger.info(f"Cleaned up temp file: {temp_filepath}")
            
            # Create metadata dictionary
            metadata = {
                "content_type": "csv",
                "uploaded_at": datetime.utcnow().isoformat(),
                "file_size": os.path.getsize(temp_filepath) if not settings.USE_GCS else blob.size
            }
            
            # Create dataset data
            dataset_data = {
                "name": "synthetic_data.csv",
                "description": f"Generated and Uploaded synthetic dataset",
                "dataset_type": "tabular",
                "file_path": str(filepath),
                "project_id": project_id,
                "user_id": user.id,
                # Store additional metadata
                "dataset_metadata": metadata,
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()  # Add updated_at field
            }
            
            # Create database record
            try:
                logger.info(f"Creating synthetic dataset record with data: {dataset_data}")
                response = self.db.table('datasets').insert(dataset_data).execute()
                
                if not response.data or len(response.data) == 0:
                    logger.error("Failed to insert synthetic dataset record")
                    raise ValidationError(
                        message="Failed to create synthetic dataset record in database",
                        error_code="DATABASE_INSERT_ERROR"
                    )
                    
                # Extract dataset ID from response
                dataset_id = response.data[0]['id']
                logger.info(f"Synthetic dataset record created with ID: {dataset_id}")
                
                # Return dataset response
                return DatasetResponse(
                    id=dataset_id,
                    name="synthetic_data.csv",
                    description=f"Generated and Uploaded synthetic dataset",
                    dataset_type="tabular",
                    file_path=str(filepath),
                    project_id=project_id,
                    user_id=user.id,
                    created_at=datetime.utcnow().isoformat(),
                    updated_at=datetime.utcnow().isoformat(),
                    metadata=metadata
                )
                
            except Exception as e:
                logger.error(f"Error creating synthetic dataset record: {str(e)}")
                raise ValidationError(
                    message=f"Error creating synthetic dataset record: {str(e)}",
                    error_code="DATABASE_ERROR"
                )
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error generating and uploading synthetic dataset: {str(e)}")
            raise ValidationError(
                message=f"Error generating synthetic dataset: {str(e)}",
                error_code="SYNTHETIC_DATA_ERROR"
            )



    def list_datasets(self, project_id: int, user: User) -> List[DatasetResponse]:
        """List all datasets for a project"""
        try:
            # Verify project access
            project = self._get_project(project_id, user)
            
            # Use Supabase query instead of SQLAlchemy
            response = self.db.table('datasets').select('*').eq('project_id', project_id).execute()
            
            # Convert datasets to response format
            dataset_responses = []
            for dataset_data in response.data:
                try:
                    # Create a DatasetResponse object from the data
                    dataset_response = DatasetResponse(
                        id=dataset_data.get('id'),
                        name=dataset_data.get('name'),
                        description=dataset_data.get('description'),
                        dataset_type=dataset_data.get('dataset_type'),
                        file_path=dataset_data.get('file_path'),
                        dataset_metadata=dataset_data.get('dataset_metadata'),
                        project_id=dataset_data.get('project_id'),
                        user_id=dataset_data.get('user_id'),
                        created_at=dataset_data.get('created_at'),
                        updated_at=dataset_data.get('updated_at')
                    )
                    dataset_responses.append(dataset_response)
                except Exception as e:
                    logger.error(f"Error converting dataset {dataset_data.get('id')}: {str(e)}")
                    # Continue with other datasets even if one fails
                    continue
            
            if not dataset_responses:
                logger.warning(f"No datasets found for project {project_id}")
            
            return dataset_responses
            
        except Exception as e:
            logger.error(f"Error listing datasets: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error listing datasets: {str(e)}"
            )

    async def _create_audit(
        self,
        project_id: int,
        user: User,
        audit_type: str,
        model_id: Optional[int] = None,
        dataset_id: Optional[int] = None
    ) -> Audit:
        """Create an audit record"""
        try:
            project = self._get_project(project_id, user)
            
            # Verify model and dataset if provided
            if model_id:
                self._get_model(model_id, project_id)
            if dataset_id:
                self._get_dataset(dataset_id, project_id)
            
            # Create audit data
            user_id = getattr(user, 'id', None)
            if user_id is None and hasattr(user, '__getitem__'):
                try:
                    user_id = user['id']
                except (KeyError, TypeError):
                    user_id = None
                    
            audit_data = {
                "project_id": project_id,
                "user_id": user_id,
                "audit_type": audit_type,
                "status": "running",
                "model_id": model_id,
                "dataset_id": dataset_id,
                "created_at": datetime.utcnow().isoformat()
            }
            
            # Insert audit record using Supabase
            response = self.db.table('audits').insert(audit_data).execute()
            
            if not response.data or len(response.data) == 0:
                raise Exception("Failed to create audit record")
                
            audit_data = response.data[0]
            # Convert dict to Audit-like object
            audit = type('Audit', (), audit_data)
            return audit
            
        except Exception as e:
            logger.error(f"Error creating audit: {str(e)}")
            raise

    async def run_performance_audit(
        self, 
        project_id: int, 
        user: User,
        model_id: int,
        dataset_id: int
    ) -> AuditResponse:
        """Run performance audit on model"""
        logger.info(f"Starting performance audit for model_id={model_id}, dataset_id={dataset_id}")
        try:
            # Enable TensorFlow eager execution
            logger.info("Enabling TensorFlow eager execution")
            tf.config.run_functions_eagerly(True)
            
            audit = await self._create_audit(
                project_id, user, "performance", model_id, dataset_id
            )
            logger.info(f"Created audit record with ID: {audit.id}")
            
            # Get model and dataset first
            logger.info("Retrieving model and dataset information")
            model = self._get_model(model_id, project_id)
            dataset = self._get_dataset(dataset_id, project_id)
            logger.info(f"Model: {model.name} ({model.model_type}), Dataset: {dataset.name}")
            
            # Log paths for debugging
            logger.info(f"Model file path: {model.file_path}")
            logger.info(f"Dataset file path: {dataset.file_path}")
            
            # Create output directory structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create directory paths
            report_dirs = {}
            categories = ['performance', 'explainability', 'fairness_bias', 'drift_robustness']
            
            if settings.USE_GCS:
                # For GCS, we just need the logical path structure
                base_path = f"{self.outputs_dir}/{project_id}/{model.name}-{model.version}"
                
                for category in categories:
                    category_path = f"{base_path}/{category}"
                    plots_path = f"{category_path}/plots"
                    metrics_path = f"{category_path}/metrics"
                    
                    # Store paths without creating actual directories (GCS doesn't need them)
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            else:
                # For local storage, create physical directories
                base_path = os.path.join(settings.OUTPUT_DIR, str(project_id), f"{model.name}-{model.version}")
                os.makedirs(base_path, exist_ok=True)
                
                for category in categories:
                    category_path = os.path.join(base_path, category)
                    plots_path = os.path.join(category_path, 'plots')
                    metrics_path = os.path.join(category_path, 'metrics')
                    
                    # Create the directories
                    os.makedirs(plots_path, exist_ok=True)
                    os.makedirs(metrics_path, exist_ok=True)
                    
                    # Store the paths
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            
            
            # Load model and data with proper error handling
            logger.info(f"Attempting to load model from: {model.file_path}")
            loaded_model = await self._load_model(model.file_path, model.model_type)
            logger.info(f"Model loaded successfully: {model.name}")
            
            # Make sure we're looking in the right place for datasets
            logger.info(f"Attempting to load dataset from: {dataset.file_path}")
            X, y, feature_names = await self._load_data(
                dataset.file_path, 
                dataset.dataset_type,
                model.model_type
            )
            logger.info(f"Data loaded with shape: {X.shape}, target shape: {y.shape}")
            
            # Initialize metrics dictionary
            logger.info("Initializing metrics dictionary")
            metrics = {
                'model_info': {
                    'type': model.model_type,
                    'name': model.name,
                    'version': model.version
                },
                'data_info': {
                    'total_samples': len(X),
                    'feature_count': X.shape[1],
                    'feature_names': feature_names
                }
            }
            
            # Handle class imbalance for classification
            if model.model_type == 'classification':
                logger.info("Computing class weights for classification")
                class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
                metrics['data_info']['class_distribution'] = {
                    str(cls): int(np.sum(y == cls)) for cls in np.unique(y)
                }
                metrics['data_info']['class_weights'] = {
                    str(cls): float(weight) for cls, weight in zip(np.unique(y), class_weights)
                }
                logger.info(f"Class distribution: {metrics['data_info']['class_distribution']}")
            
            # Split data for analysis
            logger.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Check if model is a Keras model
            is_keras_model = isinstance(loaded_model, (tf.keras.Model, tf.keras.Sequential))
            logger.info(f"Model type: {'Keras' if is_keras_model else 'scikit-learn'}")
            
            if is_keras_model:
                logger.info("Processing Keras model...")
                # Create new optimizer instance
                optimizer = tf.keras.optimizers.Adam()
                
                # Compile model with new optimizer
                logger.info("Compiling Keras model...")
                loaded_model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy' if model.model_type == 'classification' else 'mse',
                    metrics=['accuracy'] if model.model_type == 'classification' else ['mae']
                )
                logger.info("Model compiled successfully")
                
                # Train model without sample weights
                logger.info("Training Keras model...")
                history = loaded_model.fit(
                    X_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0,
                    class_weight=None  # Explicitly set class_weight to None
                )
                logger.info("Model training completed")
                
                # Get predictions
                logger.info("Generating predictions...")
                y_pred = loaded_model.predict(X_test)
                
                if model.model_type == 'classification':
                    logger.info("Processing classification metrics...")
                    # Convert predictions to binary for classification
                    y_pred = (y_pred > 0.5).astype(int)
                    
                    # Calculate metrics using scikit-learn without sample weights
                    metrics['basic_metrics'] = {
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "precision": float(precision_score(y_test, y_pred, average='weighted')),
                        "recall": float(recall_score(y_test, y_pred, average='weighted')),
                        "f1": float(f1_score(y_test, y_pred, average='weighted'))
                    }
                    logger.info(f"Basic metrics calculated: {metrics['basic_metrics']}")
                    
                    # ROC curve and AUC
                    logger.info("Calculating ROC curve and AUC...")
                    try:
                        # Try predict_proba first
                        if hasattr(loaded_model, 'predict_proba'):
                            y_pred_proba = loaded_model.predict_proba(X_test)
                            if y_pred_proba.shape[1] == 1:
                                y_pred_proba = y_pred_proba.ravel()
                            else:
                                y_pred_proba = y_pred_proba[:, 1]
                        else:
                            # Fallback to predict if predict_proba is not available
                            y_pred_proba = loaded_model.predict(X_test)
                            # Normalize predictions to [0,1] range if needed
                            if y_pred_proba.min() < 0 or y_pred_proba.max() > 1:
                                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                        
                        # Handle multiclass case
                        if len(np.unique(y_test)) > 2:
                            metrics['roc_curve'] = {}
                            for i in range(len(np.unique(y_test))):
                                # Convert to binary classification for each class
                                y_test_binary = (y_test == i).astype(int)
                                fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                                metrics['roc_curve'][f'class_{i}'] = {
                                    'fpr': fpr.tolist(),
                                    'tpr': tpr.tolist(),
                                    'auc': float(auc(fpr, tpr))
                                }
                            logger.info(f"Multiclass ROC curves calculated successfully")
                        else:
                            # Binary classification case
                            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                            metrics['roc_curve'] = {
                                'fpr': fpr.tolist(),
                                'tpr': tpr.tolist(),
                                'auc': float(auc(fpr, tpr))
                            }
                            logger.info(f"AUC score: {metrics['roc_curve']['auc']:.4f}")
                    except Exception as e:
                        logger.warning(f"Could not calculate ROC curve: {str(e)}")
                        metrics['roc_curve'] = {
                            'error': str(e),
                            'status': 'failed'
                        }
                    
                    # Confusion matrix
                    logger.info("Calculating confusion matrix...")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Handle single-class predictions
                    if cm.shape == (1, 1):
                        # Only one class was predicted
                        metrics['confusion_matrix'] = {
                            'matrix': cm.tolist(),
                            'true_negatives': int(cm[0, 0]),
                            'false_positives': 0,
                            'false_negatives': 0,
                            'true_positives': 0,
                            'warning': 'Model predicted only one class'
                        }
                    else:
                        # Normal case with multiple classes
                        metrics['confusion_matrix'] = {
                            'matrix': cm.tolist(),
                            'true_negatives': int(cm[0, 0]),
                            'false_positives': int(cm[0, 1]),
                            'false_negatives': int(cm[1, 0]),
                            'true_positives': int(cm[1, 1])
                        }
                    
                    logger.info(f"Confusion matrix calculated: {metrics['confusion_matrix']}")
                    
                    # Learning curve from history
                    logger.info("Processing learning curve data...")
                    metrics['learning_curve'] = {
                        'epochs': list(range(1, len(history.history['loss']) + 1)),
                        'train_loss': history.history['loss'],
                        'val_loss': history.history['val_loss'],
                        'train_accuracy': history.history['accuracy'],
                        'val_accuracy': history.history['val_accuracy']
                    }
                else:  # Regression
                    logger.info("Processing regression metrics...")
                    # Calculate metrics using scikit-learn without sample weights
                    metrics['basic_metrics'] = {
                        "mse": float(mean_squared_error(y_test, y_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                        "mae": float(mean_absolute_error(y_test, y_pred)),
                        "r2": float(r2_score(y_test, y_pred))
                    }
                    logger.info(f"Basic metrics calculated: {metrics['basic_metrics']}")
                    
                    # Residual analysis
                    logger.info("Performing residual analysis...")
                    residuals = y_test - y_pred.flatten()
                    metrics['residual_analysis'] = {
                        'mean_residual': float(np.mean(residuals)),
                        'std_residual': float(np.std(residuals)),
                        'residuals': residuals.tolist()
                    }
                    
                    # Learning curve from history
                    logger.info("Processing learning curve data...")
                    metrics['learning_curve'] = {
                        'epochs': list(range(1, len(history.history['loss']) + 1)),
                        'train_loss': history.history['loss'],
                        'val_loss': history.history['val_loss']
                    }
            else:
                logger.info("Processing scikit-learn model...")
                # Handle scikit-learn model evaluation
                cv = KFold(n_splits=5, shuffle=True, random_state=42)
                
                if model.model_type == 'classification':
                    logger.info("Performing cross-validation for classification...")
                    cv_scores = cross_val_score(loaded_model, X, y, cv=cv, scoring='accuracy')
                    metrics['cross_validation'] = {
                        'mean_score': float(np.mean(cv_scores)),
                        'std_score': float(np.std(cv_scores)),
                        'scores': cv_scores.tolist()
                    }
                    logger.info(f"Cross-validation scores: {cv_scores}")
                else:
                    logger.info("Performing cross-validation for regression...")
                    cv_scores = cross_val_score(loaded_model, X, y, cv=cv, scoring='r2')
                    metrics['cross_validation'] = {
                        'mean_score': float(np.mean(cv_scores)),
                        'std_score': float(np.std(cv_scores)),
                        'scores': cv_scores.tolist()
                    }
                    logger.info(f"Cross-validation scores: {cv_scores}")
                
                # Train model and get predictions
                logger.info("Training model and generating predictions...")
                loaded_model.fit(X_train, y_train)
                y_pred = loaded_model.predict(X_test)
                
                if model.model_type == 'classification':
                    logger.info("Calculating classification metrics...")
                    # Basic classification metrics
                    metrics['basic_metrics'] = {
                        "accuracy": float(accuracy_score(y_test, y_pred)),
                        "precision": float(precision_score(y_test, y_pred, average='weighted')),
                        "recall": float(recall_score(y_test, y_pred, average='weighted')),
                        "f1": float(f1_score(y_test, y_pred, average='weighted'))
                    }
                    logger.info(f"Basic metrics calculated: {metrics['basic_metrics']}")
                    
                    # ROC curve and AUC
                    if hasattr(loaded_model, 'predict_proba'):
                        logger.info("Calculating ROC curve and AUC...")
                        try:
                            # Try predict_proba first
                            if hasattr(loaded_model, 'predict_proba'):
                                y_pred_proba = loaded_model.predict_proba(X_test)
                                if y_pred_proba.shape[1] == 1:
                                    y_pred_proba = y_pred_proba.ravel()
                                else:
                                    y_pred_proba = y_pred_proba[:, 1]
                            else:
                                # Fallback to predict if predict_proba is not available
                                y_pred_proba = loaded_model.predict(X_test)
                                # Normalize predictions to [0,1] range if needed
                                if y_pred_proba.min() < 0 or y_pred_proba.max() > 1:
                                    y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())
                            
                            # Handle multiclass case
                            if len(np.unique(y_test)) > 2:
                                metrics['roc_curve'] = {}
                                for i in range(len(np.unique(y_test))):
                                    # Convert to binary classification for each class
                                    y_test_binary = (y_test == i).astype(int)
                                    fpr, tpr, _ = roc_curve(y_test_binary, y_pred_proba)
                                    metrics['roc_curve'][f'class_{i}'] = {
                                        'fpr': fpr.tolist(),
                                        'tpr': tpr.tolist(),
                                        'auc': float(auc(fpr, tpr))
                                    }
                                logger.info(f"Multiclass ROC curves calculated successfully")
                            else:
                                # Binary classification case
                                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                                metrics['roc_curve'] = {
                                    'fpr': fpr.tolist(),
                                    'tpr': tpr.tolist(),
                                    'auc': float(auc(fpr, tpr))
                                }
                                logger.info(f"AUC score: {metrics['roc_curve']['auc']:.4f}")
                        except Exception as e:
                            logger.warning(f"Could not calculate ROC curve: {str(e)}")
                            metrics['roc_curve'] = {
                                'error': str(e),
                                'status': 'failed'
                            }
                    
                    # Confusion matrix
                    logger.info("Calculating confusion matrix...")
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Handle single-class predictions
                    if cm.shape == (1, 1):
                        # Only one class was predicted
                        metrics['confusion_matrix'] = {
                            'matrix': cm.tolist(),
                            'true_negatives': int(cm[0, 0]),
                            'false_positives': 0,
                            'false_negatives': 0,
                            'true_positives': 0,
                            'warning': 'Model predicted only one class'
                        }
                    else:
                        # Normal case with multiple classes
                        metrics['confusion_matrix'] = {
                            'matrix': cm.tolist(),
                            'true_negatives': int(cm[0, 0]),
                            'false_positives': int(cm[0, 1]),
                            'false_negatives': int(cm[1, 0]),
                            'true_positives': int(cm[1, 1])
                        }
                    
                    logger.info(f"Confusion matrix calculated: {metrics['confusion_matrix']}")
                    
                    # Learning curve
                    logger.info("Calculating learning curve...")
                    train_sizes, train_scores, test_scores = learning_curve(
                        loaded_model, X, y, cv=5, n_jobs=-1, 
                        train_sizes=np.linspace(0.1, 1.0, 10)
                    )
                    metrics['learning_curve'] = {
                        'train_sizes': train_sizes.tolist(),
                        'train_scores': train_scores.tolist(),
                        'test_scores': test_scores.tolist()
                    }
                else:  # Regression
                    logger.info("Calculating regression metrics...")
                    # Basic regression metrics
                    metrics['basic_metrics'] = {
                        "mse": float(mean_squared_error(y_test, y_pred)),
                        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
                        "mae": float(mean_absolute_error(y_test, y_pred)),
                        "r2": float(r2_score(y_test, y_pred))
                    }
                    logger.info(f"Basic metrics calculated: {metrics['basic_metrics']}")
                    
                    # Residual analysis
                    logger.info("Performing residual analysis...")
                    residuals = y_test - y_pred
                    metrics['residual_analysis'] = {
                        'mean_residual': float(np.mean(residuals)),
                        'std_residual': float(np.std(residuals)),
                        'residuals': residuals.tolist()
                    }
                    
                    # Learning curve
                    logger.info("Calculating learning curve...")
                    train_sizes, train_scores, test_scores = learning_curve(
                        loaded_model, X, y, cv=5, n_jobs=-1, 
                        train_sizes=np.linspace(0.1, 1.0, 10)
                    )
                    metrics['learning_curve'] = {
                        'train_sizes': train_sizes.tolist(),
                        'train_scores': train_scores.tolist(),
                        'test_scores': test_scores.tolist()
                    }
            
            # Feature importance (if available)
            if hasattr(loaded_model, 'feature_importances_'):
                logger.info("Calculating feature importance...")
                metrics['feature_importance'] = {
                    'importances': loaded_model.feature_importances_.tolist(),
                    'feature_names': feature_names,
                    'method': 'tree_based'
                }
            
            # Update audit with results
            logger.info("Updating audit record with results...")
            
            # Use Supabase update instead of direct attribute setting and commit
            audit_update = {
                "results": metrics,
                "status": "completed",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.db.table('audits').update(audit_update).eq('id', audit.id).execute()
            
            if not response.data or len(response.data) == 0:
                raise Exception("Failed to update audit record")
                
            updated_audit = response.data[0]
            
            logger.info("Performance audit completed successfully")
            # Save performance metrics to JSON file
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create proper file paths based on storage type
            if settings.USE_GCS:
                # Use the metrics path directly from the dictionary
                perf_metrics_path = f"{report_dirs['performance']['metrics']}/performance-metrics-{timestamp}.json"
                logger.info(f"Saving performance metrics to GCS path: {perf_metrics_path}")
            else:
                # For local storage, use os.path.join
                perf_metrics_path = os.path.join(report_dirs['performance']['metrics'], f'performance-metrics-{timestamp}.json')
                logger.info(f"Saving performance metrics to local path: {perf_metrics_path}")
            
            # Save metrics
            self.save_metrics(metrics, perf_metrics_path)
            logger.info(f"Performance metrics saved to: {perf_metrics_path}")
            
            # Create audit response
            audit_response = AuditResponse(
                id=audit.id,
                project_id=project_id,
                model_id=model_id,
                dataset_id=dataset_id,
                user_id=getattr(user, 'id', None) if user else None,
                audit_type="performance",
                status="completed",
                results=metrics,
                created_at=audit.created_at,
                updated_at=datetime.utcnow().isoformat()
            )
            
            # Update audit status in the database
            await self._update_audit_status(audit.id, "completed", perf_metrics_path)
            
            return audit_response
            
        except Exception as e:
            if 'audit' in locals():
                logger.error(f"Error in performance audit: {str(e)}", exc_info=True)
                
                # Update audit status to failed using Supabase
                error_update = {
                    "status": "failed",
                  
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                try:
                    self.db.table('audits').update(error_update).eq('id', audit.id).execute()
                except Exception as update_error:
                    logger.error(f"Error updating audit status: {str(update_error)}")
                    
            logger.error(f"Error running performance audit: {str(e)}")
            raise

    async def run_fairness_audit(
        self, 
        project_id: int, 
        user: User,
        model_id: int,
        dataset_id: int
    ) -> AuditResponse:
        """Run fairness audit on the model"""    
        logger.info(f"Starting fairness audit for model_id={model_id}, dataset_id={dataset_id}")
        try:
            audit = await self._create_audit(
                project_id, user, "fairness", model_id, dataset_id
            )
            
            # Get model and dataset first
            logger.info("Retrieving model and dataset information")
            model = self._get_model(model_id, project_id)
            dataset = self._get_dataset(dataset_id, project_id)
            logger.info(f"Model: {model.name} ({model.model_type}), Dataset: {dataset.name}")
            
            # Log paths for debugging
            logger.info(f"Model file path: {model.file_path}")
            logger.info(f"Dataset file path: {dataset.file_path}")
            
            # Create output directory structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create directory paths
            report_dirs = {}
            categories = ['performance', 'explainability', 'fairness_bias', 'drift_robustness']
            
            if settings.USE_GCS:
                # For GCS, we just need the logical path structure
                base_path = f"{self.outputs_dir}/{project_id}/{model.name}-{model.version}"
                
                for category in categories:
                    category_path = f"{base_path}/{category}"
                    plots_path = f"{category_path}/plots"
                    metrics_path = f"{category_path}/metrics"
                    
                    # Store paths without creating actual directories (GCS doesn't need them)
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            else:
                # For local storage, create physical directories
                base_path = os.path.join(settings.OUTPUT_DIR, str(project_id), f"{model.name}-{model.version}")
                os.makedirs(base_path, exist_ok=True)
                
                for category in categories:
                    category_path = os.path.join(base_path, category)
                    plots_path = os.path.join(category_path, 'plots')
                    metrics_path = os.path.join(category_path, 'metrics')
                    
                    # Create the directories
                    os.makedirs(plots_path, exist_ok=True)
                    os.makedirs(metrics_path, exist_ok=True)
                    
                    # Store the paths
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            
            # Load model and data
            logger.info("Loading model and data...")
            loaded_model = await self._load_model(model.file_path, model.model_type)
            X, y, feature_names = await self._load_data(
                dataset_path=dataset.file_path,
                data_type=dataset.dataset_type,
                model_type=model.model_type
            )
            
            # Split data for analysis
            logger.info("Splitting data into train and test sets")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
            
            # Define comprehensive sensitive features for detection
            sensitive_features = [
                # Demographics  
                "gender", "sex", "age", "race", "ethnicity", "religion", "nationality", "citizenship",  
                "disability", "disability_status", "pregnancy_status",  

                # Socioeconomic & Employment  
                "income", "salary", "wage", "education", "education_level", "marital_status",  
                "employment", "employment_status", "occupation", "job_title",  

                # Health & Genetic Information  
                "health", "health_status", "medical_history", "genetic_data", "disorder", "disease",  

                # Location & Citizenship  
                "birthplace", "country_of_birth", "residence", "zip_code", "postal_code",  

                # Other Identifiers  
                "criminal_record", "conviction_history", "political_affiliation", "sexual_orientation"
            ]

            # Log available features for debugging
            logger.info(f"Available features in dataset: {feature_names}")
            logger.info(f"Total number of features: {len(feature_names)}")
            
            # Improved sensitive feature detection with multiple strategies
            valid_sensitive_features = []
            
            # Strategy 1: Exact matches (case-sensitive)
            available_features = set(feature_names)
            exact_matches = [f for f in sensitive_features if f in available_features]
            valid_sensitive_features.extend(exact_matches)
            if exact_matches:
                logger.info(f"Found exact matches: {exact_matches}")
            
            # Strategy 2: Case-insensitive matches
            if not valid_sensitive_features:
                logger.info("No exact matches found, trying case-insensitive matches")
                case_insensitive_matches = [
                    f for f in feature_names 
                    if any(f.lower() == s.lower() for s in sensitive_features)
                ]
                valid_sensitive_features.extend(case_insensitive_matches)
                if case_insensitive_matches:
                    logger.info(f"Found case-insensitive matches: {case_insensitive_matches}")
            
            # Strategy 3: Partial matches (contains)
            if not valid_sensitive_features:
                logger.info("No case-insensitive matches found, trying partial matches")
                partial_matches = [
                    f for f in feature_names 
                    if any(s.lower() in f.lower() or f.lower() in s.lower() for s in sensitive_features)
                ]
                valid_sensitive_features.extend(partial_matches)
                if partial_matches:
                    logger.info(f"Found partial matches: {partial_matches}")
            
            # Strategy 4: Auto-detect categorical features that could be sensitive
            if not valid_sensitive_features:
                logger.info("No direct matches found, checking for categorical features")
                categorical_features = []
                for i, feature_name in enumerate(feature_names):
                    feature_values = X_test[:, i]
                    unique_values = np.unique(feature_values)
                    
                    # Consider it categorical/potentially sensitive if:
                    # - Has few unique values (< 10% of total samples or < 20 absolute)
                    # - Not numeric or contains non-numeric patterns
                    if len(unique_values) <= min(len(X_test) * 0.1, 20):
                        categorical_features.append(feature_name)
                        logger.info(f"Detected potentially sensitive categorical feature: {feature_name} with {len(unique_values)} unique values")
                
                valid_sensitive_features.extend(categorical_features[:5])  # Limit to first 5 to avoid noise
                
            # Remove duplicates while preserving order
            valid_sensitive_features = list(dict.fromkeys(valid_sensitive_features))
            
            # Additional validation: ensure detected features actually exist in the dataset
            validated_features = []
            for feature in valid_sensitive_features:
                if feature in feature_names:
                    validated_features.append(feature)
                else:
                    logger.warning(f"Detected feature '{feature}' not found in actual dataset feature names")
            
            valid_sensitive_features = validated_features
            logger.info(f"Final validated sensitive features: {valid_sensitive_features}")
            
            if not valid_sensitive_features:
                logger.warning("No sensitive features found in the dataset")
                logger.info(f"Dataset features analyzed: {feature_names}")
                # Return a meaningful response instead of empty metrics
                fairness_metrics = {
                    'status': 'no_sensitive_features',
                    'message': 'No sensitive features found in the dataset',
                    'available_features': feature_names,
                    'searched_sensitive_features': sensitive_features,
                    'detection_strategies_used': [
                        'exact_match', 'case_insensitive_match', 'partial_match', 'categorical_detection'
                    ],
                    'recommendation': 'Please ensure your dataset includes demographic or sensitive features for fairness analysis. Consider renaming features to match common sensitive attribute names like "age", "gender", "income", etc.',
                    'sensitive_features': [],
                    'metrics': {},
                    'statistical_tests': {},
                    'interpretation': {}
                }
                
                # Use Supabase update instead of direct attribute setting
                audit_update = {
                    "results": fairness_metrics,
                    "status": "completed",
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                response = self.db.table('audits').update(audit_update).eq('id', audit.id).execute()
                
                return AuditResponse(
                    id=audit.id,
                    project_id=project_id,
                    model_id=model_id,
                    dataset_id=dataset_id,
                    user_id=getattr(user, 'id', None) if user else None,
                    audit_type="fairness",
                    status="completed",
                    results=fairness_metrics,
                    created_at=audit.created_at,
                    updated_at=datetime.utcnow().isoformat()
                )
            
            logger.info(f"Proceeding with fairness analysis for features: {valid_sensitive_features}")
            
            # Initialize fairness metrics structure
            fairness_metrics = {
                'status': 'completed',
                'dataset_info': {
                    'total_features': len(feature_names),
                    'available_features': feature_names,
                    'detected_sensitive_features': valid_sensitive_features,
                    'sample_size': len(X_test)
                },
                'sensitive_features': valid_sensitive_features,
                'metrics': {},
                'statistical_tests': {},
                'interpretation': {},
                'processing_timestamp': datetime.utcnow().isoformat()
            }
            
            # Calculate fairness metrics for each sensitive feature
            for feature in valid_sensitive_features:
                try:
                    feature_idx = feature_names.index(feature)
                    feature_values = np.unique(X_test[:, feature_idx])
                    
                    logger.info(f"Processing feature '{feature}' with {len(feature_values)} unique values: {feature_values}")
                    
                    # Initialize feature metrics structure
                    feature_metrics = {
                        'demographic_parity': {},
                        'equal_opportunity': {},
                        'equalized_odds': {},
                        'disparate_impact': {},
                        'treatment_equality': {},
                        'statistical_parity': {},
                        'statistical_tests': {},
                        'interpretation': {
                            'demographic_parity_threshold': 0.1,
                            'equal_opportunity_threshold': 0.1,
                            'disparate_impact_threshold': 0.8,
                            'statistical_parity_threshold': 0.1
                        }
                    }
                    
                    # Store group metrics for later calculations
                    group_metrics_store = {}
                    
                    # Calculate metrics for each group
                    logger.info(f"Found {len(feature_values)} unique values for feature '{feature}': {feature_values}")
                    
                    for value in feature_values:
                        mask = X_test[:, feature_idx] == value
                        group_size = np.sum(mask)
                        logger.info(f"Group '{value}' for feature '{feature}' has {group_size} samples")
                        
                        if group_size < 2:  # Skip very small groups (reduced to 2 for more inclusion)
                            logger.info(f"Skipping group {value} for feature {feature} due to small size: {group_size}")
                            continue
                        
                        try:
                            # Get predictions for this group
                            logger.info(f"Getting predictions for group {value} with {group_size} samples")
                            group_preds = loaded_model.predict(X_test[mask])
                            group_true = y_test[mask]
                            
                            logger.info(f"Group {value}: predictions shape={group_preds.shape}, true values shape={group_true.shape}")
                            logger.info(f"Group {value}: unique predictions={np.unique(group_preds)}, unique true values={np.unique(group_true)}")
                            
                            # Ensure binary classification for fairness metrics
                            if len(np.unique(group_true)) > 2:
                                logger.warning(f"Multiclass target detected for group {value}. Converting to binary (positive class = 1)")
                                group_true_binary = (group_true == 1).astype(int)
                                group_preds_binary = (group_preds == 1).astype(int)
                            else:
                                group_true_binary = group_true.astype(int)
                                group_preds_binary = group_preds.astype(int)
                            
                            logger.info(f"Group {value}: binary predictions={np.unique(group_preds_binary)}, binary true={np.unique(group_true_binary)}")
                            
                            # Calculate confusion matrix components
                            try:
                                cm = confusion_matrix(group_true_binary, group_preds_binary, labels=[0, 1])
                                logger.info(f"Group {value}: confusion matrix shape={cm.shape}, matrix=\n{cm}")
                                
                                if cm.shape == (2, 2):
                                    tn, fp, fn, tp = cm.ravel()
                                else:
                                    # Handle case where only one class is present
                                    if cm.shape == (1, 1):
                                        if np.unique(group_true_binary)[0] == 0:
                                            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
                                        else:
                                            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
                                    else:
                                        logger.warning(f"Unexpected confusion matrix shape for group {value}: {cm.shape}")
                                        continue
                                
                                logger.info(f"Group {value}: tn={tn}, fp={fp}, fn={fn}, tp={tp}")
                                
                            except Exception as cm_error:
                                logger.error(f"Error calculating confusion matrix for group {value}: {str(cm_error)}")
                                continue
                            
                            # Calculate group metrics with safer division
                            group_metrics = {
                                'size': int(group_size),
                                'true_negatives': int(tn),
                                'false_positives': int(fp),
                                'false_negatives': int(fn),
                                'true_positives': int(tp),
                                'positive_rate': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
                                'negative_rate': float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0,
                                'false_positive_rate': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
                                'false_negative_rate': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
                                'prediction_rate': float(np.sum(group_preds_binary) / len(group_preds_binary)),
                                'base_rate': float(np.sum(group_true_binary) / len(group_true_binary))
                            }
                            
                            # Store for later calculations
                            group_metrics_store[str(value)] = group_metrics
                            
                            # Calculate basic fairness metrics
                            feature_metrics['demographic_parity'][str(value)] = group_metrics['prediction_rate']
                            feature_metrics['equal_opportunity'][str(value)] = 1 - group_metrics['false_negative_rate']  # True positive rate
                            feature_metrics['equalized_odds'][str(value)] = {
                                'fpr': group_metrics['false_positive_rate'],
                                'tpr': 1 - group_metrics['false_negative_rate']
                            }
                            # Calculate treatment equality ratio with JSON-safe values
                            if group_metrics['false_negatives'] > 0:
                                te_ratio = float(group_metrics['false_positives'] / group_metrics['false_negatives'])
                            else:
                                te_ratio = 999999.0  # Use large number instead of infinity for JSON compatibility
                            
                            feature_metrics['treatment_equality'][str(value)] = {
                                'fp': group_metrics['false_positives'],
                                'fn': group_metrics['false_negatives'],
                                'ratio': te_ratio
                            }
                            
                            logger.info(f"Stored metrics for group {value}: demographic_parity={group_metrics['prediction_rate']:.3f}, equal_opportunity={1 - group_metrics['false_negative_rate']:.3f}")
                            logger.info(f"Group {value} metrics stored: prediction_rate={group_metrics['prediction_rate']:.3f}, base_rate={group_metrics['base_rate']:.3f}")
                            
                        except Exception as group_error:
                            logger.error(f"Error processing group {value} for feature {feature}: {str(group_error)}")
                            continue
                    
                    # Calculate comparative fairness metrics (disparate impact, statistical parity)
                    if len(group_metrics_store) > 1:
                        group_values = list(group_metrics_store.keys())
                        reference_group = group_values[0]  # Use first group as reference
                        reference_metrics = group_metrics_store[reference_group]
                        
                        for group_value, group_data in group_metrics_store.items():
                            if group_value != reference_group:
                                # Disparate impact ratio
                                if reference_metrics['prediction_rate'] > 0:
                                    di_ratio = group_data['prediction_rate'] / reference_metrics['prediction_rate']
                                    feature_metrics['disparate_impact'][group_value] = float(di_ratio)
                                
                                # Statistical parity difference
                                sp_diff = group_data['prediction_rate'] - reference_metrics['prediction_rate']
                                feature_metrics['statistical_parity'][group_value] = float(sp_diff)
                    
                    # Perform statistical tests for independence
                    statistical_tests = {}
                    for value in feature_values:
                        mask = X_test[:, feature_idx] == value
                        if np.sum(mask) < 5:
                            continue
                        
                        try:
                            group_preds = loaded_model.predict(X_test[mask])
                            group_true = y_test[mask]
                            
                            # Create contingency table
                            contingency_table = confusion_matrix(group_true.astype(int), group_preds.astype(int))
                            
                            # Only perform test if we have a proper 2x2 table
                            if contingency_table.shape == (2, 2) and np.sum(contingency_table) > 0:
                                chi2, p_value = chi2_contingency(contingency_table)[:2]
                                
                                # Ensure statistical test values are JSON serializable
                                chi2_val = float(chi2) if not (np.isnan(chi2) or np.isinf(chi2)) else None
                                p_val = float(p_value) if not (np.isnan(p_value) or np.isinf(p_value)) else None
                                
                                statistical_tests[str(value)] = {
                                    'chi2_statistic': chi2_val,
                                    'p_value': p_val,
                                    'significant': bool(p_val < 0.05) if p_val is not None else False
                                }
                        except Exception as test_error:
                            logger.warning(f"Could not perform statistical test for group {value}: {str(test_error)}")
                            continue
                    
                    feature_metrics['statistical_tests'] = statistical_tests
                    fairness_metrics['metrics'][feature] = feature_metrics
                    
                    # Log summary of processing
                    processed_groups = len([k for k, v in feature_metrics['demographic_parity'].items() if v is not None])
                    logger.info(f"Successfully processed {processed_groups} out of {len(feature_values)} groups for feature '{feature}'")
                    logger.info(f"Feature '{feature}' demographic_parity: {feature_metrics['demographic_parity']}")
                    logger.info(f"Successfully calculated fairness metrics for feature '{feature}'")
                    
                except Exception as feature_error:
                    logger.error(f"Error processing feature {feature}: {str(feature_error)}")
                    # Add a placeholder for this feature to indicate it was attempted
                    fairness_metrics['metrics'][feature] = {
                        'error': f"Failed to calculate metrics: {str(feature_error)}",
                        'demographic_parity': {},
                        'equal_opportunity': {},
                        'equalized_odds': {},
                        'disparate_impact': {},
                        'treatment_equality': {},
                        'statistical_parity': {},
                        'statistical_tests': {},
                        'interpretation': {
                            'demographic_parity_threshold': 0.1,
                            'equal_opportunity_threshold': 0.1,
                            'disparate_impact_threshold': 0.8,
                            'statistical_parity_threshold': 0.1
                        }
                    }
                    continue
            
            # Ensure all values are JSON serializable
            def make_json_serializable(obj):
                """Convert numpy types and other non-serializable types to JSON-compatible types"""
                if isinstance(obj, dict):
                    return {key: make_json_serializable(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return float(obj)
                elif isinstance(obj, np.bool_):
                    return bool(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif obj == float('inf') or obj == float('-inf'):
                    return 999999.0 if obj == float('inf') else -999999.0
                elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
                    return None
                else:
                    return obj
            
            # Clean the fairness metrics to ensure JSON serializability
            fairness_metrics = make_json_serializable(fairness_metrics)
            
            # Update audit with results
            
            # Use Supabase update instead of direct attribute setting and commit
            audit_update = {
                "results": fairness_metrics,
                "status": "completed",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.db.table('audits').update(audit_update).eq('id', audit.id).execute()
            
            if not response.data or len(response.data) == 0:
                raise Exception("Failed to update audit record")
                
            updated_audit = response.data[0]
            
            logger.info("Fairness audit completed successfully")
            
            # Create proper file paths based on storage type
            if settings.USE_GCS:
                # Use the metrics path directly from the dictionary
                fair_metrics_path = f"{report_dirs['fairness_bias']['metrics']}/fairness-metrics-{timestamp}.json"
                logger.info(f"Saving fairness metrics to GCS path: {fair_metrics_path}")
            else:
                # For local storage, use os.path.join
                fair_metrics_path = os.path.join(report_dirs['fairness_bias']['metrics'], f'fairness-metrics-{timestamp}.json')
                logger.info(f"Saving fairness metrics to local path: {fair_metrics_path}")
                
            self.save_metrics(fairness_metrics, fair_metrics_path)
            logger.info(f"Fairness metrics saved to: {fair_metrics_path}")
            
            # Update audit status
            await self._update_audit_status(audit.id, "completed", fair_metrics_path)
            
            # Create AuditResponse object
            audit_response = AuditResponse(
                id=audit.id,
                project_id=project_id,
                model_id=model_id,
                dataset_id=dataset_id,
                user_id=getattr(user, 'id', None) if user else None,
                audit_type="fairness",
                status="completed",
                results=fairness_metrics,
                created_at=audit.created_at,
                updated_at=datetime.utcnow().isoformat()
            )
            
            return audit_response
            
        except Exception as e:
            if 'audit' in locals():
                logger.error(f"Error in fairness audit: {str(e)}", exc_info=True)
                
                # Update audit status to failed using Supabase
                error_update = {
                    "status": "failed",
                   
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                try:
                    self.db.table('audits').update(error_update).eq('id', audit.id).execute()
                except Exception as update_error:
                    logger.error(f"Error updating audit status: {str(update_error)}")
                    
            logger.error(f"Error running fairness audit: {str(e)}")
            raise

    async def run_explainability_audit(
        self, 
        project_id: int, 
        user: User,
        model_id: int,
        dataset_id: int
    ) -> AuditResponse:
        """Run explainability audit on the model"""
        logger.info(f"Starting explainability audit for model_id={model_id}, dataset_id={dataset_id}")
        try:
            # Create audit record
            audit = await self._create_audit(
                project_id, user, "explainability", model_id, dataset_id
            )
            logger.info(f"Created audit record with ID: {audit.id}")
            
            # Get model and dataset first
            logger.info("Retrieving model and dataset information")
            model = self._get_model(model_id, project_id)
            dataset = self._get_dataset(dataset_id, project_id)
            logger.info(f"Model: {model.name} ({model.model_type}), Dataset: {dataset.name}")
            
            # Log paths for debugging
            logger.info(f"Model file path: {model.file_path}")
            logger.info(f"Dataset file path: {dataset.file_path}")
            
            # Create output directory structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create directory paths
            report_dirs = {}
            categories = ['performance', 'explainability', 'fairness_bias', 'drift_robustness']
            
            if settings.USE_GCS:
                # For GCS, we just need the logical path structure
                base_path = f"{self.outputs_dir}/{project_id}/{model.name}-{model.version}"
                
                for category in categories:
                    category_path = f"{base_path}/{category}"
                    plots_path = f"{category_path}/plots"
                    metrics_path = f"{category_path}/metrics"
                    
                    # Store paths without creating actual directories (GCS doesn't need them)
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            else:
                # For local storage, create physical directories
                base_path = os.path.join(settings.OUTPUT_DIR, str(project_id), f"{model.name}-{model.version}")
                os.makedirs(base_path, exist_ok=True)
                
                for category in categories:
                    category_path = os.path.join(base_path, category)
                    plots_path = os.path.join(category_path, 'plots')
                    metrics_path = os.path.join(category_path, 'metrics')
                    
                    # Create the directories
                    os.makedirs(plots_path, exist_ok=True)
                    os.makedirs(metrics_path, exist_ok=True)
                    
                    # Store the paths
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            
            # Load model and data
            loaded_model = await self._load_model(model.file_path, model.model_type)
            X, y, feature_names = await self._load_data(
                dataset_path=dataset.file_path,
                data_type=dataset.dataset_type,
                model_type=model.model_type
            )
            
            # Log initial data shape
            logger.info(f"Initial data shape: X={X.shape}, y={y.shape}")
            logger.info(f"Initial feature names: {feature_names}")
            
            # Validate and clean feature data
            X, feature_names = self._validate_feature_data(X, feature_names, loaded_model)
            
            # Log data shape after validation
            logger.info(f"Data shape after validation: X={X.shape}, y={y.shape}")
            logger.info(f"Feature names after validation: {feature_names}")
            
            # Get feature types
            feature_types = self._get_feature_types(X, feature_names)
            
            # Initialize metrics dictionary
            metrics = {
                'model_info': {
                    'type': model.model_type,
                    'name': model.name,
                    'version': model.version,
                    'model_class': loaded_model.__class__.__name__,
                    'expected_features': getattr(loaded_model, 'n_features_in_', X.shape[1])
                },
                'data_info': {
                    'total_samples': len(X),
                    'feature_count': X.shape[1],
                    'feature_names': feature_names,
                    'feature_types': feature_types
                }
            }
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            explainability_metrics = {}
            
            # 1. Feature Importance Analysis
            try:
                logger.info("Calculating feature importance...")
                if isinstance(loaded_model, (sklearn.naive_bayes.GaussianNB, sklearn.naive_bayes.MultinomialNB)):
                    # For Naive Bayes, use feature probabilities
                    if hasattr(loaded_model, 'feature_log_prob_'):
                        feature_importance = np.abs(loaded_model.feature_log_prob_[1] - loaded_model.feature_log_prob_[0])
                        explainability_metrics['feature_importance'] = {
                            'importances': feature_importance.tolist(),
                            'feature_names': feature_names,
                            'method': 'naive_bayes_probability'
                        }
                elif hasattr(loaded_model, 'feature_importances_'):
                    # For tree-based models
                    explainability_metrics['feature_importance'] = {
                        'importances': loaded_model.feature_importances_.tolist(),
                        'feature_names': feature_names,
                        'method': 'tree_based'
                    }
                else:
                    # Fallback to permutation importance
                    perm_importance = permutation_importance(
                        loaded_model, X_test, y_test,
                        n_repeats=10,
                        random_state=42
                    )
                    explainability_metrics['feature_importance'] = {
                        'importances': perm_importance['importances_mean'].tolist(),
                        'importances_std': perm_importance['importances_std'].tolist(),
                        'p_values': perm_importance['pvalues'].tolist(),
                        'feature_names': feature_names,
                        'method': 'permutation'
                    }
            except Exception as e:
                logger.warning(f"Feature importance analysis failed: {str(e)}")
                explainability_metrics['feature_importance'] = {
                    'error': str(e),
                    'status': 'failed'
                }
            
            # 2. SHAP Values with appropriate explainer selection
            try:
                logger.info("Calculating SHAP values...")
                
                # Select appropriate SHAP explainer based on model type
                if isinstance(loaded_model, (tf.keras.Model, tf.keras.Sequential)):
                    explainer = shap.DeepExplainer(loaded_model, X_train)
                    explainer_type = 'deep'
                elif isinstance(loaded_model, (sklearn.ensemble.RandomForestClassifier, 
                                            sklearn.ensemble.RandomForestRegressor,
                                            sklearn.ensemble.GradientBoostingClassifier,
                                            sklearn.ensemble.GradientBoostingRegressor,
                                            sklearn.tree.DecisionTreeClassifier,
                                            sklearn.tree.DecisionTreeRegressor)):
                    explainer = shap.TreeExplainer(loaded_model)
                    explainer_type = 'tree'
                elif isinstance(loaded_model, (sklearn.naive_bayes.GaussianNB, 
                                            sklearn.naive_bayes.MultinomialNB)):
                    # For Naive Bayes, use KernelExplainer with predict_proba
                    background = shap.sample(X_train, 100)
                    explainer = shap.KernelExplainer(loaded_model.predict_proba, background)
                    explainer_type = 'kernel'
                else:
                    # Default to KernelExplainer for other models
                    background = shap.sample(X_train, 100)
                    explainer = shap.KernelExplainer(loaded_model.predict, background)
                    explainer_type = 'kernel'
                
                logger.info(f"Using {explainer_type} SHAP explainer")
                
                # Calculate SHAP values for a subset of test data
                shap_values = explainer.shap_values(X_test[:100])
                
                # Store SHAP values with metadata
                if isinstance(shap_values, list):
                    explainability_metrics['shap_values'] = {
                        f'class_{i}': sv.tolist() for i, sv in enumerate(shap_values)
                    }
                    shap_importance = np.abs(shap_values[0]).mean(axis=0)
                else:
                    explainability_metrics['shap_values'] = shap_values.tolist()
                    shap_importance = np.abs(shap_values).mean(axis=0)
                
                explainability_metrics['shap_importance'] = {
                    'importances': shap_importance.tolist(),
                    'feature_names': feature_names,
                    'explainer_type': explainer_type
                }
                
            except Exception as e:
                logger.warning(f"SHAP analysis failed: {str(e)}")
                explainability_metrics['shap_values'] = {
                    'error': str(e),
                    'status': 'failed'
                }
            
            # 3. LIME Explanations with improved feature handling
            try:
                logger.info("Generating LIME explanations...")
                
                # Create LIME explainer with proper feature names and types
                explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_train,
                    feature_names=feature_names,
                    class_names=['Negative', 'Positive'] if model.model_type == 'classification' else None,
                    mode='classification' if model.model_type == 'classification' else 'regression',
                    discretize_continuous=True,
                    feature_selection='auto'
                )
                
                # Generate explanations for a few samples
                lime_explanations = {}
                for i in range(min(5, len(X_test))):
                    try:
                        # Ensure the instance has the correct number of features
                        instance = X_test[i]
                        if len(instance) != len(feature_names):
                            logger.warning(f"Skipping instance {i}: feature count mismatch")
                            continue
                            
                        exp = explainer.explain_instance(
                            instance,
                            loaded_model.predict_proba if hasattr(loaded_model, 'predict_proba') else loaded_model.predict
                        )
                        lime_explanations[f'instance_{i}'] = {
                            'prediction': float(exp.predict_proba[1] if model.model_type == 'classification' else exp.predicted_value),
                            'feature_importance': exp.as_list(),
                            'feature_names': feature_names
                        }
                    except Exception as e:
                        logger.warning(f"LIME explanation failed for instance {i}: {str(e)}")
                        continue
                
                explainability_metrics['lime_explanations'] = lime_explanations
                
            except Exception as e:
                logger.warning(f"LIME analysis failed: {str(e)}")
                explainability_metrics['lime_explanations'] = {
                    'error': str(e),
                    'status': 'failed'
                }
            
            # Update audit with results
            # Use Supabase update instead of direct attribute setting and commit
            audit_update = {
                "results": explainability_metrics,
                "status": "completed",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.db.table('audits').update(audit_update).eq('id', audit.id).execute()
            
            if not response.data or len(response.data) == 0:
                raise Exception("Failed to update audit record")
                
            updated_audit = response.data[0]
            
            logger.info("Explainability audit completed successfully")
            
            # Save metrics with different path handling for GCS vs local storage
            if settings.USE_GCS:
                exp_metric_path = f"{report_dirs['explainability']['metrics']}/explainability-metrics-{timestamp}.json"
                logger.info(f"Saving explainability metrics to GCS path: {exp_metric_path}")
            else:
                exp_metric_path = os.path.join(report_dirs['explainability']['metrics'], f'explainability-metrics-{timestamp}.json')
                logger.info(f"Saving explainability metrics to local path: {exp_metric_path}")
                
            self.save_metrics(explainability_metrics, exp_metric_path)
            logger.info(f"explainability metrics saved at: {exp_metric_path}")
            
            # Create AuditResponse object with updated audit data
            audit_response = AuditResponse(
                id=updated_audit['id'],
                project_id=updated_audit['project_id'],
                model_id=updated_audit['model_id'],
                dataset_id=updated_audit['dataset_id'],
                user_id=updated_audit['user_id'],
                audit_type=updated_audit['audit_type'],
                status=updated_audit['status'],
                results=updated_audit.get('results', {}),
                created_at=updated_audit['created_at'],
                updated_at=updated_audit.get('updated_at')
            )
            
            return audit_response
            
        except Exception as e:
            logger.error(f"Error in explainability audit: {str(e)}", exc_info=True)
            if 'audit' in locals():
                
                # Update audit status to failed using Supabase
                error_update = {
                    "status": "failed",
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                try:
                    self.db.table('audits').update(error_update).eq('id', audit.id).execute()
                except Exception as update_error:
                    logger.error(f"Error updating audit status: {str(update_error)}")
                    
            raise

    async def run_drift_audit(
        self, 
        project_id: int, 
        user: User,
        model_id: int,
        dataset_id: int
    ) -> AuditResponse:
        """Run drift audit on the model"""
        logger.info(f"Starting drift audit for model_id={model_id}, dataset_id={dataset_id}")
        try:
            # Create audit record
            audit = await self._create_audit(
                project_id, user, "drift", model_id, dataset_id
            )
            logger.info(f"Created audit record with ID: {audit.id}")
            
            # Get model and dataset first
            logger.info("Retrieving model and dataset information")
            model = self._get_model(model_id, project_id)
            dataset = self._get_dataset(dataset_id, project_id)
            logger.info(f"Model: {model.name} ({model.model_type}), Dataset: {dataset.name}")
            
            # Log paths for debugging
            logger.info(f"Model file path: {model.file_path}")
            logger.info(f"Dataset file path: {dataset.file_path}")
            
            # Create output directory structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create directory paths
            report_dirs = {}
            categories = ['performance', 'explainability', 'fairness_bias', 'drift_robustness']
            
            if settings.USE_GCS:
                # For GCS, we just need the logical path structure
                base_path = f"{self.outputs_dir}/{project_id}/{model.name}-{model.version}"
                
                for category in categories:
                    category_path = f"{base_path}/{category}"
                    plots_path = f"{category_path}/plots"
                    metrics_path = f"{category_path}/metrics"
                    
                    # Store paths without creating actual directories (GCS doesn't need them)
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            else:
                # For local storage, create physical directories
                base_path = os.path.join(settings.OUTPUT_DIR, str(project_id), f"{model.name}-{model.version}")
                os.makedirs(base_path, exist_ok=True)
                
                for category in categories:
                    category_path = os.path.join(base_path, category)
                    plots_path = os.path.join(category_path, 'plots')
                    metrics_path = os.path.join(category_path, 'metrics')
                    
                    # Create the directories
                    os.makedirs(plots_path, exist_ok=True)
                    os.makedirs(metrics_path, exist_ok=True)
                    
                    # Store the paths
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
            
            # Load data with proper model type
            X, y, feature_names = await self._load_data(
                dataset.file_path, 
                dataset.dataset_type,
                model.model_type
            )
            logger.info(f"Loaded data with shape: {X.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            logger.info(f"Split data into train: {X_train.shape}, test: {X_test.shape}")
            
            # Initialize drift metrics
            drift_metrics = {
                'model_info': {
                    'type': model.model_type,
                    'name': model.name,
                    'version': model.version
                },
                'data_info': {
                    'total_samples': len(X),
                    'feature_count': X.shape[1],
                    'feature_names': feature_names
                }
            }
            
            # Convert to DataFrame for easier handling
            X_df = pd.DataFrame(X, columns=feature_names)
            X_train_df = pd.DataFrame(X_train, columns=feature_names)
            X_test_df = pd.DataFrame(X_test, columns=feature_names)
            
            # Feature drift analysis
            feature_drift = {}
            for column in feature_names:
                try:
                    # Skip if sample size is too small
                    if len(X_train_df[column]) < 10 or len(X_test_df[column]) < 10:
                        logger.warning(f"Skipping drift analysis for {column}: insufficient samples")
                        continue
                    
                    # Handle missing values
                    if X_df[column].isnull().any():
                        logger.warning(f"Column {column} contains missing values")
                        continue
                    
                    # Handle categorical features
                    if X_df[column].dtype in ['object', 'category']:
                        # Chi-square test for categorical variables
                        contingency_table = pd.crosstab(
                            X_train_df[column], 
                            X_test_df[column]
                        )
                        chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                        feature_drift[column] = {
                            "type": "categorical",
                            "chi2_statistic": float(chi2),
                            "p_value": float(p_value)
                        }
                    # Handle numerical features
                    elif X_df[column].dtype in ['int64', 'float64']:
                        # Kolmogorov-Smirnov test for numerical variables
                        ks_stat, p_value = stats.ks_2samp(
                            X_train_df[column].dropna(),
                            X_test_df[column].dropna()
                        )
                        feature_drift[column] = {
                            "type": "numerical",
                            "ks_statistic": float(ks_stat),
                            "p_value": float(p_value)
                        }
                    else:
                        logger.warning(f"Unsupported data type for column {column}")
                        continue
                        
                except Exception as e:
                    logger.warning(f"Error analyzing drift for column {column}: {str(e)}")
                    continue
            
            drift_metrics['feature_drift'] = feature_drift
            
            # Label drift analysis
            if isinstance(y, (np.ndarray, pd.Series)):
                try:
                    # Convert to pandas Series for easier handling
                    y_train_series = pd.Series(y_train)
                    y_test_series = pd.Series(y_test)
                    
                    # Calculate distributions
                    train_label_dist = y_train_series.value_counts(normalize=True).to_dict()
                    test_label_dist = y_test_series.value_counts(normalize=True).to_dict()
                    
                    # Chi-square test for label drift
                    contingency_table = pd.crosstab(y_train_series, y_test_series)
                    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
                    
                    drift_metrics['label_drift'] = {
                        'train_distribution': train_label_dist,
                        'test_distribution': test_label_dist,
                        'chi2_statistic': float(chi2),
                        'p_value': float(p_value)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error analyzing label drift: {str(e)}")
                    drift_metrics['label_drift'] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            
            # Covariate drift analysis
            try:
                # Handle categorical features
                categorical_features = X_df.select_dtypes(include=['object', 'category']).columns
                numerical_features = X_df.select_dtypes(include=['int64', 'float64']).columns
                
                if len(numerical_features) > 0:
                    # Use Isolation Forest for numerical features
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    iso_forest.fit(X_train_df[numerical_features])
                    drift_scores = iso_forest.score_samples(X_test_df[numerical_features])
                    
                    drift_metrics['covariate_drift'] = {
                        'mean_score': float(np.mean(drift_scores)),
                        'std_score': float(np.std(drift_scores)),
                        'anomaly_rate': float(np.mean(drift_scores < 0)),
                        'feature_types': {
                            'numerical': numerical_features.tolist(),
                            'categorical': categorical_features.tolist()
                        }
                    }
                else:
                    logger.warning("No numerical features available for covariate drift analysis")
                    drift_metrics['covariate_drift'] = {
                        'error': 'No numerical features available',
                        'status': 'skipped'
                    }
                    
            except Exception as e:
                logger.warning(f"Error analyzing covariate drift: {str(e)}")
                drift_metrics['covariate_drift'] = {
                    'error': str(e),
                    'status': 'failed'
                }
            
            # Update audit with results
            # Use Supabase update instead of direct attribute setting and commit
            audit_update = {
                "results": drift_metrics,
                "status": "completed",
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            response = self.db.table('audits').update(audit_update).eq('id', audit.id).execute()
            
            if not response.data or len(response.data) == 0:
                raise Exception("Failed to update audit record")
                
            updated_audit = response.data[0]
            
            logger.info("Drift analysis completed successfully")
            
            # Create proper file paths based on storage type
            if settings.USE_GCS:
                # Use the metrics path directly from the dictionary
                drift_metric_path = f"{report_dirs['drift_robustness']['metrics']}/drift-metrics-{timestamp}.json"
                logger.info(f"Saving drift metrics to GCS path: {drift_metric_path}")
            else:
                # For local storage, use os.path.join
                drift_metric_path = os.path.join(report_dirs['drift_robustness']['metrics'], f'drift-metrics-{timestamp}.json')
                logger.info(f"Saving drift metrics to local path: {drift_metric_path}")
                
            self.save_metrics(drift_metrics, drift_metric_path)
            logger.info(f"Drift metrics saved to: {drift_metric_path}")
            
            # Update audit status
            await self._update_audit_status(audit.id, "completed", drift_metric_path)
            
            # Create AuditResponse object
            audit_response = AuditResponse(
                id=audit.id,
                project_id=project_id,
                model_id=model_id,
                dataset_id=dataset_id,
                user_id=getattr(user, 'id', None) if user else None,
                audit_type="drift",
                status="completed",
                results=drift_metrics,
                created_at=audit.created_at,
                updated_at=datetime.utcnow().isoformat()
            )
            
            return audit_response
            
        except Exception as e:
            logger.error(f"Error in drift analysis: {str(e)}", exc_info=True)
            if 'audit' in locals():
                # Update audit status to failed using Supabase
                error_update = {
                    "status": "failed",
                    "updated_at": datetime.utcnow().isoformat()
                }
                
                try:
                    self.db.table('audits').update(error_update).eq('id', audit.id).execute()
                except Exception as update_error:
                    logger.error(f"Error updating audit status: {str(update_error)}")
                    
            raise

    def get_audit_report(self, project_id: int, audit_id: int, user: User) -> Dict[str, Any]:
        """Get audit report"""
        try:
            project = self._get_project(project_id, user)
            
            # Use Supabase query instead of SQLAlchemy
            response = self.db.table('audits').select('*').eq('id', audit_id).eq('project_id', project_id).execute()
            
            if not response.data or len(response.data) == 0:
                raise ValidationError(f"Audit {audit_id} not found")
                
            audit_data = response.data[0]
                
            return {
                "id": audit_data['id'],
                "type": audit_data['audit_type'],
                "status": audit_data['status'],
                "results": audit_data.get('results', {}),
                "created_at": audit_data['created_at']
            }
        except Exception as e:
            logger.error(f"Error getting audit report: {str(e)}")
            raise

    def _generate_pdf_report(self, elements, output_path):
        """Generate PDF report from elements"""
        try:
            if settings.USE_GCS:
                # Create a temporary file
                temp_dir = "/tmp"
               
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, os.path.basename(output_path))
                logger.info(f"PDF will be saved to GCS path: {temp_path}")
                # Generate PDF locally
                
                doc = SimpleDocTemplate(temp_path, pagesize=A4)
                doc.build(elements)
                
                # Upload to GCS
                with open(temp_path, 'rb') as f:
                    content = f.read()
                    
                # Convert output_path to GCS path parts
                path_parts = output_path.split('/')
                logger.info(f"Uploading to GCS: {path_parts}")
                self.storage.bucket.blob('/'.join(path_parts)).upload_from_string(
                    content, 
                    content_type='application/pdf'
                )
                
                # Clean up
                os.remove(temp_path)
                
                # Return GCS path
                gcs_path = f"gs://{self.storage.bucket.name}/{'/'.join(path_parts)}"
                logger.info(f"PDF saved to GCS: {gcs_path}")
                return True
            else:
                # For local storage
                doc = SimpleDocTemplate(output_path, pagesize=A4)
                doc.build(elements)
                logger.info(f"PDF saved locally: {output_path}")
                return True
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return False

    def _create_executive_summary(self, metrics):
        """Create executive summary section"""
        summary = []
        
        # Performance Summary
        perf_metrics = metrics['performance']['basic_metrics']
        perf_summary = f"Model achieved {perf_metrics.get('accuracy', 0):.2%} accuracy" if 'accuracy' in perf_metrics else "Performance metrics available"
        
        # Fairness Summary
        fairness_metrics = metrics['fairness']
        fairness_summary = "Fairness analysis completed" if fairness_metrics else "No sensitive features found for fairness analysis"
        
        # Explainability Summary
        explainability_metrics = metrics['explainability']
        explainability_summary = "Explainability analysis completed" if explainability_metrics else "Could not explain the Model, check the dataset"
        
        # Drift Summary
        drift_metrics = metrics['drift']
        drift_summary = "Drift analysis completed" if drift_metrics else "Drift analysis not available"
        
        # Recommendations
        recommendations = []
        if perf_metrics.get('accuracy', 0) < 0.7:
            recommendations.append("Consider model retraining to improve accuracy")
        if fairness_metrics:
            recommendations.append("Review fairness metrics for potential biases")
        if explainability_metrics:
            recommendations.append("Ensure model predictions are interpretable by using SHAP, LIME, or other explainability techniques")
        if drift_metrics:
            recommendations.append("Monitor model drift regularly")
        
        
        summary.extend([
            Paragraph("Executive Summary", self.heading_style),
            Paragraph(f"Model Performance: {perf_summary}", self.normal_style),
            Paragraph(f"Fairness Analysis: {fairness_summary}", self.normal_style),
            Paragraph(f"Drift Analysis: {drift_summary}", self.normal_style),
            Paragraph("Recommendations:", self.subheading_style),
            *[Paragraph(f"• {rec}", self.normal_style) for rec in recommendations]
        ])
        
        return summary

    def _download_from_gcs_if_needed(self, file_path):
        """Download file from GCS to local temp file if it's a GCS path"""
        if settings.USE_GCS and file_path and (file_path.startswith("gs://") or not os.path.exists(file_path)):
            try:
                # Parse GCS path if it's in gs:// format
                if file_path.startswith("gs://"):
                    parts = file_path.replace("gs://", "").split("/")
                    bucket_name = parts[0]
                    blob_path = "/".join(parts[1:])
                else:
                    # Assume it's already in the format returned by GCS storage
                    blob_path = file_path
                
                # Create a temporary file
                temp_file = os.path.join(tempfile.gettempdir(), f"temp_img_{uuid.uuid4()}.png")
                
                # Download from GCS
                logger.info(f"Downloading image from GCS: {blob_path} to {temp_file}")
                blob = self.storage.bucket.blob(blob_path)
                blob.download_to_filename(temp_file)
                
                return temp_file
            except Exception as e:
                logger.error(f"Error downloading from GCS: {str(e)}")
                return None
        return file_path

    def _create_visualization_section(self, metrics, report_dirs):
        """Create visualization section with error handling"""
        visualizations = []
        logger.info("Creating Visualizing plots")
        try:
            # Performance visualizations
            if 'performance' in metrics:
                perf_fig = self.create_performance_visualizations(metrics['performance'], report_dirs)
                if perf_fig:
                    visualizations.append(Paragraph("Performance Visualizations", self.heading_style))
                    # Save image based on storage type
                    if settings.USE_GCS:
                        pil_img = self.fig_to_image(perf_fig)
                        buf = io.BytesIO()
                        pil_img.save(buf, format='PNG')
                        buf.seek(0)
                        visualizations.append(ReportLabImage(buf, width=400, height=300))
                        img_path = f"{report_dirs['performance']['plots']}/perf_viz.png"
                    else:

                        img_path = os.path.join(report_dirs['performance']['plots'], 'perf_viz.png')
                    saved_path = self.save_visualization(perf_fig, img_path)
                    
                    # Download from GCS if needed for ReportLabImage
                    local_path = self._download_from_gcs_if_needed(saved_path)
                    if local_path:
                        visualizations.append(ReportLabImage(local_path, width=400, height=300))
                    else:
                        visualizations.append(Paragraph("Error loading performance visualization", self.error_style))


            # Fairness visualizations
            if 'fairness' in metrics and metrics['fairness'].get('metrics'):
                fairness_fig = self.create_fairness_visualizations(metrics['fairness'], report_dirs)
                if fairness_fig:
                    visualizations.append(Paragraph("Fairness Visualizations", self.heading_style))
                    # Save image based on storage type
                    if settings.USE_GCS:
                        img_path = f"{report_dirs['fairness_bias']['plots']}/fairness_viz.png"
                    else:
                        img_path = os.path.join(report_dirs['fairness_bias']['plots'], 'fairness_viz.png')

                    saved_path = self.save_visualization(fairness_fig, img_path)
                    
                    # Download from GCS if needed for ReportLabImage
                    local_path = self._download_from_gcs_if_needed(saved_path)
                    if local_path:
                        visualizations.append(ReportLabImage(local_path, width=400, height=300))
                    else:
                        visualizations.append(Paragraph("Error loading fairness visualization", self.error_style))

            
            # Drift visualizations
            if 'drift' in metrics:
                drift_fig = self.create_drift_visualizations(metrics['drift'], report_dirs)
                if drift_fig:
                    visualizations.append(Paragraph("Drift Analysis Visualizations", self.heading_style))
                    # Save image based on storage type
                    if settings.USE_GCS:
                        img_path = f"{report_dirs['drift_robustness']['plots']}/drift_viz.png"
                    else:
                        img_path = os.path.join(report_dirs['drift_robustness']['plots'], 'drift_viz.png')

                    saved_path = self.save_visualization(drift_fig, img_path)
                    
                    # Download from GCS if needed for ReportLabImage
                    local_path = self._download_from_gcs_if_needed(saved_path)
                    if local_path:
                        visualizations.append(ReportLabImage(local_path, width=400, height=300))
                    else:
                        visualizations.append(Paragraph("Error loading drift visualization", self.error_style))


            
            # Explainability visualizations
            if 'explainability' in metrics:
                explain_fig = self.create_explainability_visualizations(metrics['explainability'], report_dirs)
                if explain_fig:
                    visualizations.append(Paragraph("Explainability Visualizations", self.heading_style))
                    # Save image based on storage type
                    if settings.USE_GCS:
                        img_path = f"{report_dirs['explainability']['plots']}/explain_viz.png"
                    else:
                        img_path = os.path.join(report_dirs['explainability']['plots'], 'explain_viz.png')

                    saved_path = self.save_visualization(explain_fig, img_path)
                    
                    # Download from GCS if needed for ReportLabImage
                    local_path = self._download_from_gcs_if_needed(saved_path)
                    if local_path:
                        visualizations.append(ReportLabImage(local_path, width=400, height=300))
                    else:
                        visualizations.append(Paragraph("Error loading explainability visualization", self.error_style))

                    
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            visualizations.append(Paragraph(f"Error generating visualizations: {str(e)}", self.error_style))
        
        return visualizations

    def _cleanup_resources(self):
        """Clean up resources after report generation"""
        plt.close('all')
        gc.collect()

    def _track_progress(self, current_step, total_steps):
        """Track progress of report generation"""
        progress = (current_step / total_steps) * 100
        logger.info(f"Report generation progress: {progress:.1f}%")

    async def generate_consolidated_report(
        self,
        project_id: int,
        user: User,
        model_id: int,
        dataset_id: int
    ) -> ReportResponse:
        """Generate consolidated report for a model"""
        try:
            # Enable TensorFlow eager execution
            tf.config.run_functions_eagerly(True)
            
            # Get model and dataset
            logger.info("Retrieving model and dataset information")
            model = self._get_model(model_id, project_id)
            dataset = self._get_dataset(dataset_id, project_id)
            
            logger.info(f"Model file path: {model.file_path}")
            logger.info(f"Dataset file path: {dataset.file_path}")
            
            # Extract and log the paths for debugging
            if settings.USE_GCS:
                # For model file
                if model.file_path.startswith("gs://"):
                    model_parts = model.file_path.replace("gs://", "").split("/")
                    model_bucket = model_parts[0]
                    model_blob_path = '/'.join(model_parts[1:])
                    logger.info(f"Model GCS path - Bucket: {model_bucket}, Blob path: {model_blob_path}")
                
                # For dataset file
                if dataset.file_path.startswith("gs://"):
                    dataset_parts = dataset.file_path.replace("gs://", "").split("/")
                    dataset_bucket = dataset_parts[0]
                    dataset_blob_path = '/'.join(dataset_parts[1:])
                    logger.info(f"Dataset GCS path - Bucket: {dataset_bucket}, Blob path: {dataset_blob_path}")
            
            # Create output directory structure
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            
            # Create directory paths
            report_dirs = {}
            categories = ['performance', 'explainability', 'fairness_bias', 'drift_robustness']
            
            if settings.USE_GCS:
                # For GCS, we just need the logical path structure
                base_path = f"{self.outputs_dir}/{project_id}/{model.name}-{model.version}"
                
                for category in categories:
                    category_path = f"{base_path}/{category}"
                    plots_path = f"{category_path}/plots"
                    metrics_path = f"{category_path}/metrics"
                    
                    # Store paths without creating actual directories (GCS doesn't need them)
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
                
                # Create PDF path for GCS
                pdf_path = f"{base_path}/{model.name}_{model.version}_{timestamp}.pdf"
                logger.info(f"PDF will be saved to GCS path: {pdf_path}")
            else:
                # For local storage, create physical directories
                base_path = os.path.join(settings.OUTPUT_DIR, str(project_id), f"{model.name}-{model.version}")
                os.makedirs(base_path, exist_ok=True)
                
                for category in categories:
                    category_path = os.path.join(base_path, category)
                    plots_path = os.path.join(category_path, 'plots')
                    metrics_path = os.path.join(category_path, 'metrics')
                    
                    # Create the directories
                    os.makedirs(plots_path, exist_ok=True)
                    os.makedirs(metrics_path, exist_ok=True)
                    
                    # Store the paths
                    report_dirs[category] = {
                        'base': category_path,
                        'plots': plots_path,
                        'metrics': metrics_path
                    }
                
                # Create PDF path for local storage
                pdf_path = os.path.join(base_path, f"{model.name}_{model.version}_{timestamp}.pdf")
                logger.info(f"PDF will be saved to local path: {pdf_path}")
            
            blockchain_hash = hashlib.sha256(pdf_path.encode()).hexdigest()
            # Initialize report elements
            elements = []
            
            # Add title and metadata
            elements.extend([
                Paragraph(f"PRISM REPORT", self.title_style),
                Paragraph(f"Powered by BlockConvey", self.right_aligned_small_style),
                Paragraph(f"Model Audit", self.heading_style),
                Paragraph(f"{model.name}", self.heading_style),
                Paragraph(f"Version: {model.version}", self.subheading_style),
                Paragraph(f"Model Type: {model.model_type}", self.normal_style),
                Paragraph(f"Model Description: {model.description}", self.normal_style),
                Paragraph(f"Dataset Name: {dataset.name}", self.normal_style),
                Paragraph(f"Generated on: {timestamp}", self.normal_style),
                Paragraph(f"Blockchain Hash: {blockchain_hash}", self.normal_style),
                Spacer(1, 20)
            ])
            
            # Track progress
            total_steps = 5
            current_step = 0
            
            # Generate performance audit
            self._track_progress(current_step, total_steps)
            try:
                performance_metrics = self.get_performance_metrics(project_id, model_id, model.version)
                logger.info(f"Performance metrics: {performance_metrics}")
            except Exception as e:
                logger.warning(f"Performance metrics not found: {e}")
                performance_metrics = None
            current_step += 1

            # Generate fairness audit
            self._track_progress(current_step, total_steps)
            try:
                fairness_metrics = self.get_fairness_metrics(project_id, model_id, model.version)
                logger.info(f"Fairness metrics: {fairness_metrics}")
            except Exception as e:
                logger.warning(f"Fairness metrics not found: {e}")
                fairness_metrics = None
            current_step += 1
            
            # Generate drift analysis
            self._track_progress(current_step, total_steps)
            try:
                drift_metrics = self.get_drift_metrics(project_id, model_id, model.version)
                logger.info(f"Data Drift metrics: {drift_metrics}")
            except Exception as e:
                logger.warning(f"Data Drift metrics not found: {e}")
                drift_metrics = None
            current_step += 1
            
            # Generate explainability audit
            self._track_progress(current_step, total_steps)
            try:
                explainability_metrics = self.get_explainability_metrics(project_id, model_id, model.version)
                logger.info(f"Explainability metrics: {explainability_metrics}")
            except Exception as e:
                logger.warning(f"Explainability metrics not found: {e}")
                explainability_metrics = None
            current_step += 1
            
            # Combine all metrics
            all_metrics = {
                'performance': performance_metrics.get('metrics', {}) if performance_metrics else {},
                'fairness': fairness_metrics.get('metrics', {}) if fairness_metrics else {},
                'drift': drift_metrics.get('metrics', {}) if drift_metrics else {},
                'explainability': explainability_metrics.get('metrics', {}) if explainability_metrics else {}
            }
            
            # Add executive summary
            elements.extend(self._create_executive_summary(all_metrics))
            elements.append(Spacer(1, 20))
            
            #basic_metrics = all_metrics['performance']['basic_metrics']

            basic_metrics = all_metrics.get('performance', {})
            
            # Table data with headers
            table_data = [["Metric", "Value"]]  # Table header
            if basic_metrics:
                for key, value in basic_metrics.items():
                    if isinstance(value, (int, float)):
                        # Format numbers with 4 decimal places
                        formatted_value = f"{value:.4f}"
                    elif isinstance(value, dict):
                        # Handle dictionary values by converting to string
                        formatted_value = str(value)
                    else:
                        # Handle other types by converting to string
                        formatted_value = str(value)
                    table_data.append([key.capitalize(), formatted_value])

                
            metrics_table = Table(table_data, colWidths=[2*inch, 4*inch])
            metrics_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(metrics_table)
            elements.append(Spacer(1, 20))
            
            # Add visualizations
            self._track_progress(current_step, total_steps)
            elements.extend(self._create_visualization_section(all_metrics, report_dirs))
            current_step += 1
            
            # Generate PDF
            if not self._generate_pdf_report(elements, pdf_path):
                raise Exception("Failed to generate PDF report")
            
            # Clean up resources
            self._cleanup_resources()
            
            # Create report record
            logger.info(f"Creating report record with project_id={project_id}, model_id={model_id}, dataset_id={dataset_id}")
            
            # Determine the correct file path for the database record
            if settings.USE_GCS:
                file_path_for_db = f"gs://{self.storage.bucket.name}/{pdf_path}"
            else:
                file_path_for_db = pdf_path
                
            report_data = {
                "model_id": model_id,
                "dataset_id": dataset_id,
                "project_id": project_id,
                "report_type": "consolidated",
                "file_path": file_path_for_db,
                "blockchain_hash": blockchain_hash,
                "report_metadata": {
                    "timestamp": timestamp,
                    "model_name": model.name,
                    "model_version": model.version,
                    "dataset_name": dataset.name
                },
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            }
            
            logger.info("Report data prepared successfully")
            
            try:
                # Use Supabase insert instead of SQLAlchemy add
                response = self.db.table('reports').insert(report_data).execute()
                
                if not response.data or len(response.data) == 0:
                    raise Exception("Failed to create report record")
                    
                report_response = response.data[0]
                logger.info(f"Report saved to database with ID: {report_response['id']}")
                
                # Convert response to ReportResponse format
                return {
                    "id": report_response['id'],
                    "project_id": report_response['project_id'],
                    "model_id": report_response['model_id'],
                    "dataset_id": report_response['dataset_id'],
                    "report_type": report_response['report_type'],
                    "file_path": report_response['file_path'],
                    "blockchain_hash": report_response['blockchain_hash'],
                    "report_metadata": report_response.get('report_metadata', {}),
                    "created_at": report_response['created_at'],
                    "updated_at": report_response.get('updated_at')
                }
            except Exception as e:
                logger.error(f"Error saving report to database: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error generating consolidated report: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            # Ensure resources are cleaned up
            self._cleanup_resources()

    def create_performance_visualizations(self, metrics, report_dirs) -> Optional[PILImage]:
        """Create performance metric visualizations"""
        logger.info("Creating Visualizing plots for performance metrics")
        # Create a figure with subplots based on model type
        if metrics['model_info']['type'] == 'classification':
            fig, axes = plt.subplots(2, 2, figsize=(15, 18))
            fig.suptitle('Classification Model Performance Analysis', fontsize=16)
            
            # 1. Basic Metrics Bar Chart
            basic_metrics = metrics['basic_metrics']
            metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            metric_values = [
                basic_metrics['accuracy'],
                basic_metrics['precision'],
                basic_metrics['recall'],
                basic_metrics['f1']
            ]
            axes[0, 0].bar(metric_names, metric_values)
            axes[0, 0].set_title('Basic Classification Metrics')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].set_ylim(0, 1)
            
            # 2. Class Distribution
            if 'class_distribution' in metrics['data_info']:
                class_dist = metrics['data_info']['class_distribution']
                if isinstance(class_dist, dict):
                    # Convert class labels to strings if they're not already
                    labels = [str(k) for k in class_dist.keys()]
                    values = list(class_dist.values())
                    
                    # Create a pie chart with percentage labels
                    axes[0, 1].pie(
                        values,
                        labels=labels,
                        autopct='%1.1f%%',
                        colors=plt.cm.Set3.colors
                    )
                    axes[0, 1].set_title('Class Distribution')
                    
                    # Add a legend if there are many classes
                    if len(labels) > 5:
                        axes[0, 1].legend(
                            labels,
                            title="Classes",
                            loc="center left",
                            bbox_to_anchor=(1, 0, 0.5, 1)
                        )
                else:
                    axes[0, 1].text(0.5, 0.5, 'Invalid class distribution data', 
                                ha='center', va='center')
                    axes[0, 1].set_title('Class Distribution')
            else:
                axes[0, 1].text(0.5, 0.5, 'No class distribution data available', 
                            ha='center', va='center')
                axes[0, 1].set_title('Class Distribution')            
                        
            # 3. ROC Curve
            if 'roc_curve' in metrics:
                if 'status' in metrics['roc_curve'] and metrics['roc_curve']['status'] == 'failed':
                    axes[1, 0].text(0.5, 0.5, f"ROC curve calculation failed: {metrics['roc_curve'].get('error', 'Unknown error')}", 
                                   ha='center', va='center')
                    axes[1, 0].set_title('ROC Curve')
                else:
                    # Check if it's a multiclass ROC curve
                    if isinstance(metrics['roc_curve'], dict) and any(key.startswith('class_') for key in metrics['roc_curve'].keys()):
                        # Plot ROC curve for each class
                        for class_key, class_data in metrics['roc_curve'].items():
                            if isinstance(class_data, dict) and 'fpr' in class_data and 'tpr' in class_data:
                                fpr, tpr = class_data['fpr'], class_data['tpr']
                                auc_score = class_data.get('auc', 0)
                                axes[1, 0].plot(fpr, tpr, label=f'{class_key} (AUC = {auc_score:.2f})')
                        
                        axes[1, 0].plot([0, 1], [0, 1], 'k--')  # Diagonal line
                        axes[1, 0].set_xlabel('False Positive Rate')
                        axes[1, 0].set_ylabel('True Positive Rate')
                        axes[1, 0].set_title('Multiclass ROC Curves')
                        axes[1, 0].legend()
                    else:
                        # Binary classification case
                        fpr, tpr = metrics['roc_curve']['fpr'], metrics['roc_curve']['tpr']
                        auc_score = metrics['roc_curve'].get('auc', 0)
                        axes[1, 0].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_score:.2f})')
                        axes[1, 0].plot([0, 1], [0, 1], 'k--')
                        axes[1, 0].set_xlabel('False Positive Rate')
                        axes[1, 0].set_ylabel('True Positive Rate')
                        axes[1, 0].set_title('ROC Curve')
                        axes[1, 0].legend()
            else:
                axes[1, 0].text(0.5, 0.5, 'No ROC curve data available', 
                                ha='center', va='center')
                axes[1, 0].set_title('ROC Curve')
            
            # 4. Learning Curve
            if 'learning_curve' in metrics:
                train_sizes = metrics['learning_curve']['train_sizes']
                train_scores = np.array(metrics['learning_curve']['train_scores'])
                test_scores = np.array(metrics['learning_curve']['test_scores'])
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                axes[1, 1].plot(train_sizes, train_mean, label='Training Score')
                axes[1, 1].fill_between(
                    train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.1
                )
                axes[1, 1].plot(train_sizes, test_mean, label='Cross-validation Score')
                axes[1, 1].fill_between(
                    train_sizes,
                    test_mean - test_std,
                    test_mean + test_std,
                    alpha=0.1
                )
                axes[1, 1].set_xlabel('Training Examples')
                axes[1, 1].set_ylabel('Score')
                axes[1, 1].set_title('Learning Curve')
                axes[1, 1].legend()
                
            else:
                axes[1, 1].text(0.5, 0.5, 'No Learning curve data available', 
                                ha='center', va='center')
                axes[1, 1].set_title('Learning Curve')  
            

        else:  # Regression
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Regression Model Performance Analysis', fontsize=16)
            
            # 1. Basic Metrics Bar Chart
            basic_metrics = metrics['basic_metrics']
            metric_names = ['MSE', 'RMSE', 'MAE', 'R²']
            metric_values = [
                basic_metrics['mse'],
                basic_metrics['rmse'],
                basic_metrics['mae'],
                basic_metrics['r2']
            ]
            axes[0, 0].bar(metric_names, metric_values)
            axes[0, 0].set_title('Basic Regression Metrics')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            
            # 2. Residual Analysis
            if 'residual_analysis' in metrics:
                residuals = metrics['residual_analysis']['residuals']
                axes[0, 1].hist(residuals, bins=50, edgecolor='black')
                axes[0, 1].set_xlabel('Residuals')
                axes[0, 1].set_ylabel('Count')
                axes[0, 1].set_title('Residual Distribution')
            
            # 3. Learning Curve
            if 'learning_curve' in metrics:
                train_sizes = metrics['learning_curve']['train_sizes']
                train_scores = np.array(metrics['learning_curve']['train_scores'])
                test_scores = np.array(metrics['learning_curve']['test_scores'])
                
                train_mean = np.mean(train_scores, axis=1)
                train_std = np.std(train_scores, axis=1)
                test_mean = np.mean(test_scores, axis=1)
                test_std = np.std(test_scores, axis=1)
                
                axes[1, 0].plot(train_sizes, train_mean, label='Training Score')
                axes[1, 0].fill_between(
                    train_sizes,
                    train_mean - train_std,
                    train_mean + train_std,
                    alpha=0.1
                )
                axes[1, 0].plot(train_sizes, test_mean, label='Cross-validation Score')
                axes[1, 0].fill_between(
                    train_sizes,
                    test_mean - test_std,
                    test_mean + test_std,
                    alpha=0.1
                )
                axes[1, 0].set_xlabel('Training Examples')
                axes[1, 0].set_ylabel('Score')
                axes[1, 0].set_title('Learning Curve')
                axes[1, 0].legend()
            
            # 4. Cross-validation Scores
            if 'cross_validation' in metrics:
                cv_scores = metrics['cross_validation']['scores']
                axes[1, 1].boxplot(cv_scores)
                axes[1, 1].set_title('Cross-validation Score Distribution')
                axes[1, 1].set_ylabel('Score')
        
        plt.tight_layout()
        #timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        #perf_plot_path = os.path.join(report_dirs['performance'], 'plots', f'performance_plots_{timestamp}.png')
        # Use save_visualization instead of direct fig.savefig
       # self.save_visualization(fig, perf_plot_path)
        #logger.info(f'Plot saved at: {perf_plot_path}')
        # Convert to PIL image
        #image = self.fig_to_image(fig)
        plt.close(fig)
        return fig
    
    def create_fairness_visualizations(self, metrics, report_dirs) -> Optional[plt.Figure]:
        """Create fairness metric visualizations"""
        logger.info("Creating Visualizing plots for fairness metrics")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Fairness Metrics Visualization', fontsize=16)
        
        # Demographic parity plot with confidence intervals
        if 'metrics' in metrics:
            for feature, feature_metrics in metrics['metrics'].items():
                if 'demographic_parity' in feature_metrics:
                    groups = list(feature_metrics['demographic_parity'].keys())
                    rates = list(feature_metrics['demographic_parity'].values())
                    axes[0, 0].bar(groups, rates)
                    axes[0, 0].set_title(f'Demographic Parity: {feature}')
                    axes[0, 0].tick_params(axis='x', rotation=45)
                    # Add threshold line
                    threshold = feature_metrics['interpretation']['demographic_parity_threshold']
                    axes[0, 0].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
                    axes[0, 0].legend()
        
        # Equal opportunity plot with statistical significance
        if 'metrics' in metrics:
            for feature, feature_metrics in metrics['metrics'].items():
                if 'equal_opportunity' in feature_metrics:
                    groups = list(feature_metrics['equal_opportunity'].keys())
                    rates = list(feature_metrics['equal_opportunity'].values())
                    axes[0, 1].bar(groups, rates)
                    axes[0, 1].set_title(f'Equal Opportunity: {feature}')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                    # Add threshold line
                    threshold = feature_metrics['interpretation']['equal_opportunity_threshold']
                    axes[0, 1].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
                    axes[0, 1].legend()
        
        # Disparate impact plot with interpretation
        if 'metrics' in metrics:
            for feature, feature_metrics in metrics['metrics'].items():
                if 'disparate_impact' in feature_metrics:
                    groups = list(feature_metrics['disparate_impact'].keys())
                    ratios = list(feature_metrics['disparate_impact'].values())
                    axes[1, 0].bar(groups, ratios)
                    axes[1, 0].set_title(f'Disparate Impact: {feature}')
                    axes[1, 0].tick_params(axis='x', rotation=45)
                    # Add threshold line
                    threshold = feature_metrics['interpretation']['disparate_impact_threshold']
                    axes[1, 0].axhline(y=threshold, color='r', linestyle='--', label='Threshold')
                    axes[1, 0].legend()
        
        # Statistical significance heatmap
        if 'statistical_tests' in metrics:
            for feature, tests in metrics['statistical_tests'].items():
                groups = list(tests.keys())
                p_values = [test['p_value'] for test in tests.values()]
                axes[1, 1].bar(groups, p_values)
                axes[1, 1].set_title(f'Statistical Significance: {feature}')
                axes[1, 1].tick_params(axis='x', rotation=45)
                # Add significance threshold
                axes[1, 1].axhline(y=0.05, color='r', linestyle='--', label='p=0.05')
                axes[1, 1].legend()
        
        plt.tight_layout()
        #timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        #fair_plot_path = os.path.join(report_dirs['fairness_bias'], 'plots', f'fairness_plots_{timestamp}.png')
        # Use save_visualization instead of direct fig.savefig
        #self.save_visualization(fig, fair_plot_path)
        #logger.info(f'Plot saved at: {fair_plot_path}')
        # Convert to PIL image
        #image = self.fig_to_image(fig)
        plt.close(fig)
        return fig
    
    def create_drift_visualizations(self, metrics, report_dirs) -> Optional[plt.Figure]:
        """Create drift analysis visualizations"""
        logger.info("Creating Visualizing plots for drift metrics")
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Drift Analysis Visualization', fontsize=16)
        
        # Feature drift plot
        if 'feature_drift' in metrics:
            features = list(metrics['feature_drift'].keys())
            drift_scores = [m['ks_statistic'] for m in metrics['feature_drift'].values()]
            axes[0, 0].bar(features, drift_scores)
            axes[0, 0].set_title('Feature Drift Scores')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Label drift plot
        if 'label_drift' in metrics and isinstance(metrics['label_drift'], dict):
            label_drift = metrics['label_drift']
            if 'train_distribution' in label_drift and 'test_distribution' in label_drift:
                train_dist = label_drift['train_distribution']
                test_dist = label_drift['test_distribution']
                x = np.arange(len(train_dist))
                width = 0.35
                axes[0, 1].bar(x - width/2, list(train_dist.values()), width, label='Train')
                axes[0, 1].bar(x + width/2, list(test_dist.values()), width, label='Test')
                axes[0, 1].set_title('Label Distribution Drift')
                axes[0, 1].set_xticks(x)
                axes[0, 1].set_xticklabels(list(train_dist.keys()))
                axes[0, 1].legend()
            
            # Chi-Square Statistic & P-value visualization
            if 'chi2_statistic' in label_drift and 'p_value' in label_drift:
                chi2_value = label_drift['chi2_statistic']
                p_value = label_drift['p_value']
                axes[1, 1].bar(['Chi-Square', 'P-Value'], [chi2_value, p_value], color=['blue', 'red'])
                axes[1, 1].set_title('Label Drift Statistics')
            
            elif 'error' in label_drift:
                axes[1, 1].text(0.5, 0.5, label_drift['error'], fontsize=12, ha='center', va='center')
                axes[1, 1].set_title('Label Drift (Failed)')
        
        # Covariate drift plot
        if 'covariate_drift' in metrics and isinstance(metrics['covariate_drift'], dict):
            cov_drift = metrics['covariate_drift']
            if 'mean_score' in cov_drift and 'std_score' in cov_drift:
                scores = [cov_drift['mean_score'], cov_drift['std_score'], cov_drift['anomaly_rate']]
                labels = ['Mean Score', 'Std Dev', 'Anomaly Rate']
                axes[1, 0].bar(labels, scores, color=['blue', 'red', 'green'])
                axes[1, 0].set_title('Covariate Drift Metrics')
            elif 'error' in cov_drift:
                axes[1, 0].text(0.5, 0.5, cov_drift['error'], fontsize=12, ha='center', va='center')
                axes[1, 0].set_title('Covariate Drift (Skipped)')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        
        plt.tight_layout()
        #timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        #drift_plot_path = os.path.join(report_dirs['drift_robustness'], 'plots', f'drift_plots_{timestamp}.png')
        # Use save_visualization instead of direct fig.savefig
        #self.save_visualization(fig, drift_plot_path)
        #logger.info(f'Plot saved at: {drift_plot_path}')
        # Convert to PIL image
        #image = self.fig_to_image(fig)
        plt.close(fig)
        return fig
    
    def create_explainability_visualizations(self, metrics, report_dirs) -> Optional[plt.Figure]:
        """Create explainability metric visualizations"""
        logger.info("Creating Visualizing plots for explainability metrics")
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Model Explainability Analysis', fontsize=16)
            
            # SHAP Values Summary Plot
            if 'shap_importance' in metrics and metrics['shap_importance']:
                importances = metrics['shap_importance'].get('importances', [])
                feature_names = metrics['shap_importance'].get('feature_names', [])
                if importances and feature_names:
                    importances = np.array(importances)
                    if importances.ndim > 1:
                        importances = importances.mean(axis=0)
                    sorted_idx = np.argsort(importances)
                    axes[0, 0].barh(range(len(importances)), importances[sorted_idx])
                    axes[0, 0].set_yticks(range(len(importances)))
                    axes[0, 0].set_yticklabels([feature_names[i] for i in sorted_idx])
                    axes[0, 0].set_title('SHAP Feature Importance')
                else:
                    axes[0, 0].text(0.5, 0.5, 'No SHAP importance data available', 
                                  ha='center', va='center')
                    axes[0, 0].set_title('SHAP Feature Importance')
            else:
                axes[0, 0].text(0.5, 0.5, 'No SHAP data available', 
                              ha='center', va='center')
                axes[0, 0].set_title('SHAP Feature Importance')
            
            # Feature Importance Plot
            if 'feature_importance' in metrics and metrics['feature_importance']:
                importances = metrics['feature_importance'].get('importances', [])
                feature_names = metrics['feature_importance'].get('feature_names', [])
                if importances and feature_names:
                    importances = np.array(importances)
                    if importances.ndim > 1:
                        importances = importances.mean(axis=0)
                    sorted_idx = np.argsort(importances)
                    axes[0, 1].barh(range(len(importances)), importances[sorted_idx])
                    axes[0, 1].set_yticks(range(len(importances)))
                    axes[0, 1].set_yticklabels([feature_names[i] for i in sorted_idx])
                    axes[0, 1].set_title('Feature Importance')
                else:
                    axes[0, 1].text(0.5, 0.5, 'No feature importance data available', 
                                  ha='center', va='center')
                    axes[0, 1].set_title('Feature Importance')
            else:
                axes[0, 1].text(0.5, 0.5, 'No feature importance data available', 
                              ha='center', va='center')
                axes[0, 1].set_title('Feature Importance')
            
            # LIME Explanations Plot
            if 'lime_explanations' in metrics and metrics['lime_explanations']:
                # Plot the first instance's LIME explanation
                first_instance = next(iter(metrics['lime_explanations'].values()))
                if 'feature_importance' in first_instance:
                    features = [x[0] for x in first_instance['feature_importance']]
                    values = [x[1] for x in first_instance['feature_importance']]
                    values = np.array(values)
                    if values.ndim > 1:
                        values = values.mean(axis=0)
                    sorted_idx = np.argsort(values)
                    axes[1, 0].barh(range(len(values)), values[sorted_idx])
                    axes[1, 0].set_yticks(range(len(values)))
                    axes[1, 0].set_yticklabels([features[i] for i in sorted_idx])
                    axes[1, 0].set_title('LIME Feature Importance (First Instance)')
                else:
                    axes[1, 0].text(0.5, 0.5, 'No LIME feature importance data available', 
                                  ha='center', va='center')
                    axes[1, 0].set_title('LIME Feature Importance')
            else:
                axes[1, 0].text(0.5, 0.5, 'No LIME explanations available', 
                              ha='center', va='center')
                axes[1, 0].set_title('LIME Feature Importance')
            
            # Model Info Plot
            if 'model_info' in metrics and metrics['model_info']:
                info = metrics['model_info']
                info_text = f"Model Type: {info.get('type', 'N/A')}\n"
                info_text += f"Model Class: {info.get('model_class', 'N/A')}\n"
                info_text += f"Features: {info.get('expected_features', 'N/A')}"
                axes[1, 1].text(0.5, 0.5, info_text, 
                              ha='center', va='center', fontsize=10)
                axes[1, 1].set_title('Model Information')
            else:
                axes[1, 1].text(0.5, 0.5, 'No model information available', 
                              ha='center', va='center')
                axes[1, 1].set_title('Model Information')
            
            plt.tight_layout()
            
            # Save the plot
            #timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            #explain_plot_path = os.path.join(report_dirs['explainability'], 'plots', f'explainability_plots_{timestamp}.png')
            # Use save_visualization instead of direct fig.savefig
            #self.save_visualization(fig, explain_plot_path)
            #logger.info(f'Plot saved at: {explain_plot_path}')
            # Convert to PIL image
            #image = self.fig_to_image(fig)
            plt.close(fig)    
            return fig
            
        except Exception as e:
            logger.error(f"Error creating explainability visualizations: {str(e)}")
            return None

    
    def fig_to_image(self, fig) -> Optional[PILImage]:
        """Convert matplotlib figure to PIL Image"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        # Use PILImage instead of reportlab's Image
        img = PILImage.open(buf)
        # Return PIL Image
        return img
            
    def _validate_model_compatibility(self, model: Any, method: str) -> bool:
        """Validate model compatibility with explainability method"""
        try:
            if method == 'shap':
                # Check for SHAP compatibility
                if isinstance(model, (tf.keras.Model, tf.keras.Sequential)):
                    return True
                if hasattr(model, 'predict_proba'):
                    return True
                return hasattr(model, 'predict')
            elif method == 'lime':
                # Check for LIME compatibility
                return hasattr(model, 'predict') or hasattr(model, 'predict_proba')
            elif method == 'pdp':
                # Check for PDP compatibility
                return hasattr(model, 'predict')
            return True
        except Exception as e:
            logger.error(f"Error validating model compatibility: {str(e)}")
            return False

    def _process_in_batches(self, data: np.ndarray, batch_size: int = 100) -> List[np.ndarray]:
        """Process data in batches to manage memory"""
        try:
            if len(data) <= batch_size:
                return [data]
            return [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        except Exception as e:
            logger.error(f"Error processing batches: {str(e)}")
            return [data]  # Return original data if batch processing fails

    def _monitor_memory_usage(self) -> float:
        """Monitor current memory usage"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except Exception as e:
            logger.warning(f"Error monitoring memory usage: {str(e)}")
            return 0.0

    def _is_categorical_feature(self, feature_values: np.ndarray) -> bool:
        """Check if a feature is categorical"""
        try:
            unique_values = np.unique(feature_values)
            # Consider feature categorical if:
            # 1. Number of unique values is small
            # 2. Values are integers
            # 3. Values are strings
            return (len(unique_values) < 10 or 
                    np.issubdtype(feature_values.dtype, np.integer) or
                    feature_values.dtype.kind in ['U', 'S'])
        except Exception as e:
            logger.warning(f"Error checking categorical feature: {str(e)}")
            return False

    def _combine_batch_results(self, batch_results: List[Any]) -> Any:
        """Combine results from batch processing"""
        try:
            if not batch_results:
                return None
            
            # Handle permutation importance results
            if isinstance(batch_results[0], dict) and 'importances_mean' in batch_results[0]:
                combined = {
                    'importances_mean': np.mean([r['importances_mean'] for r in batch_results], axis=0),
                    'importances_std': np.mean([r['importances_std'] for r in batch_results], axis=0),
                    'pvalues': np.mean([r['pvalues'] for r in batch_results], axis=0)
                }
                return combined
            
            # Handle other types of results
            return np.mean(batch_results, axis=0)
        except Exception as e:
            logger.error(f"Error combining batch results: {str(e)}")
            return batch_results[0]  # Return first result if combination fails

    def _combine_shap_values(self, shap_values_list: List[Any]) -> Any:
        """Combine SHAP values from batch processing"""
        try:
            if not shap_values_list:
                return None
            
            # Handle list of SHAP values (multi-class case)
            if isinstance(shap_values_list[0], list):
                combined = []
                for i in range(len(shap_values_list[0])):
                    class_values = [sv[i] for sv in shap_values_list]
                    combined.append(np.concatenate(class_values, axis=0))
                return combined
            
            # Handle single SHAP values (binary case)
            return np.concatenate(shap_values_list, axis=0)
        except Exception as e:
            logger.error(f"Error combining SHAP values: {str(e)}")
            return shap_values_list[0]  # Return first result if combination fails

    def _validate_feature_data(self, X: np.ndarray, feature_names: List[str], model: Any) -> Tuple[np.ndarray, List[str]]:
        """Validate and clean feature data"""
        try:
            # Log initial feature count
            logger.info(f"Initial feature count: {X.shape[1]}")
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0)
            
            # Remove constant features
            non_constant_features = np.where(X.std(axis=0) != 0)[0]
            X = X[:, non_constant_features]
            feature_names = [feature_names[i] for i in non_constant_features]
            
            # Handle infinite values
            X = np.nan_to_num(X, posinf=0.0, neginf=0.0)
            
            # Ensure feature count matches model's expected features
            if hasattr(model, 'n_features_in_'):
                expected_features = model.n_features_in_
                if X.shape[1] != expected_features:
                    logger.warning(f"Feature count mismatch: data has {X.shape[1]} features, model expects {expected_features}")
                    # Select only the first n features to match model's expectation
                    X = X[:, :expected_features]
                    feature_names = feature_names[:expected_features]
                    logger.info(f"Adjusted feature count to match model: {X.shape[1]}")
            elif hasattr(model, 'n_features_'):
                expected_features = model.n_features_
                if X.shape[1] != expected_features:
                    logger.warning(f"Feature count mismatch: data has {X.shape[1]} features, model expects {expected_features}")
                    # Select only the first n features to match model's expectation
                    X = X[:, :expected_features]
                    feature_names = feature_names[:expected_features]
                    logger.info(f"Adjusted feature count to match model: {X.shape[1]}")
            
            # Log final feature count
            logger.info(f"Final feature count after cleaning: {X.shape[1]}")
            
            return X, feature_names
        except Exception as e:
            logger.error(f"Error validating feature data: {str(e)}")
            return X, feature_names

    def _get_feature_types(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, str]:
        """Determine feature types"""
        try:
            feature_types = {}
            for i, name in enumerate(feature_names):
                if self._is_categorical_feature(X[:, i]):
                    feature_types[name] = 'categorical'
                else:
                    feature_types[name] = 'continuous'
            return feature_types
        except Exception as e:
            logger.error(f"Error determining feature types: {str(e)}")
            return {}

    def _get_latest_metrics_file(self, metrics_dir: str) -> str:
        """Get the latest metrics file from the directory"""
        try:
            logger.info(f"Looking for latest metrics file in: {metrics_dir}")
            
            if settings.USE_GCS and metrics_dir.startswith("gs://"):
                # Extract bucket name and path from GCS URL
                parts = metrics_dir.replace("gs://", "").split("/")
                bucket_name = parts[0]
                prefix = '/'.join(parts[1:])
                
                logger.info(f"Searching GCS bucket: {bucket_name}, prefix: {prefix}")
                
                # List blobs with the specified prefix
                blobs = list(self.storage.bucket.list_blobs(prefix=prefix))
                logger.info(f"Found {len(blobs)} blobs with prefix: {prefix}")
                
                metric_files = [blob.name for blob in blobs if blob.name.endswith('.json')]
                
                if not metric_files:
                    logger.warning(f"No metrics files found in GCS path: {metrics_dir}")
                    raise FileNotFoundError(f"No metrics found in GCS path: {metrics_dir}")
                
                logger.info(f"Found {len(metric_files)} metrics files in GCS: {metric_files}")
                
                # Sort files by timestamp in filename and get the latest
                latest_file = max(metric_files, key=lambda x: x.split('_')[-1].split('.')[0])
                full_path = f"gs://{bucket_name}/{latest_file}"
                logger.info(f"Latest metrics file: {full_path}")
                return full_path
            else:
                # Local file system
                logger.info(f"Searching local directory: {metrics_dir}")
                
                if not os.path.exists(metrics_dir):
                    logger.warning(f"Metrics directory does not exist: {metrics_dir}")
                    raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
                
                # Get all JSON files in the metrics directory
                metric_files = [f for f in os.listdir(metrics_dir) if f.endswith('.json')]
                if not metric_files:
                    logger.warning(f"No metrics files found in directory: {metrics_dir}")
                    raise FileNotFoundError(f"No metrics found")
                
                logger.info(f"Found {len(metric_files)} metrics files locally: {metric_files}")
                
                # Sort files by timestamp in filename and get the latest
                latest_file = max(metric_files, key=lambda x: x.split('_')[-1].split('.')[0])
                full_path = os.path.join(metrics_dir, latest_file)
                logger.info(f"Latest metrics file: {full_path}")
                return full_path
        except Exception as e:
            logger.error(f"Error getting latest metrics file: {str(e)}")
            raise

    def get_performance_metrics(self, project_id: int, model_id: int, model_version: str) -> Dict[str, Any]:
        """Get performance metrics for a model"""
        try:
            # Get model
            model = self._get_model(model_id, project_id)
            logger.info(f"Model: {model}")
            
            # Verify model version
            if model.version != model_version:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model version mismatch. Expected {model_version}, got {model.version}"
                )
            
            metrics = None
            error_messages = []
            
            # Try to get metrics from GCS
            if settings.USE_GCS:
                logger.info(f"Looking for metrics in GCS bucket for project {project_id}, model {model.name}-{model.version}")
                
                # First, try exact model name match
                try:
                    primary_pattern = f"outputs/{project_id}/{model.name}-{model.version}/performance/metrics/"
                    logger.info(f"Searching with primary pattern: {primary_pattern}")
                    
                    # List blobs with this prefix
                    blobs = list(self.storage.bucket.list_blobs(prefix=primary_pattern))
                    logger.info(f"Found {len(blobs)} blobs with primary pattern")
                    
                    # Filter for JSON files
                    json_blobs = [b for b in blobs if b.name.endswith('.json')]
                    if json_blobs:
                        # Get the latest blob based on update time
                        latest_blob = max(json_blobs, key=lambda b: b.updated)
                        logger.info(f"Found metrics file: {latest_blob.name}")
                        
                        # Download and parse the metrics
                        metrics = json.loads(latest_blob.download_as_string())
                        logger.info(f"Successfully retrieved metrics from: {latest_blob.name}")
                    else:
                        error_message = f"No JSON files found in primary path"
                        logger.warning(error_message)
                        error_messages.append(error_message)
                except Exception as e:
                    error_message = f"Error accessing primary path: {str(e)}"
                    logger.warning(error_message)
                    error_messages.append(error_message)
                
                # If primary pattern didn't work, try a broader search for directories with the version
                if metrics is None:
                    try:
                        logger.info(f"Performing broader search by version in project {project_id}")
                        prefix = f"outputs/{project_id}/"
                        
                        # List all blobs in project folder
                        all_blobs = list(self.storage.bucket.list_blobs(prefix=prefix))
                        logger.info(f"Found {len(all_blobs)} total blobs in project folder")
                        
                        # Find all directories that contain the model version
                        version_dirs = set()
                        for blob in all_blobs:
                            # Extract potential directory pattern that includes the version
                            path_parts = blob.name.split('/')
                            for i, part in enumerate(path_parts):
                                if model.version in part and i > 1:  # Skip 'outputs' and project_id parts
                                    # Get the directory name containing the version
                                    version_dir = '/'.join(path_parts[:i+1])
                                    version_dirs.add(version_dir)
                        
                        logger.info(f"Found {len(version_dirs)} directories containing version {model.version}: {version_dirs}")
                        
                        # Check each potential directory for performance metrics
                        for version_dir in version_dirs:
                            try:
                                metrics_path = f"{version_dir}/performance/metrics/"
                                logger.info(f"Checking for metrics in: {metrics_path}")
                                
                                metrics_blobs = [
                                    b for b in all_blobs 
                                    if b.name.startswith(metrics_path) and b.name.endswith('.json')
                                ]
                                
                                if metrics_blobs:
                                    latest_blob = max(metrics_blobs, key=lambda b: b.updated)
                                    logger.info(f"Found metrics in version directory: {latest_blob.name}")
                                    metrics = json.loads(latest_blob.download_as_string())
                                    break
                            except Exception as inner_e:
                                logger.warning(f"Error checking directory {version_dir}: {str(inner_e)}")
                                continue
                        
                        # If still no metrics, look for any performance metrics in the project
                        if metrics is None:
                            logger.info("Searching for any performance metrics in the project")
                            perf_metrics_blobs = [
                                b for b in all_blobs 
                                if b.name.endswith('.json') and '/performance/metrics/' in b.name
                            ]
                            
                            if perf_metrics_blobs:
                                latest_blob = max(perf_metrics_blobs, key=lambda b: b.updated)
                                logger.info(f"Found general performance metrics: {latest_blob.name}")
                                metrics = json.loads(latest_blob.download_as_string())
                            else:
                                error_message = "No performance metrics JSON files found in project"
                                logger.warning(error_message)
                                error_messages.append(error_message)
                    except Exception as e:
                        error_message = f"Error in broader project search: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)
            
            # If we still don't have metrics, return a detailed error
            if metrics is None:
                error_detail = "Could not find performance metrics for model " + \
                              f"{model.name}-{model.version} in project {project_id}. " + \
                              "Attempted searches: " + "; ".join(error_messages)
                logger.error(error_detail)
                raise HTTPException(status_code=404, detail="Performance metrics not found")
            
            return {
                "project_id": project_id,
                "model_name": model.name,
                "model_version": model.version,
                "timestamp": metrics.get("timestamp", ""),
                "metrics": metrics
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_fairness_metrics(self, project_id: int, model_id: int, model_version: str) -> Dict[str, Any]:
        """Get fairness metrics for a model"""
        try:
            # Get model
            model = self._get_model(model_id, project_id)
            
            # Verify model version
            if model.version != model_version:
                raise ValidationError(f"Model version mismatch. Expected {model_version}, got {model.version}")
            
            metrics = None
            error_messages = []
            
            # Try to get metrics from GCS
            if settings.USE_GCS:
                logger.info(f"Looking for fairness metrics in GCS bucket for project {project_id}, model {model.name}-{model.version}")
                
                # First, try exact model name match with both possible directory names
                possible_metric_dirs = [
                    f"outputs/{project_id}/{model.name}-{model.version}/fairness_bias/metrics/",
                    f"outputs/{project_id}/{model.name}-{model.version}/fairness/metrics/"
                ]
                
                for primary_pattern in possible_metric_dirs:
                    if metrics:
                        break
                        
                    try:
                        logger.info(f"Searching with pattern: {primary_pattern}")
                        
                        # List blobs with this prefix
                        blobs = list(self.storage.bucket.list_blobs(prefix=primary_pattern))
                        logger.info(f"Found {len(blobs)} blobs with pattern")
                        
                        # Filter for JSON files
                        json_blobs = [b for b in blobs if b.name.endswith('.json')]
                        if json_blobs:
                            # Get the latest blob based on update time
                            latest_blob = max(json_blobs, key=lambda b: b.updated)
                            logger.info(f"Found fairness metrics file: {latest_blob.name}")
                            
                            # Download and parse the metrics
                            metrics = json.loads(latest_blob.download_as_string())
                            logger.info(f"Successfully retrieved fairness metrics from: {latest_blob.name}")
                        else:
                            error_message = f"No JSON files found in path: {primary_pattern}"
                            logger.warning(error_message)
                            error_messages.append(error_message)
                    except Exception as e:
                        error_message = f"Error accessing path {primary_pattern}: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)
                
                # If primary patterns didn't work, try a broader search for directories with the version
                if metrics is None:
                    try:
                        logger.info(f"Performing broader search by version in project {project_id}")
                        prefix = f"outputs/{project_id}/"
                        
                        # List all blobs in project folder
                        all_blobs = list(self.storage.bucket.list_blobs(prefix=prefix))
                        logger.info(f"Found {len(all_blobs)} total blobs in project folder")
                        
                        # Find all directories that contain the model version
                        version_dirs = set()
                        for blob in all_blobs:
                            # Extract potential directory pattern that includes the version
                            path_parts = blob.name.split('/')
                            for i, part in enumerate(path_parts):
                                if model.version in part and i > 1:  # Skip 'outputs' and project_id parts
                                    # Get the directory name containing the version
                                    version_dir = '/'.join(path_parts[:i+1])
                                    version_dirs.add(version_dir)
                        
                        logger.info(f"Found {len(version_dirs)} directories containing version {model.version}: {version_dirs}")
                        
                        # Check each potential directory for fairness metrics
                        for version_dir in version_dirs:
                            if metrics:
                                break
                                
                            for metric_subdir in ['fairness_bias/metrics/', 'fairness/metrics/']:
                                try:
                                    metrics_path = f"{version_dir}/{metric_subdir}"
                                    logger.info(f"Checking for metrics in: {metrics_path}")
                                    
                                    metrics_blobs = [
                                        b for b in all_blobs 
                                        if b.name.startswith(metrics_path) and b.name.endswith('.json')
                                    ]
                                    
                                    if metrics_blobs:
                                        latest_blob = max(metrics_blobs, key=lambda b: b.updated)
                                        logger.info(f"Found fairness metrics in directory: {latest_blob.name}")
                                        metrics = json.loads(latest_blob.download_as_string())
                                        break
                                except Exception as inner_e:
                                    logger.warning(f"Error checking directory {version_dir}/{metric_subdir}: {str(inner_e)}")
                                    continue
                        
                        # If still no metrics, look for any fairness metrics in the project
                        if metrics is None:
                            logger.info("Searching for any fairness metrics in the project")
                            fairness_metrics_blobs = [
                                b for b in all_blobs 
                                if b.name.endswith('.json') and 
                                ('/fairness_bias/metrics/' in b.name or '/fairness/metrics/' in b.name)
                            ]
                            
                            if fairness_metrics_blobs:
                                latest_blob = max(fairness_metrics_blobs, key=lambda b: b.updated)
                                logger.info(f"Found general fairness metrics: {latest_blob.name}")
                                metrics = json.loads(latest_blob.download_as_string())
                            else:
                                error_message = "No fairness metrics JSON files found in project"
                                logger.warning(error_message)
                                error_messages.append(error_message)
                    except Exception as e:
                        error_message = f"Error in broader project search: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)
            
            # If we still don't have metrics, return a detailed error
            if metrics is None:
                error_detail = "Could not find fairness metrics for model " + \
                              f"{model.name}-{model.version} in project {project_id}. " + \
                              "Attempted searches: " + "; ".join(error_messages)
                logger.error(error_detail)
                raise HTTPException(status_code=404, detail="Fairness metrics not found")
            
            return {
                "project_id": project_id,
                "model_name": model.name,
                "model_version": model.version,
                "timestamp": metrics.get("timestamp", ""),
                "metrics": metrics
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting fairness metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_explainability_metrics(self, project_id: int, model_id: int, model_version: str) -> Dict[str, Any]:
        """Get explainability metrics for a model"""
        try:
            # Get model
            model = self._get_model(model_id, project_id)
            
            # Verify model version
            if model.version != model_version:
                raise ValidationError(f"Model version mismatch. Expected {model_version}, got {model.version}")
            
            metrics = None
            error_messages = []
            
            # Try to get metrics from GCS
            if settings.USE_GCS:
                logger.info(f"Looking for explainability metrics in GCS bucket for project {project_id}, model {model.name}-{model.version}")
                
                # First, try exact model name match
                try:
                    primary_pattern = f"outputs/{project_id}/{model.name}-{model.version}/explainability/metrics/"
                    logger.info(f"Searching with primary pattern: {primary_pattern}")
                    
                    # List blobs with this prefix
                    blobs = list(self.storage.bucket.list_blobs(prefix=primary_pattern))
                    logger.info(f"Found {len(blobs)} blobs with primary pattern")
                    
                    # Filter for JSON files
                    json_blobs = [b for b in blobs if b.name.endswith('.json')]
                    if json_blobs:
                        # Get the latest blob based on update time
                        latest_blob = max(json_blobs, key=lambda b: b.updated)
                        logger.info(f"Found explainability metrics file: {latest_blob.name}")
                        
                        # Download and parse the metrics
                        metrics = json.loads(latest_blob.download_as_string())
                        logger.info(f"Successfully retrieved explainability metrics from: {latest_blob.name}")
                    else:
                        error_message = f"No JSON files found in primary path"
                        logger.warning(error_message)
                        error_messages.append(error_message)
                except Exception as e:
                    error_message = f"Error accessing primary path: {str(e)}"
                    logger.warning(error_message)
                    error_messages.append(error_message)
                
                # If primary pattern didn't work, try a broader search for directories with the version
                if metrics is None:
                    try:
                        logger.info(f"Performing broader search by version in project {project_id}")
                        prefix = f"outputs/{project_id}/"
                        
                        # List all blobs in project folder
                        all_blobs = list(self.storage.bucket.list_blobs(prefix=prefix))
                        logger.info(f"Found {len(all_blobs)} total blobs in project folder")
                        
                        # Find all directories that contain the model version
                        version_dirs = set()
                        for blob in all_blobs:
                            # Extract potential directory pattern that includes the version
                            path_parts = blob.name.split('/')
                            for i, part in enumerate(path_parts):
                                if model.version in part and i > 1:  # Skip 'outputs' and project_id parts
                                    # Get the directory name containing the version
                                    version_dir = '/'.join(path_parts[:i+1])
                                    version_dirs.add(version_dir)
                        
                        logger.info(f"Found {len(version_dirs)} directories containing version {model.version}: {version_dirs}")
                        
                        # Check each potential directory for explainability metrics
                        for version_dir in version_dirs:
                            try:
                                metrics_path = f"{version_dir}/explainability/metrics/"
                                logger.info(f"Checking for metrics in: {metrics_path}")
                                
                                metrics_blobs = [
                                    b for b in all_blobs 
                                    if b.name.startswith(metrics_path) and b.name.endswith('.json')
                                ]
                                
                                if metrics_blobs:
                                    latest_blob = max(metrics_blobs, key=lambda b: b.updated)
                                    logger.info(f"Found explainability metrics in version directory: {latest_blob.name}")
                                    metrics = json.loads(latest_blob.download_as_string())
                                    break
                            except Exception as inner_e:
                                logger.warning(f"Error checking directory {version_dir}: {str(inner_e)}")
                                continue
                        
                        # If still no metrics, look for any explainability metrics in the project
                        if metrics is None:
                            logger.info("Searching for any explainability metrics in the project")
                            explainability_metrics_blobs = [
                                b for b in all_blobs 
                                if b.name.endswith('.json') and '/explainability/metrics/' in b.name
                            ]
                            
                            if explainability_metrics_blobs:
                                latest_blob = max(explainability_metrics_blobs, key=lambda b: b.updated)
                                logger.info(f"Found general explainability metrics: {latest_blob.name}")
                                metrics = json.loads(latest_blob.download_as_string())
                            else:
                                error_message = "No explainability metrics JSON files found in project"
                                logger.warning(error_message)
                                error_messages.append(error_message)
                    except Exception as e:
                        error_message = f"Error in broader project search: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)
            
            # If we still don't have metrics, return a detailed error
            if metrics is None:
                error_detail = "Could not find explainability metrics for model " + \
                              f"{model.name}-{model.version} in project {project_id}. " + \
                              "Attempted searches: " + "; ".join(error_messages)
                logger.error(error_detail)
                raise HTTPException(status_code=404, detail="Explainability metrics not found")
            
            return {
                "project_id": project_id,
                "model_name": model.name,
                "model_version": model.version,
                "timestamp": metrics.get("timestamp", ""),
                "metrics": metrics
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting explainability metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_drift_metrics(self, project_id: int, model_id: int, model_version: str) -> Dict[str, Any]:
        """Get drift metrics for a model"""
        try:
            # Get model
            model = self._get_model(model_id, project_id)
            
            # Verify model version
            if model.version != model_version:
                raise ValidationError(f"Model version mismatch. Expected {model_version}, got {model.version}")
            
            metrics = None
            error_messages = []
            
            # Try to get metrics from GCS
            if settings.USE_GCS:
                logger.info(f"Looking for drift metrics in GCS bucket for project {project_id}, model {model.name}-{model.version}")
                
                # First, try exact model name match with both possible directory names
                possible_metric_dirs = [
                    f"outputs/{project_id}/{model.name}-{model.version}/drift_robustness/metrics/",
                    f"outputs/{project_id}/{model.name}-{model.version}/drift/metrics/"
                ]
                
                for primary_pattern in possible_metric_dirs:
                    if metrics:
                        break
                        
                    try:
                        logger.info(f"Searching with pattern: {primary_pattern}")
                        
                        # List blobs with this prefix
                        blobs = list(self.storage.bucket.list_blobs(prefix=primary_pattern))
                        logger.info(f"Found {len(blobs)} blobs with pattern")
                        
                        # Filter for JSON files
                        json_blobs = [b for b in blobs if b.name.endswith('.json')]
                        if json_blobs:
                            # Get the latest blob based on update time
                            latest_blob = max(json_blobs, key=lambda b: b.updated)
                            logger.info(f"Found drift metrics file: {latest_blob.name}")
                            
                            # Download and parse the metrics
                            metrics = json.loads(latest_blob.download_as_string())
                            logger.info(f"Successfully retrieved drift metrics from: {latest_blob.name}")
                        else:
                            error_message = f"No JSON files found in path: {primary_pattern}"
                            logger.warning(error_message)
                            error_messages.append(error_message)
                    except Exception as e:
                        error_message = f"Error accessing path {primary_pattern}: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)
                
                # If primary patterns didn't work, try a broader search for directories with the version
                if metrics is None:
                    try:
                        logger.info(f"Performing broader search by version in project {project_id}")
                        prefix = f"outputs/{project_id}/"
                        
                        # List all blobs in project folder
                        all_blobs = list(self.storage.bucket.list_blobs(prefix=prefix))
                        logger.info(f"Found {len(all_blobs)} total blobs in project folder")
                        
                        # Find all directories that contain the model version
                        version_dirs = set()
                        for blob in all_blobs:
                            # Extract potential directory pattern that includes the version
                            path_parts = blob.name.split('/')
                            for i, part in enumerate(path_parts):
                                if model.version in part and i > 1:  # Skip 'outputs' and project_id parts
                                    # Get the directory name containing the version
                                    version_dir = '/'.join(path_parts[:i+1])
                                    version_dirs.add(version_dir)
                        
                        logger.info(f"Found {len(version_dirs)} directories containing version {model.version}: {version_dirs}")
                        
                        # Check each potential directory for drift metrics
                        for version_dir in version_dirs:
                            if metrics:
                                break
                                
                            for metric_subdir in ['drift_robustness/metrics/', 'drift/metrics/']:
                                try:
                                    metrics_path = f"{version_dir}/{metric_subdir}"
                                    logger.info(f"Checking for metrics in: {metrics_path}")
                                    
                                    metrics_blobs = [
                                        b for b in all_blobs 
                                        if b.name.startswith(metrics_path) and b.name.endswith('.json')
                                    ]
                                    
                                    if metrics_blobs:
                                        latest_blob = max(metrics_blobs, key=lambda b: b.updated)
                                        logger.info(f"Found drift metrics in directory: {latest_blob.name}")
                                        metrics = json.loads(latest_blob.download_as_string())
                                        break
                                except Exception as inner_e:
                                    logger.warning(f"Error checking directory {version_dir}/{metric_subdir}: {str(inner_e)}")
                                    continue
                        
                        # If still no metrics, look for any drift metrics in the project
                        if metrics is None:
                            logger.info("Searching for any drift metrics in the project")
                            drift_metrics_blobs = [
                                b for b in all_blobs 
                                if b.name.endswith('.json') and 
                                ('/drift_robustness/metrics/' in b.name or '/drift/metrics/' in b.name)
                            ]
                            
                            if drift_metrics_blobs:
                                latest_blob = max(drift_metrics_blobs, key=lambda b: b.updated)
                                logger.info(f"Found general drift metrics: {latest_blob.name}")
                                metrics = json.loads(latest_blob.download_as_string())
                            else:
                                error_message = "No drift metrics JSON files found in project"
                                logger.warning(error_message)
                                error_messages.append(error_message)
                    except Exception as e:
                        error_message = f"Error in broader project search: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)
            
            # If we still don't have metrics, return a detailed error
            if metrics is None:
                error_detail = "Could not find drift metrics for model " + \
                              f"{model.name}-{model.version} in project {project_id}. " + \
                              "Attempted searches: " + "; ".join(error_messages)
                logger.error(error_detail)
                raise HTTPException(status_code=404, detail="Drift metrics not found")
            
            return {
                "project_id": project_id,
                "model_name": model.name,
                "model_version": model.version,
                "timestamp": metrics.get("timestamp", ""),
                "metrics": metrics
            }
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting drift metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def get_pdf_file_path(self, project_id: int, model_id: int, model_version: str) -> Optional[str]:
        """Get the path to the latest PDF report for a model"""
        logger.info(f"Getting PDF file path for {model_id}-{model_version}")
        model = self._get_model(model_id, project_id)
        if model.version != model_version:
            raise ValidationError(f"Model version mismatch. Expected {model_version}, got {model.version}")
        
        logger.info(f"Getting PDF file path for {model.name}-{model.version}")
        
        if settings.USE_GCS:
            # For GCS storage
            try:
                # Construct base path for the model in GCS
                base_path = f"{self.outputs_dir}/{project_id}/{model.name}-{model.version}"
                logger.info(f"Looking for PDFs in GCS path: {base_path}")
                
                # List all blobs that match the pattern and end with .pdf
                pdf_blobs = list(self.storage.bucket.list_blobs(prefix=base_path))
                pdf_blobs = [blob for blob in pdf_blobs if blob.name.lower().endswith('.pdf')]
                
                if not pdf_blobs:
                    logger.warning(f"No PDF files found in GCS path: {base_path}")
                    # Try to list all PDFs in the project directory as a fallback
                    project_path = f"{self.outputs_dir}/{project_id}"
                    pdf_blobs = list(self.storage.bucket.list_blobs(prefix=project_path))
                    pdf_blobs = [blob for blob in pdf_blobs if blob.name.lower().endswith('.pdf')]
                    
                    if not pdf_blobs:
                        logger.warning(f"No PDF files found in project GCS path: {project_path}")
                        return None
                
                # Sort by last modified time (newest first)
                pdf_blobs.sort(key=lambda blob: blob.updated, reverse=True)
                
                # Return the full GCS path to the newest PDF
                newest_pdf = pdf_blobs[0].name
                logger.info(f"Found newest PDF in GCS: {newest_pdf}")
                return newest_pdf
                
            except Exception as e:
                logger.error(f"Error listing PDF files from GCS: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error accessing PDF reports: {str(e)}")
        else:
            # For local storage
            try:
                metrics_dir = os.path.join(
                    settings.OUTPUT_DIR,
                    str(project_id),
                    f"{model.name}-{model.version}"
                )
                logger.info(f"Looking for PDFs in local path: {metrics_dir}")
                
                if not os.path.exists(metrics_dir):
                    logger.warning(f"Directory does not exist: {metrics_dir}")
                    return None
                
                pdf_files = [
                    os.path.join(metrics_dir, f)
                    for f in os.listdir(metrics_dir)
                    if f.lower().endswith(".pdf")
                ]
                logger.info(f"PDF files found: {pdf_files}")
                
                if not pdf_files:
                    return None

                # Sort by creation/modification time (latest first)
                pdf_files.sort(key=os.path.getctime, reverse=True)
                return pdf_files[0]
            except Exception as e:
                logger.error(f"Error listing PDF files from local storage: {str(e)}")
                return None

    def _get_file_size(self, file_path: str) -> int:
        """Get file size for local or GCS paths"""
        if settings.USE_GCS and file_path.startswith("gs://"):
            # Extract the path from the GCS URL
            parts = file_path.replace("gs://", "").split("/")
            bucket_name = parts[0]
            blob_path = '/'.join(parts[1:])
            
            logger.info(f"Getting file size for GCS path: {file_path}, blob path: {blob_path}")
            
            # Get the blob
            blob = self.storage.bucket.blob(blob_path)
            
            # Handle case where file might be in either old or new path format
            if not blob.exists() and (blob_path.startswith("models/") or blob_path.startswith("datasets/")):
                # Try with the "uploads/" prefix for the new path structure
                if not blob_path.startswith("uploads/"):
                    new_blob_path = f"uploads/{blob_path}"
                    new_blob = self.storage.bucket.blob(new_blob_path)
                    if new_blob.exists():
                        blob = new_blob
            
            # Get size
            try:
                blob.reload()  # Refresh metadata
                return blob.size
            except Exception as e:
                logger.error(f"Error getting file size from GCS: {str(e)}")
                return 0
        else:
            return os.path.getsize(file_path)

    def save_visualization(self, fig, filepath):
        """Save visualization to file"""
        # Convert figure to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        img_data = buf.getvalue()
        
        if settings.USE_GCS:
            # For GCS, convert filepath to proper format for storage client
            try:
                # Handle different path formats to ensure proper parsing
                if isinstance(filepath, dict):
                    logger.warning(f"Received dict instead of string path: {filepath}")
                    # Try to extract path from dict if possible
                    if 'plots' in filepath:
                        path_str = str(filepath['plots'])
                    else:
                        # Fallback to a safe default if needed
                        path_str = f"reports/plots/plot_{uuid.uuid4()}.png"
                else:
                    path_str = filepath
                
                # Extract path components
                if '/' in path_str:
                    path_parts = path_str.split('/')
                else:
                    # Fallback for unexpected path format
                    path_parts = ["reports", "plots", f"plot_{uuid.uuid4()}.png"]
                
                # Save to GCS
                gcs_path = self.storage.save_plot(img_data, *path_parts)
                logger.info(f"Saved visualization to GCS path: {gcs_path}")
                return gcs_path
            except Exception as e:
                logger.error(f"Error saving visualization to GCS: {str(e)}")
                # Create a temp local file as fallback
                temp_path = os.path.join(tempfile.gettempdir(), f"temp_plot_{uuid.uuid4()}.png")
                with open(temp_path, 'wb') as f:
                    f.write(img_data)
                return temp_path
        else:
            # For local storage
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                # Save to file
                with open(filepath, 'wb') as f:
                    f.write(img_data)
                return filepath
            except Exception as e:
                logger.error(f"Error saving visualization to local file: {str(e)}")
                # Create a temp file as fallback
                temp_path = os.path.join(tempfile.gettempdir(), f"temp_plot_{uuid.uuid4()}.png")
                with open(temp_path, 'wb') as f:
                    f.write(img_data)
                return temp_path

    async def _update_audit_status(self, audit_id: int, status: str, metrics_path: str = None) -> None:
        """Update the status of an audit in the database"""
        logger.info(f"Updating audit {audit_id} status to {status}")
        try:
            update_data = {
                "status": status,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # If metrics path is provided, include it in the results
            if metrics_path:
                # First get current results if any
                response = self.db.table('audits').select('results').eq('id', audit_id).execute()
                if response.data and len(response.data) > 0:
                    current_results = response.data[0].get('results', {}) or {}
                    # Add or update metrics_path in results
                    if isinstance(current_results, dict):
                        current_results['metrics_path'] = metrics_path
                        update_data['results'] = current_results
            
            # Update the audit record
            self.db.table('audits').update(update_data).eq('id', audit_id).execute()
            logger.info(f"Successfully updated audit {audit_id} status")
            
        except Exception as e:
            logger.error(f"Error updating audit status: {str(e)}")
            raise