# AI Governance API Documentation

## Overview
This API provides endpoints for managing AI model governance, including auditing, benchmarking, and monitoring capabilities.

## Base URL
`http://localhost:8000`

## Authentication
All endpoints require authentication. Include the JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

## Endpoints

### Projects
- `POST /projects/` - Create a new project
- `GET /projects/` - List all projects
- `GET /projects/{project_id}` - Get project details
- `PUT /projects/{project_id}` - Update project
- `DELETE /projects/{project_id}` - Delete project

### Models
- `POST /models/` - Create a new model entry
- `POST /models/{model_id}/upload` - Upload model file
- `GET /models/` - List all models
- `GET /models/{model_id}` - Get model details
- `PUT /models/{model_id}` - Update model
- `DELETE /models/{model_id}` - Delete model
- `POST /models/{model_id}/audit` - Run model audit
- `GET /models/{model_id}/audits` - List model audits
- `GET /models/{model_id}/audits/{audit_id}` - Get audit results

### Datasets
- `POST /datasets/` - Create a new dataset entry
- `POST /datasets/{dataset_id}/upload` - Upload dataset file
- `GET /datasets/` - List all datasets
- `GET /datasets/{dataset_id}` - Get dataset details
- `PUT /datasets/{dataset_id}` - Update dataset
- `DELETE /datasets/{dataset_id}` - Delete dataset
- `GET /datasets/{dataset_id}/stats` - Get dataset statistics
- `POST /datasets/{dataset_id}/validate` - Validate dataset

### Audits
- `POST /audits/` - Create a new audit
- `GET /audits/` - List all audits
- `GET /audits/{audit_id}` - Get audit details
- `PUT /audits/{audit_id}` - Update audit
- `DELETE /audits/{audit_id}` - Delete audit
- `GET /audits/{audit_id}/results` - Get audit results
- `GET /audits/{audit_id}/report` - Generate audit report

### Reports
- `POST /reports/` - Create a new report
- `GET /reports/` - List all reports
- `GET /reports/{report_id}` - Get report details
- `PUT /reports/{report_id}` - Update report
- `DELETE /reports/{report_id}` - Delete report
- `GET /reports/{report_id}/download` - Download report file



## Error Handling
The API uses standard HTTP status codes:
- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error

Error responses include:
```json
{
    "detail": "Error message"
}
```

## Rate Limiting
API endpoints are rate-limited to:
- 100 requests per minute for authenticated users
- 10 requests per minute for unauthenticated users 