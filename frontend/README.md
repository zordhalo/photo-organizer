# Construction Photo Analyzer - Frontend

AI-powered construction photo analysis and organization frontend built with Vite + Vanilla JavaScript.

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ 
- npm 9+
- Backend server running on `http://localhost:8000`

### Installation

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The development server will start at `http://localhost:5173`

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/          # UI components
│   ├── services/            # API clients, analysis service
│   ├── utils/               # Helper functions
│   ├── config/              # Configuration management
│   │   ├── api.config.js    # API endpoint configuration
│   │   ├── app.config.js    # Application settings
│   │   └── index.js         # Config exports
│   ├── styles/              # Global styles
│   │   └── main.css         # Main stylesheet
│   └── main.js              # Entry point
├── public/                  # Static assets
├── tests/                   # Test files
├── .env.development         # Development environment variables
├── .env.production          # Production environment variables
└── vite.config.js           # Vite configuration
```

## ⚙️ Configuration

### Environment Variables

Create a `.env.local` file for local overrides (not committed to git):

| Variable | Description | Default |
|----------|-------------|---------|
| `VITE_API_URL` | Backend API base URL | `http://localhost:8000` |
| `VITE_MAX_FILE_SIZE` | Max upload file size (bytes) | `10485760` (10MB) |
| `VITE_ENABLE_FALLBACK` | Enable offline fallback mode | `true` |

### API Configuration

The API configuration is managed in `src/config/api.config.js`:

```javascript
import { API_CONFIG, getEndpointUrl } from './config';

// Access base URL
console.log(API_CONFIG.BASE_URL); // http://localhost:8000

// Get full endpoint URL
const analyzeUrl = getEndpointUrl('ANALYZE'); // http://localhost:8000/analyze
```

### Available Endpoints

| Endpoint | Path | Description |
|----------|------|-------------|
| `ANALYZE` | `/analyze` | Single image analysis |
| `ANALYZE_BATCH` | `/analyze-batch` | Batch image analysis |
| `HEALTH` | `/health` | API health check |
| `STATS` | `/stats` | Server statistics |

## 🛠️ Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server with HMR |
| `npm run build` | Build for production |
| `npm run preview` | Preview production build |
| `npm test` | Run tests in watch mode |
| `npm run test:run` | Run tests once |
| `npm run test:coverage` | Run tests with coverage |

## 🧪 Testing

Tests are located in the `tests/` directory and use Vitest.

```bash
# Run tests in watch mode
npm test

# Run tests once
npm run test:run

# Run with coverage
npm run test:coverage
```

## 🔧 Development

### Adding a New Component

1. Create component file in `src/components/`
2. Export from component's index file
3. Import and use in your code

### Adding a New Service

1. Create service file in `src/services/`
2. Import configuration from `@config`
3. Export service functions

### Path Aliases

The following aliases are configured:

- `@` → `/src`
- `@components` → `/src/components`
- `@services` → `/src/services`
- `@utils` → `/src/utils`
- `@config` → `/src/config`

## 🏗️ Building for Production

```bash
# Build the project
npm run build

# Preview the build
npm run preview
```

The build output will be in the `dist/` directory.

## 🔗 Integration with Backend

The frontend is designed to work with the Python FastAPI backend:

1. Ensure backend is running: `cd backend && uvicorn app.main:app --reload`
2. Start frontend: `npm run dev`
3. The frontend will automatically connect to `http://localhost:8000`

### CORS Configuration

The backend should allow requests from `http://localhost:5173` during development.

## 📝 Phase 1 Checklist

- [x] Project initialized with Vite + Vanilla JS
- [x] Directory structure created
- [x] Configuration management setup
- [x] Environment variables configured
- [x] Core dependencies installed (axios, vitest)
- [x] Development server runs successfully
- [x] README with setup instructions

## 🚀 Next Steps (Phase 2)

- [ ] Implement API client service
- [ ] Add image upload functionality
- [ ] Create analysis result components
- [ ] Implement error handling

## 📄 License

MIT
