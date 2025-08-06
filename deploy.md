# PolicyIntel API - Render Deployment Guide

## Prerequisites
1. GitHub account
2. Render account (free tier)
3. Your API keys ready (Cohere, Gemini)

## Step 1: Push to GitHub

1. Initialize git repository (if not already done):
```bash
git init
git add .
git commit -m "Initial commit - PolicyIntel API"
```

2. Create a new repository on GitHub and push:
```bash
git remote add origin https://github.com/YOUR_USERNAME/PolicyIntel.git
git branch -M main
git push -u origin main
```

## Step 2: Deploy on Render

1. Go to [render.com](https://render.com) and sign up/login
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file

## Step 3: Set Environment Variables

In Render dashboard, add these environment variables:
- `COHERE_API_KEY`: Your Cohere API key
- `GEMINI_API_KEY_1`: Your first Gemini API key
- `GEMINI_API_KEY_2`: Your second Gemini API key (optional)
- `GEMINI_API_KEY_3`: Your third Gemini API key (optional)

## Step 4: Deploy

1. Click "Create Web Service"
2. Wait for build to complete (10-15 minutes for first deployment)
3. Your API will be available at `https://your-app-name.onrender.com`

## Free Tier Limitations

⚠️ **Important Render Free Tier Limits:**
- **Sleep after 15 minutes of inactivity** - service will spin down
- **750 hours/month** - service will be suspended after this limit
- **Cold starts** - first request after sleep takes 30+ seconds
- **512 MB RAM** - your app is memory-optimized for this
- **Build timeout: 20 minutes**

## Optimizations for Free Tier

The app has been optimized for Render's free tier:
- Memory management with garbage collection
- Chunked processing for large documents
- Limited concurrent processing
- Health check endpoints
- OCR batch processing

## Testing Your Deployment

Once deployed, test the health endpoint:
```bash
curl https://your-app-name.onrender.com/health
```

Expected response:
```json
{
  "status": "healthy",
  "service": "PolicyIntel API",
  "version": "1.0.0"
}
```

## Monitoring

- Check Render dashboard for logs
- Monitor memory usage
- Watch for build failures

## Troubleshooting

1. **Build fails**: Check logs in Render dashboard
2. **OCR not working**: Tesseract installation might have failed
3. **Memory errors**: Reduce batch sizes in the code
4. **Timeout errors**: Large documents may exceed free tier limits

## Production Considerations

For production use, consider upgrading to Render's paid plans for:
- No sleep/auto-scaling
- More memory and CPU
- Longer build times
- Better reliability
