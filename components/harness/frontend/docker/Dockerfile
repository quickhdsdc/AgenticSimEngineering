# Build stage
FROM node:lts as build-stage
# Set working directory
WORKDIR /app
# Copy the package.json
COPY package*.json ./
# Install npm dependencies
RUN npm install --loglevel "error"
# Copy project files
COPY .. .
# Build the application and remove node_modules
RUN npm run build --loglevel "error" && rm -rf node_modules

# Production Stage
FROM nginx:mainline-alpine-slim AS production-stage
# Define port 80 as exposed port
EXPOSE 80
# Define environment variables
ENV APP_VERSION="0.1.0" \
    BACKEND_URL="http://localhost:8000" \
    AAS_SERVER_URL="http://localhost:8081" \
    AAS_UI_URL="http://localhost:3000"
# Create a non-root user and group
RUN addgroup -g 1000 -S mygroup && \
    adduser -u 1000 -D -S -G mygroup myuser
# Install system dependencies
RUN apk add --no-cache curl
# Copy files from build stage to production stage
COPY --from=build-stage /app/dist /usr/share/nginx/html
# Copy the entrypoint script and make it executable
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN rm /etc/nginx/conf.d/*
RUN chmod +x /app/entrypoint.sh
# Replace the default Nginx configuration with a custom one
COPY docker/nginx/nginx.conf /etc/nginx/nginx.conf
COPY docker/nginx/llmagent.conf /etc/nginx/conf.d/llmagent.conf
# Change the ownership of the /app, /var/cache/nginx and /tmp directories to the non-root user
RUN chown -R myuser:mygroup /app /var/cache/nginx /tmp /etc/nginx /usr/share/nginx/html
# Give the myuser/mygroup/other read/write/execute permissions to the /app, /var/cache/nginx and /tmp directories
RUN chmod -R 777 /app /var/cache/nginx /tmp /etc/nginx
# Switch to the non-root user
USER myuser
# Define the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]