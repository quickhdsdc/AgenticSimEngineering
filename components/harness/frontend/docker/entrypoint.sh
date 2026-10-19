#!/bin/sh
# Informational message
echo "========================================="
echo "Starting NGINX with base path: /"
echo "========================================="
echo

echo "nginx configuration:"
echo "----------------------------------------------------------------"
echo "etc/nginx/nginx.conf:"
echo "_____________________"
cat /etc/nginx/nginx.conf
echo "_____________________"
echo "/etc/nginx/conf.d/llmagent.conf:"
echo "_____________________"
cat /etc/nginx/conf.d/llmagent.conf
echo "_____________________"
echo "----------------------------------------------------------------"
echo

echo "Environment variables:"
echo "-------------------------------------------------------------------------------------------------------------------------"
printf "%-38s %s\n" "APP_VERSION:" "$APP_VERSION"
printf "%-38s %s\n" "BACKEND_URL:" "$BACKEND_URL"
printf "%-38s %s\n" "AAS_SERVER_URL:" "$AAS_SERVER_URL"
printf "%-38s %s\n" "AAS_UI_URL:" "$AAS_UI_URL"

# Replace the placeholders in all relevant files (.js, .html, .css)
find /usr/share/nginx/html -type f \( -name '*.js' -o -name '*.html' -o -name '*.css' \) -exec sed -i \
    -e "s|/__ENV_APP_VERSION__PLACEHOLDER__/|$APP_VERSION|g" \
    -e "s|/__ENV_BACKEND_URL__PLACEHOLDER__/|$BACKEND_URL|g" \
    -e "s|/__ENV_AAS_SERVER_URL__PLACEHOLDER__/|$AAS_SERVER_URL|g" \
    -e "s|/__ENV_AAS_UI_URL__PLACEHOLDER__/|$AAS_UI_URL|g" \
    {} \;

exec nginx -g "daemon off;"
