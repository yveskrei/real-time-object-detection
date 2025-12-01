FROM registry.access.redhat.com/ubi9/nodejs-22:9.7
WORKDIR /app
USER root
COPY . /app

# Install dependencies
RUN npm install

# Build the Vite app
RUN npm run build

# Install serve globally
RUN npm i -g serve

EXPOSE 3000

# Serve the built dist folder
CMD ["serve", "-s", "dist", "-l", "3000"]