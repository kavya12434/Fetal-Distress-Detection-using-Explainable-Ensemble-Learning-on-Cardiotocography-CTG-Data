# Multi-stage: build the Vite bundle with Node, serve the static output with nginx.
# Build context is frontend/.

# ---- stage 1: build ----
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Empty API base makes the client issue same-origin relative requests
# (/api/...), which nginx proxies to the backend service. Without this the
# bundle would hardcode http://localhost:8000 from api.ts's fallback.
ARG VITE_API_URL=""
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# ---- stage 2: serve ----
FROM nginx:alpine AS serve

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Match the dev-server port so the URL is the same in both workflows.
EXPOSE 5173

CMD ["nginx", "-g", "daemon off;"]
