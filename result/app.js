const express = require('express');
const { Pool } = require('pg');
const Redis = require('ioredis');

const app = express();
const pgPool = new Pool({ connectionString: 'postgresql://user:password@localhost/netflix_recommender' });
const redis = new Redis();

app.get('/recommendations/:userId', async (req, res) => {
    const { userId } = req.params;
    // Verificar si la recomendación está en Redis
    // Si no, usar .NET para calcular y almacenar el resultado en Redis
    // Responder con las recomendaciones
});

app.listen(3000, () => console.log('Server running on port 3000'));