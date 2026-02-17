const express = require("express");
const app = express();

app.use(express.raw({type: "*/*"}));

app.get("/", (req, res) => res.send("Welcome to API"));
app.get("/health", (req, res) => res.send("ok"));
app.post("/echo", (req, res) => res.send(req.body));
app.use((req, res) => res.status(404).send("not found"));

app.listen(8080);
