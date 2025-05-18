const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");

const app = express();
const PORT = 4000;

// Debugging log to confirm the server script is running
console.log("Starting the server...");

// Enable CORS and serve static files
app.use(cors());
app.use(express.static(path.join(__dirname, "../login"))); // Serve static files from the login folder
console.log("Static files are being served from:", path.join(__dirname, "../login"));

// Configure Multer for file uploads
const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    cb(null, file.originalname);
  },
});
const upload = multer({ storage });

// Root route to serve login.html as the default page
app.get("/", (req, res) => {
  console.log("Serving login.html");
  res.sendFile(path.join(__dirname, "../login/login.html"));
});

// Route to serve file.html for uploading images
app.get("/upload", (req, res) => {
  console.log("Serving file.html");
  res.sendFile(path.join(__dirname, "../login/file.html"));
});

// Route to serve ai.html for showing results
app.get("/result", (req, res) => {
  console.log("Serving ai.html");
  res.sendFile(path.join(__dirname, "../login/ai.html"));
});

// Route to handle image analysis
app.post("/analyze", upload.single("file"), (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: "No file uploaded" });
  }

  console.log("Received a file for analysis:", req.file.path);

  // Placeholder logic for ML analysis (replace with actual logic)
  const metadata = { camera: "iPhone 13 maybe", fakeDetected: Math.random() > 0.5 };
  const fft = "FFT analysis: Image looks suspicious 🤨";

  console.log("Analysis result:", { metadata, fft });

  // Send the result back as JSON
  res.json({ metadata, fft });
});

// Start the server
app.listen(PORT, () => {
  console.log(`🚀 Backend server running at http://localhost:${PORT}`);
});
