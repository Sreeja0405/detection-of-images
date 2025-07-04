const express = require("express");
const multer = require("multer");
const cors = require("cors");
const path = require("path");
const { exec } = require("child_process");

const app = express();
const PORT = 4000;

// Ensure working in current directory
process.chdir(__dirname);

// Enable CORS
app.use(cors());

// Serve frontend (analyze.html)
app.use(express.static(path.join(__dirname, "../login")));
console.log("✅ Serving static files from:", path.join(__dirname, "../login"));

// Multer setup
const storage = multer.diskStorage({
  destination: "uploads/",
  filename: (req, file, cb) => {
    cb(null, Date.now() + "_" + file.originalname);
  },
});
const upload = multer({ storage });

// Default route
app.get("/", (req, res) => {
  res.redirect("/analyze.html");
});

// Analyze image route
app.post("/analyze", upload.single("image"), (req, res) => {
  if (!req.file) {
    return res.status(400).send("No image uploaded");
  }

  const imagePath = path.join(__dirname, "uploads", req.file.filename);
  const pythonCommand = `python predict.py "${imagePath}"`;

  console.log("🧠 Running ML prediction for:", imagePath);

  exec(pythonCommand, (error, stdout, stderr) => {
    if (error) {
      console.error("❌ Python error:", error.message);
      return res.status(500).send("Prediction failed");
    }

    if (stderr) {
      console.warn("⚠️ Python stderr:", stderr);
    }

    const lines = stdout.trim().split("\n");
    const result = lines[lines.length - 1];

    // result format: AI,82.34,17.66
    const [prediction, aiConfidence, realConfidence] = result.split(",");

    console.log("✅ Prediction:", prediction);

    // Redirect with query parameters
    res.redirect(
      `/analyze.html?prediction=${prediction}&ai_confidence=${aiConfidence}&real_confidence=${realConfidence}`
    );
  });
});

// Start server
app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});
