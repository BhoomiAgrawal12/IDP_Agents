const backendUrl = "https://d0e9-34-145-154-126.ngrok-free.app/"; 

async function getPrediction(data) {
  const response = await fetch(`${backendUrl}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  });
  const result = await response.json();
  console.log(result);
}
