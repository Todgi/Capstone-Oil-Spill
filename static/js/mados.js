async function predictMados(opticalFile, sarFile) {
    const formData = new FormData();
    formData.append('optical', opticalFile);
    formData.append('sar', sarFile);

    try {
        const response = await fetch('/predict_mados', {
            method: 'POST',
            body: formData
        });
        const result = await response.json();
        
        if (result.success) {
            // Tampilkan hasil prediksi pada peta
            displayPredictions(result.predictions);
        } else {
            console.error('Prediction failed:', result.error);
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

function displayPredictions(predictions) {
    // Implementasi visualisasi hasil prediksi pada peta
    // Sesuaikan dengan kebutuhan visualisasi Anda
}