async function predictWithMADOS() {
    const rasterInput = document.getElementById('rasterInput');
    
    if (!rasterInput.files[0]) {
        alert('Please select a raster file');
        return;
    }

    const formData = new FormData();
    formData.append('file', rasterInput.files[0]);
    formData.append('image_path', 'dummy_path'); // MADOS expects this

    try {
        const response = await fetch('http://localhost:5000/predict_mados', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`MADOS service error: ${response.status}`);
        }

        const result = await response.json();
        console.log('MADOS prediction:', result);
        
        // Process MADOS results
        if (result.oil_coordinates && result.oil_coordinates.length > 0) {
            // Add oil spill markers to map
            result.oil_coordinates.forEach(([lat, lon]) => {
                L.marker([lat, lon]).addTo(window.map)
                    .bindPopup('Oil detected by MADOS');
            });
        }
        
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to process with MADOS: ' + error.message);
    }
} 