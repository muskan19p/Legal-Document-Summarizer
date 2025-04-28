// Handle file drop / click
const dropArea = document.getElementById('drop-area');
const fileInput = document.getElementById('fileElem');
const cancelBtn = document.getElementById('cancelBtn');
const summarizeBtn = document.getElementById('summarizeBtn');
const fileNameDisplay = document.getElementById('file-name');

// Open file dialog on click
dropArea.addEventListener('click', () => {
    fileInput.click();
});

// Handle file input change
fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) {
        alert(`File selected: ${file.name}`);
        // You can now upload the file to server if needed
    }
});

// Handle drag and drop
dropArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropArea.style.borderColor = '#000'; // Highlight border on drag
});

dropArea.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropArea.style.borderColor = '#ccc'; // Reset border
});

dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    dropArea.style.borderColor = '#ccc';
    const file = e.dataTransfer.files[0];
    if (file) {
        fileInput.files = e.dataTransfer.files; // sync dropped file with input
        alert(`File dropped: ${file.name}`);
    }
});

// Summarize Button
  document.querySelector('.btn-summarize')?.addEventListener('click', () => {
    alert('Summarizing document...')
    // Add logic to send file for summarization
    document.querySelector('.summary-box').innerText = "This is a sample summary of the uploaded document.";
  });

  // Cancel Button
  document.querySelector('.btn-cancel')?.addEventListener('click', () => {
    alert('Action cancelled.');
    document.querySelector('.summary-box').innerText = "";
  });


// Cancel Feedback Button
document.querySelector('.btn-cancel-feedback')?.addEventListener('click', () => {
    alert('Feedback cleared.');
    document.querySelector('.feedback-box').value = ""; // Clear the textarea
});


  const phoneInput = document.querySelector('.phone');

phoneInput.addEventListener('input', function (e) {
    this.value = this.value.replace(/[^0-9\s\-()]/g, ''); 
});


  document.addEventListener("DOMContentLoaded", function () {
    // Submit rating
    const stars = document.querySelectorAll('.star');
    stars.forEach(star => {
        star.addEventListener('click', function () {
            const rating = this.getAttribute('data-value');
            fetch('/submit-rating/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ rating: rating }),
            })
            .then(response => response.json())
            .then(data => {
                console.log(data.message); // Show confirmation
            });
        });
    });

// Submit feedback
    const feedbackButton = document.querySelector('.btn-submit-feedback');
    feedbackButton.addEventListener('click', function () {
        const feedback = document.querySelector('.feedback-box').innerText;
        fetch('/submit-feedback/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ feedback: feedback }),
        })
        .then(response => response.json())
        .then(data => {
            console.log(data.message); // Show confirmation
        });
    });
});