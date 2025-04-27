// Handle file drop / click
  const dropArea = document.querySelector('.drop-area');
  dropArea.addEventListener('click', () => {
    alert('Trigger file input here.')
    // Implement actual file input if needed
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

