$(document).ready(function () {
    // Initialize speech synthesis API
    const synth = window.speechSynthesis;

    // Toggle Dark Mode
    $('#toggle-dark-mode').click(function () {
        $('body').toggleClass('dark-mode');
    });

    // Predefined question click event
    $('.predefined-questions li').click(function () {
        var question = $(this).text(); // Get the clicked question's text
        $('#user_input').val(question); // Set the input field value to the clicked question
    });

    // Handle form submission for asking questions
    $('#question-form').submit(function (event) {
        event.preventDefault(); // Prevent the default form submission

        // Get user input and validate it
        var user_input = $('#user_input').val().trim();
        if (!user_input) {
            alert('Please enter a question!');
            return;
        }

        // Clear input field
        $('#user_input').val('');

        // Append user's question to the chat body
        $('#response-container').append(`
            <div class="chat-message user-message">
                <div class="message-box">${user_input}</div>
            </div>
        `);

        // Send the question to the backend using AJAX
        $.ajax({
            url: '/ask',
            type: 'POST',
            data: { user_input: user_input },
            success: function (response) {
                if (response && response.length > 0) {
                    response.forEach(function (item) {
                        var question = item.question;
                        var summary = item.summary;

                        // Append bot's response to the chat body
                        $('#response-container').append(`
                            <div class="chat-message bot-message">
                                <div class="message-box">${summary}</div>
                                <!-- Listen to Answer Button -->
                                
                                <!-- Feedback Section -->
                                <div class="feedback-section mt-3">
                                    
                                        <label for="feedback_type">Feedback Type:</label>
                                        <select class="feedback-type" data-question="${question}" data-answer="${summary}">
                                            <option value="positive">Positive</option>
                                            <option value="negative">Negative</option>
                                        </select>
                                    
                                    
                                    <label for="helpfulness_rating">Helpfulness Rating (1-5):</label>
                                    <input type="number" class="feedback-helpfulness" min="1" max="5" data-question="${question}" data-answer="${summary}" value="5">
                                    
                                    <input type="text" class="feedback-comment" size="110" placeholder="Drop a Feedback..." data-question="${ question }" data-answer="${ summary }">
                                    <button class="submit-feedback-button btn btn-success mt-3">Submit</button>
                                    <button class="speak-button btn btn-primary mt-2" data-answer="${summary}">
                                        Listen
                                    </button>
                                </div>
                            </div>
                        `);
                    });
                } else {
                    $('#response-container').append(`
                        <div class="chat-message bot-message">
                            <div class="message-box" style="background: #f8d7da; color: #721c24;">
                                No relevant answers found. Please try again with a different question.
                            </div>
                        </div>
                    `);
                }

                // Scroll to the bottom of the chat body
                $('#response-container').scrollTop($('#response-container')[0].scrollHeight);
            },
            error: function () {
                $('#response-container').append(`
                    <div class="chat-message bot-message">
                        <div class="message-box" style="background: #f8d7da; color: #721c24;">
                            An error occurred while processing your request. Please try again later.
                        </div>
                    </div>
                `);
            }
        });
    });

    // Text-to-Speech Button Click Event
    $(document).on('click', '.speak-button', function () {
        var answer = $(this).data('answer');

        // Stop any ongoing speech synthesis
        if (synth.speaking) {
            synth.cancel();
        }

        // Check if the browser supports speech synthesis
        if ('speechSynthesis' in window) {
            const maxChunkLength = 200; // Set a limit for each chunk
            const chunks = [];

            // Split the response into manageable chunks
            while (answer.length > 0) {
                let chunk = answer.substring(0, maxChunkLength);
                let lastSpace = chunk.lastIndexOf(' ');
                if (lastSpace > -1 && answer.length > maxChunkLength) {
                    chunk = chunk.substring(0, lastSpace);
                }
                chunks.push(chunk);
                answer = answer.substring(chunk.length).trim();
            }

            // Speak each chunk sequentially
            const speakChunks = (index = 0) => {
                if (index < chunks.length) {
                    const speech = new SpeechSynthesisUtterance(chunks[index]);
                    speech.onend = () => speakChunks(index + 1);
                    speech.onerror = () => console.error('Speech synthesis error');
                    synth.speak(speech);
                }
            };

            speakChunks();
        } else {
            alert('Your browser does not support text-to-speech functionality.');
        }
    });

    // Submit Individual Feedback
    $(document).on('click', '.submit-feedback-button', function () {
        const feedbackSection = $(this).closest('.feedback-section');
        const question = feedbackSection.find('.feedback-type').data('question');
        const answer = feedbackSection.find('.feedback-type').data('answer');
        const feedbackType = feedbackSection.find('.feedback-type').val();
        const helpfulnessRating = feedbackSection.find('.feedback-helpfulness').val();
        const feedbackComment = feedbackSection.find('.feedback-comment').val();

        const feedbackData = {
            question: question,
            answer: answer,
            feedback_type: feedbackType,
            helpfulness_rating: parseInt(helpfulnessRating), // Ensure it's a number
            feedback_comment: feedbackComment
        };

        console.log('Submitting Feedback:', feedbackData); // Log for debugging

        // Send feedback data to the backend
        $.ajax({
            url: '/submit_feedback',
            type: 'POST',
            contentType: 'application/x-www-form-urlencoded',
            data: $.param(feedbackData), // Correctly format as JSON
            success: function (response) {
                if (response.success) {
                    alert('Feedback submitted successfully!');
                } else {
                    alert('Error submitting feedback: ' + response.message);
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                console.error('Error:', textStatus, errorThrown); // Log the error
                alert('Error submitting feedback. Please try again later.');
            }
        });
    });

    // Function to generate and download the chat as a PDF
    function downloadChatAsPDF() {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        let yOffset = 10; // Starting Y position for text in the PDF

        // Loop through chat messages and add them to the PDF
        $('#response-container').find('.chat-message').each(function (index, message) {
            // Check if it's a bot message
            const isBotMessage = $(message).hasClass('bot-message');
            const messageText = $(message).find('.message-box').text().trim();

            // Format the text for the PDF
            const prefix = isBotMessage ? 'Bot: ' : 'User: ';
            const text = prefix + messageText;

            // Add the text to the PDF, adjusting for line spacing
            const lineHeight = 10;
            const lines = doc.splitTextToSize(text, 190); // Wrap text for a max width of 190mm
            doc.text(lines, 10, yOffset);

            // Increment Y position
            yOffset += lines.length * lineHeight;

            // Check for page overflow and add a new page if necessary
            if (yOffset > 280) { // Close to the page bottom
                doc.addPage();
                yOffset = 10; // Reset Y offset for the new page
            }
        });

        // Save the PDF
        doc.save('chat.pdf');
    }

    // Attach the download function to the button click event
    $('#download-pdf').click(function () {
        downloadChatAsPDF();
    });
});
