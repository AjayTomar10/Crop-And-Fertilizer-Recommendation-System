console.log("Working");
const chatBotContainer = document.querySelector('.chat-bot-container');
        const chatBotContent = document.querySelector('.chat-bot-content');
        const closeChatButton = document.getElementById('close-chat');
        const sendButton = document.getElementById('send-btn');
        const chatInput = document.getElementById('chat-input');
        const chatMessages = document.getElementById('chat-messages');
        const chatBotFarmer =document.getElementById('.chat-bot-farmer');

        chatBotContainer.addEventListener('click', () => {
            chatBotContent.style.display = 'flex';
            document.querySelector('.chat-bot-farmer').classList.remove('hidden');
        });

        closeChatButton.addEventListener('click', (event) => {
            event.stopPropagation();
            chatBotContent.style.display = 'none';
            chatBotContainer.style.transform = 'translateY(0)';
            document.querySelector('.chat-bot-farmer').classList.add('hidden');
        });

        sendButton.addEventListener('click', () => {
            sendMessage();
        });

        chatInput.addEventListener('keypress', (event) => {
            if (event.key === 'Enter') {
                sendMessage();
            }
        });

        function sendMessage() {
            const message = chatInput.value.trim();
            if (message) {
                const messageElement = document.createElement('div');
                messageElement.classList.add('chat-bot-message');
                messageElement.textContent = message;
                chatMessages.appendChild(messageElement);
                chatInput.value = '';
                chatMessages.scrollTop = chatMessages.scrollHeight;

                // Simulate bot response
                setTimeout(() => {
                    const botMessage = document.createElement('div');
                    botMessage.classList.add('chat-bot-message');
                    botMessage.textContent = `Bot: This is a response to "${message}".`;
                    chatMessages.appendChild(botMessage);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }, 1000);
            }
        }