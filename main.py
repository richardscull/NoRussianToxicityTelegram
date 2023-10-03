from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler
from transformers import BertTokenizer, BertForSequenceClassification

import os
import json
import torch
import torch.nn.functional as F
import random

load_dotenv()
token = os.getenv("TOKEN")
tokenizer = BertTokenizer.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')
model = BertForSequenceClassification.from_pretrained('SkolkovoInstitute/russian_toxicity_classifier')

def main():
    application = ApplicationBuilder().token(token).build()
    leaderboard_handler = CommandHandler('leaderboard', leaderboard)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), echoMessage)
    
    application.add_handler(message_handler)   
    application.add_handler(leaderboard_handler)
    print("Bot started!")
    application.run_polling()

async def leaderboard(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with open("database.json", "r") as database:
        data = json.load(database)
        messageToSend = ""

        for chat_id in data:
            if chat_id != str(update.effective_chat.id):
                continue

            messageToSend += f"⚠️ МЕТР ТОКСИЧНОСТИ\nТекущий чат: {update.effective_chat.title}\n\n👥 Пользователи: \n"
            
            for user_id in data[chat_id]:
                user = data[chat_id][user_id]
                username = await context.bot.get_chat_member(chat_id=chat_id, user_id=user_id)
                messageToSend += f"{username.user.first_name}: {round(user['toxicity'])} 🧪 \n"
               
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                                text=messageToSend,
                                                reply_to_message_id=update.message.message_id)

def updateUserToxicity(chat_id: str, user_id: str, toxicity: float):
    with open("database.json", "r+") as database:
            data = json.load(database)
            if str(chat_id) not in data:
                 data[str(chat_id)] = {}
        
            if str(user_id) not in data[str(chat_id)]:
                data[str(chat_id)][str(user_id)] = {}
                data[str(chat_id)][str(user_id)]["toxicity"] = 0

            userField = data[str(chat_id)][str(user_id)]
            newToxicity = userField["toxicity"] + toxicity
            userField["toxicity"] = newToxicity if newToxicity > 0 else 0
            database.seek(0)
            json.dump(data, database, indent=4)
            database.truncate()
            return userField["toxicity"]

def getRandomNegativeAnswer():
    with open("botanswers.json", "r", encoding='utf-8') as negative_answers:
        data = json.load(negative_answers)
        return data["negative"][random.randint(0, len(data["negative"]) - 1)] + "\n\n\t> Ваш текущий рейтинг: {}"

async def echoMessage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message is not None and update.edited_message is None:
        if update.message.from_user.is_bot:
            return
        if update.message.forward_from:
            return
        
    if update.edited_message is not None:
        if update.edited_message.from_user.is_bot:
            return
        if update.edited_message.forward_from:
            return
    
    if update.message is not None:
        user_id = update.message.from_user.id
        username = update.message.from_user.username
        message_id = update.message.message_id
        message_chat_id = update.message.chat_id
        message_text = update.message.text
    elif update.edited_message:
        user_id = update.edited_message.from_user.id
        username = update.edited_message.from_user.username
        message_id = update.edited_message.message_id
        message_chat_id = update.edited_message.chat_id
        message_text = update.edited_message.text
    else:
        return
    
    outputs = model(tokenizer.encode(message_text.lower(), return_tensors='pt'))
    logits = outputs.logits
    probabilities = F.softmax(logits, dim=1) 
    isMsgToxic = torch.argmax(probabilities, dim=1).item()
    toxicityFloat = ((probabilities[0][0].item() / 20 - probabilities[0][1].item()) * -100)
    userToxicityValue = updateUserToxicity(message_chat_id, user_id, toxicityFloat)

    if isMsgToxic == 1: 
        print(f"User {username} sent toxic message: {message_text}")
        value = round(userToxicityValue)
        botAnswer = getRandomNegativeAnswer().format(value)
        await context.bot.send_message(chat_id=update.effective_chat.id,
                                        text=botAnswer,
                                        reply_to_message_id=message_id)

if __name__ == "__main__":
    main()