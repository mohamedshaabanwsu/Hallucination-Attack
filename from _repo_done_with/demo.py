from utils import load_model_and_tokenizer, complete_input, MLP
from config import ModelConfig
import torch

model = 'MIND_model' # [vicuna, llama2, baichuan, internlm, chatglm, ziya]
device = 'cuda:0'
path='/scratch/user/gabriela.nicacio/20250708_145727/best_acc_model.pt'

model_config = getattr(ModelConfig, model_name)

#For MLP
#args = ...  #build args as required by your MLP class
mlp = MLP(args, path=model_config['path'])
#Prepare your input vector here (must be a float list or numpy array)
input_vector = ...
input_tensor = torch.tensor(input_vector, dtype=torch.float32).unsqueeze(0).to(device)
with torch.no_grad():
    logits = mlp(input_tensor)
    probs = torch.softmax(logits, dim=1)
    pred = torch.argmax(probs, dim=1)
    print("Predicted class:", pred.item())

### you could type your own inputs here.
# input_list = ["",]

if __name__ == '__main__':
    dropout = 0.2

    '''for user_input in input_list:
        input_str = complete_input(model_config, user_input)
        input_ids = tokenizer(input_str, truncation=True, return_tensors='pt').input_ids.to(device)
        generate_ids = model.generate(input_ids, max_new_tokens=256)
        model_output = tokenizer.decode(generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
        print(f'Input: {user_input}\nOutput: {model_output}\n')
'''