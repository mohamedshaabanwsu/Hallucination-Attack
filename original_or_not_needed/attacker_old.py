import os, math, torch, pickle
from tqdm import tqdm
from datetime import datetime
from torch.nn.functional import cross_entropy
from config import ModelConfig
from utils import load_model_and_tokenizer, mind_load_model_and_tokenizer, complete_input, extract_model_embedding

from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
#import get_hidden_states as ghs

import numpy as np

class Attacker:

    def __init__(self, model_name, init_input, target, device='cuda:0', steps=768, topk=256, batch_size=1024, mini_batch_size=16, **kwargs):
        
        self.mind_model_config = getattr(ModelConfig, model_name)#[0]

        self.model_name = model_name
        #self.init_input = init_input
        self.init_input = "Donald Trump was the victor of the United States presidential election in the year 2020."

        self.target = target
        self.device = device
        self.steps = steps
        self.topk = topk
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.mini_batches = math.ceil(self.batch_size/self.mini_batch_size)
        self.kwargs = kwargs

        self.mind_model, self.mind_tokenizer = mind_load_model_and_tokenizer( #will be none
            self.mind_model_config['path'], self.device, False
        )
        #MIND has no tokenizer
        self.temp_step = 0
        self.temp_input = self.init_input
        self.temp_output = ''
        self.temp_loss = 1e+9
        self.temp_grad = None
        self.temp_input_ids = None
        self.temp_sample_list = []
        self.temp_sample_ids = None
        self.hds = None #new hiddens states

        self.input_slice = None
        self.target_slice = None
        self.input_list = []
        self.output_list = []
        self.loss_list = []

        self.route_input = self.init_input
        self.route_loss = 1e+9
        self.route_step_list = []
        self.route_input_list = []
        self.route_output_list = []
        self.route_loss_list = []

        #using llama3.2 #TODO fix way loaded
        self.llama_model_name = 'llama3'
       
        self.llm_model_config = getattr(ModelConfig, 'llama3')
        
        self.model, self.tokenizer = load_model_and_tokenizer( 
            "meta-llama/Llama-3.2-3B-Instruct", self.device, False
        )


        
    def get_tokenized_ids(self):
        #was for entity part text = otext.replace("@", "").replace("  ", " ").replace("  ", " ")
        self.temp_input = self.tokenizer.decode(self.tokenizer(self.temp_input.strip(), return_tensors='pt')['input_ids'].tolist()[0]).replace("<s>", "").replace("</s>", "")
        
        return self.tokenizer(self.temp_input.strip(), return_tensors='pt')['input_ids'].tolist()


    def get_hd(self): #call with updated string each time
        ids = self.get_tokenized_ids() #TODO change to temp_input/ init_???text??
        hd = self.model(torch.tensor(ids).to(self.model.device), output_hidden_states=True).hidden_states
        hds = hd[1][0][-1].clone().detach()
        for i in range(2, len(hd)):
            hds += hd[i][0][-1].clone().detach()
        hds = hds / (len(hd) - 1)
        
        start_at = 2 #whats this??
        
        hds_mean_1 = torch.mean(hd[1][0][start_at-1:], dim=0)
        assert hds_mean_1.shape[0] == hd[1][0][-1].shape[-1]
        hds_mean_2 = torch.mean(hd[-1][0][start_at-1:], dim=0)

        hds_cpu = hds.detach().cpu().numpy()
        hds_mean_1_cpu = hds_mean_1.detach().cpu().numpy()
        hds_mean_2_cpu = hds_mean_2.detach().cpu().numpy()
        #return hds.tolist(), hds_mean_1.tolist(), hds_mean_2.tolist()
        full_vec = np.concatenate((hds_cpu, hds_mean_2_cpu)) #idk if right shape: only mean2 + last written to file in MIND
        
        self.hds = full_vec.tolist() #mind expects 6144 dim vector so concat
        print("Length: ", len(self.hds))

    #with torch.no_grad():
        #outputs = model(**inputs, output_hidden_states=True)
    # hidden_
    # use last and mean2 ??

    def test(self):
        self.mind_model.eval()

        #self.temp_input is a tensor or numpy array
        input_tensor = torch.tensor(self.hds, dtype=torch.float32).unsqueeze(0).to(self.device) #TODO self.hds instead of temp_input ???
        with torch.no_grad():
            logits = self.mind_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
        self.temp_output = f"{pred_class}" #Predicted class: 

        print(f'Step  : {self.temp_step}/{self.steps}\n'
              f'Input : {self.temp_input}\n'
              f'Output: {self.temp_output}')

        self.input_list.append(self.temp_input)
        self.output_list.append(self.temp_output)



    def slice(self):
        prefix = self.llm_model_config.get('prefix', '')
        prompt = self.llm_model_config.get('prompt', '')
        suffix = self.llm_model_config.get('suffix', '')
        temp_str = prefix+prompt
        temp_tokens = self.tokenizer(temp_str).input_ids
        len1 = len(temp_tokens)
        temp_str += self.route_input
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.input_slice = slice(len1, len(temp_tokens))
        try:
            assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.route_input
        except AssertionError:
            self.input_slice = slice(self.input_slice.start-1, self.input_slice.stop)
            try:
                assert self.tokenizer.decode(temp_tokens[self.input_slice]) == self.route_input
            except AssertionError:
                if self.tokenizer.decode(temp_tokens[self.input_slice]).lstrip() != self.route_input:
                    ### Todo
                    raise NotImplementedError

        temp_str += suffix
        temp_tokens = self.tokenizer(temp_str).input_ids
        len2 = len(temp_tokens)
        if suffix.endswith(':'):
            temp_str += ' '
        #temp_str += self.target
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.target_slice = slice(len2, len(temp_tokens))


    #FIXXXX (removed too many things)
    def grad(self): #MIND loss calc/gradient
        #DONT NEED THIS????
        input_str = complete_input(self.llm_model_config, self.route_input)
        
        if input_str.endswith(':'):
            input_str += ' '
        #input_str += self.target
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids[0].to(self.device)
        print("input string: ", input_str)
        print("input_ids shape:", input_ids.shape)

        self.temp_input_ids = input_ids.detach()

        # hds input to tensor and enable gradient tracking. changed self.rout_input to self.hds
        input_tensor = torch.tensor(self.hds, dtype=torch.float32).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        input_tensor.requires_grad_()  #now tracking a leaf tensor so .grad will be accessible
        print("input_tensor shape:", input_tensor.shape)
        
        target_tensor = torch.tensor([self.target], dtype=torch.long, device=self.device)
        print("target_tensor shape:", target_tensor.shape) 
        #Forward pass
        logits = self.mind_model(input_tensor)  
        print("logits shape:", logits.shape)
        loss = torch.nn.functional.cross_entropy(logits, target_tensor)

        #backward pass
        loss.backward()
        print("input_tensor.grad shape:", input_tensor.grad.shape)  # should match input_tensor.shape
        #save the gradient of the input vector
        self.temp_grad = input_tensor.grad.detach()#[0]#.cpu().numpy()[0]  # shape: [input_size]
        
        print("self.temp_grad shape:", self.temp_grad.shape)
        


    def sample(self): #equation nine non literal algoritmic step. Whole set of adv prompt candidates. But are sampled form in equ 10
        self.temp_sample_list = []
        #gets top k tokens per pos i with largest neg gradients from Eq. 8 (tokens that by replacing decrese loss)
        values, indices = torch.topk(self.temp_grad, k=self.topk, dim=1) # Topk([eadv−exi]⊺∇exi logp(˜ y|x)) --> gets candidate token replacement set i think
        #random sampling B candidates from all top-k positions from Eq. 10#TODO fix dim =1/0 or make input tensor 2D
        sample_indices = torch.randperm(self.topk * self.temp_grad.shape[0])[:self.batch_size].tolist() # X_B = {˜ xj|˜ xj ∼ ˜ X} from j=1 to B
        #randperm(256*6144)
        for i in range(self.batch_size): # this caused error, batch size larger than sample indices
        #for i, sample_indices in enumerate(sample_indices):
            pos = sample_indices[i] // self.topk #original token posi
            pos_index = indices[pos][sample_indices[i] % self.topk].item() #adv token pos to replace with
            self.temp_sample_list.append((pos, pos_index))
        #for sample tensors  and getting candidates
        pos_list, pos_index_list = zip(*self.temp_sample_list)
        pos_tensor = torch.tensor(pos_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device)
        #pos_tensor += self.input_slice.start #TODO need? start not defined would be 0?/
        pos_index_tensor = torch.tensor(pos_index_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device)

        #creates batch of candidate x_adv tokens by replacing sampled positions eq.10
        sample_ids = self.temp_input_ids.repeat(self.batch_size, 1) #replace tokens with adv ones
        sample_ids[range(self.batch_size), pos_tensor] = pos_index_tensor
        self.temp_sample_ids = sample_ids   #used in forward

        #Stores the batch in self.temp_sample_ids for use in forward().
        


    def forward(self):

        
        #Evaluates all perturbed input vectors in self.temp_sample_ids,computes loss against the target class, and selects the best one.
    
        '''loss_list = []

        # Convert target to tensor
        target_tensor = torch.tensor([self.target], dtype=torch.long, device=self.device)

        with torch.no_grad():
            with tqdm(total=self.batch_size) as pbar:
                pbar.set_description('Processing')
                for i in range(self.batch_size):
                    input_vec = self.temp_sample_ids[i].unsqueeze(0)  # shape: [1, input_dim]
                    logits = self.model(input_vec)  # shape: [1, num_classes] --classes as in seperate embeddings in sequence?? liike for tokens?
                    loss = cross_entropy(logits, target_tensor)
                    loss_list.append(loss.item())
                    pbar.update(1)

        # Convert to tensor for easy min() operation
        loss_tensor = torch.tensor(loss_list, device=self.device)
        min_loss, min_index = loss_tensor.min(dim=0) #get one with min loss. maximize likelihood of success

        #savebest loss and input
        self.temp_loss = min_loss.item()
        self.loss_list.append(self.temp_loss)

        #update temp_input with best perturbed vector
        self.temp_input_ids = self.temp_sample_ids[min_index]
        self.temp_input = self.temp_input_ids.detach().cpu().numpy().tolist()'''

        loss = torch.empty(0, device=self.device)
        with tqdm(total=self.batch_size) as pbar:
            pbar.set_description('Processing')
            for mini_batch in range(self.mini_batches):
                start = mini_batch*self.mini_batch_size
                end = min((mini_batch+1)*self.mini_batch_size, self.batch_size)
                targets = self.temp_input_ids[self.target_slice].repeat(end-start, 1)
                #Eq. 11?? compute logits + cross entropy loss for each candidate x_adv over and over
                logits = self.model(self.temp_sample_ids[start:end]).logits
                logits = logits.permute(0, 2, 1)
                mini_batch_loss = cross_entropy(
                    logits[:, :, self.target_slice.start - 1:self.target_slice.stop - 1],
                    targets, reduction='none'
                ).mean(dim=-1)
                loss = torch.cat([loss, mini_batch_loss.detach()])
                torch.cuda.empty_cache()
                pbar.update(end-start)

        #finds the candidate with the lowest loss
        min_loss, min_index = loss.min(dim=-1)
        self.temp_loss = min_loss.item()
        self.loss_list.append(self.temp_loss)
        
        #update input ids to the token with lowest loss
        self.temp_input_ids = self.temp_sample_ids[min_index]
        self.temp_input = self.tokenizer.decode(
            self.temp_input_ids[self.input_slice],
            skip_special_tokens=True,
        )



    def update(self):
    
        if (self.temp_loss/self.route_loss-1)*100/gap_step <= torch.randn(1)[0].abs():
            is_update = True
        update_strategy = self.kwargs.get('update_strategy', 'strict')
        is_update = False
        if update_strategy == 'strict':
            if self.temp_loss < self.route_loss:
                is_update = True
        elif update_strategy == 'gaussian':
            # Make sure route_step_list is not empty to avoid errors (should be managed in pre/run)
            gap_step = min(self.temp_step - self.route_step_list[-1], 20) if self.route_step_list else 1
            
            if self.route_loss == 0 or gap_step == 0:
                criterion = float('inf')  # or some large fallback
            else:
                criterion = (self.temp_loss / self.route_loss - 1) * 100 / gap_step
            if criterion <= torch.randn(1)[0].abs():
                is_update = True

        print(
            f"Temp Loss: {self.temp_loss}\t"
            f"Route Loss: {self.route_loss}\n"
            f"Update: {'True' if is_update else 'False'}\n"
        )

        if is_update:
            self.route_step_list.append(self.temp_step)
            self.route_input = self.temp_input         # Update to the best input vector so far
            self.route_input_list.append(self.route_input)
            self.route_loss = self.temp_loss
            self.route_loss_list.append(self.route_loss)
            self.route_output_list.append(self.temp_output)


    def pre(self):
        self.test()
        print('='*128,'\n')
        self.route_step_list.append(self.temp_step)
        self.route_input_list.append(self.temp_input) #string
        self.route_output_list.append(self.temp_output) #predicted class 0/1
        self.route_loss_list.append(self.route_loss)
        self.temp_step+=1


    def save(self):
        save_dir = self.kwargs.get('save_dir', './results')
        os.makedirs(save_dir, exist_ok=True)
        save_dict = {
            'model_name': self.model_name,
            'init_input': self.init_input, #should be numeric 
            'target': self.target, #class label(0/1)
            'steps': self.steps,
            'topk': self.topk,
            'batch_size': self.batch_size,
            'mini_batch_size': self.mini_batch_size,
            'kwargs': self.kwargs,
            'input_list': self.input_list,
            'output_list': self.output_list,
            'loss_list': self.loss_list,
            'route_step_list': self.route_step_list,
            'route_input_list': self.route_input_list,
            'route_output_list': self.route_output_list,
            'route_loss_list': self.route_loss_list
        }
        pkl_name = self.model_name+datetime.now().strftime("_%y%m%d%H%M%S.pkl")
        save_path = os.path.join(save_dir, pkl_name)
        #with open(os.path.join(save_dir, pkl_name), mode='wb') as f:
        with open(save_path, mode='wb') as f:
            pickle.dump(save_dict, f)
        
        print(f"Saved attack results to {save_path}")

    def run(self):
        self.get_hd()
        self.pre() #inital test on original input
        early_stop = self.kwargs.get('early_stop', False)

        while self.temp_step <= self.steps: # temp_step+1
            '''self.slice()
            self.grad()
            self.sample()
            self.forward()
            self.test()
            self.update()'''
            #self.temp_step += 1
            
            self.slice() #dont need?
            self.grad() #get loss from MIND #TODO - change func name
            self.sample() #candidate set
            self.forward() #choose best
            self.get_hd() #put updated string hru model to get hidden states
            self.test()
            
            if early_stop and int(self.temp_output.split()[-1]) == self.target:#self.temp_output == self.target:
                print("\nEarly stopping: matched target.\n")
                break
            #didn't stop, so we know that mind output is not target, do all over again
            self.update() #update string
            self.temp_step += 1
            
        is_save = self.kwargs.get('is_save', False) #change to tru if u want saved in pkl
        if is_save:
            self.save()

