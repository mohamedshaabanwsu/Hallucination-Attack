
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import PreTrainedTokenizerBase
B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]


model_family = "llama3.2" #"llamachat"

class Model(nn.Module):
    def __init__(self, input_size, path):

        super().__init__()
        
        self.model = nn.Sequential()
        self.model.add_module("dropout", nn.Dropout(0.2))
        self.model.add_module(f"linear1", nn.Linear(input_size, 256))
        self.model.add_module(f"relu1", nn.ReLU())
        self.model.add_module(f"linear2", nn.Linear(256, 128))
        self.model.add_module(f"relu2", nn.ReLU())
        self.model.add_module(f"linear3", nn.Linear(128, 64))
        self.model.add_module(f"relu3", nn.ReLU())
        self.model.add_module(f"linear4", nn.Linear(64, 2))
        self.device = "cuda:0"
        self.model.load_state_dict(torch.load(path, map_location = "cpu")["model_state_dict"]) #cpu or gpu??? TODO
        self.model.to(self.device)
        self.model.eval()
    
    #to get MIND models output (0/1) from given hd states
    def eval(self, hd):
        # assert len(llama[0]) == 4096*2
        input_ = torch.tensor([hd]).to(self.device)
        score = self.model(input_)
        print("Score before softmax:", score.tolist())
        hallu_sm = F.softmax(score, dim=1)#[:, 1]
        print("Hallu_sm:", hallu_sm)
        print("Score before softmax: ", score.tolist())
        #return hallu_sm[0].item()
        #commented above since i want binary val
        pred_class = torch.argmax(score, dim=1).item() #gets binary
        #pred_prob = softmax_output[0][pred_class].item()
        print("Pred class:", pred_class)
        return score.tolist(), pred_class, hallu_sm[:,  1].item()
    

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

def get_model(model_type, model_family, max_new_tokens=1):
    
      
    model_path = "meta-llama/Llama-3.2-3B" #f"meta-llama/Llama-2-{model_type}-chat-hf"
        
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float32,
        #low_cpu_mem_usage=True,
        device_map='auto')
    #tokenizer = LlamaTokenizer.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)#, legacy=False)
    at_id = 571#tokenizer.convert_tokens_to_ids("@")
   
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_8bit=False,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
        
    
    generation_config = dict(
                        top_k=0,
                        top_p=1.0,
                        do_sample=True,#wasfalse
                        num_beams=1,
                        max_new_tokens=max_new_tokens,
                        return_dict_in_generate=True,
                        output_hidden_states=True,
                        output_scores = True
                    )
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer, generation_config, at_id

model, tokenizer, generation_config, at_id = get_model("3b", "llama3.2", 1) #("7b", "llamachat", 1)

def mind_load_model_and_tokenizer(path, device, eval_mode=True):
    input_size = 6144# 6144#8192change figure out what need here?TODO
    model = Model(input_size, path)
    model.model.to(device)
    model.model.eval()
    tokenizer = None 
    return model, tokenizer


def get_pe(logit, id_, start_at):
    probabilities = F.softmax(logit, dim=2)
    log_probabilities = torch.log(probabilities)
    entropy = -probabilities * log_probabilities
    entropy_sum = torch.sum(entropy, dim=-1)

    pl = []
    el = []
    for i, idx in enumerate(id_[1:]):
        if i < start_at - 1:
            continue
        pl.append(probabilities[0][i][idx].item())
        el.append(entropy_sum[0][i].item())
    return pl, el


def chat_change_with_answer(dialog, answer_,
        tokenizer):
    prompt_tokens = []
    unsafe_requests = []
    unsafe_requests.append(
        any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
    )
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    # print(dialog)
    dialog_tokens = sum(
        [
            tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            ) + [2]
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST} {answer_.strip()}",
    )
    prompt_tokens.append(dialog_tokens)
    return prompt_tokens


def prompt_chat_for_labeled(prompt):
    return [{"role": "user", "content": prompt}]
    
def get_tokenized_ids(answer, q):   #important for extracting relevant hidden states
#determined pos where answer starts
    text = f"{q.strip()} {answer.strip()}" #concatenates the question and answer to be one string
    otext = q.strip() #holds just the question string

    if "chat" in model_family:
        id1 = chat_change_with_answer(prompt_chat_for_labeled(q), answer.strip(), tokenizer)[0]
        start_at = -1
        for i in range(len(id1)):
            if id1[i:i+4] == [518, 29914, 25580, 29962]:
                start_at = i
        if start_at == -1:
            raise Exception
        else:
            start_at += 4
        
    else:
        id1 = tokenizer(text.strip(), return_tensors='pt')['input_ids'].tolist()[0] #tokenize full (? + answer)
        id2 = tokenizer(otext.strip(), return_tensors='pt')['input_ids'].tolist()[0] #tokenzie just ? alone
        start_at = -1 #inizialize to beg
        for i in range(len(id1)):
            if i >= len(id2) or id1[i] != id2[i]:
                start_at = i #get to where answer tok begins
                break
    return [id1], start_at #returns input IDs (both ? + answer) and where the answer tokens pos starts


def get_hd(answer, q):
    ids, start_at = get_tokenized_ids(answer, q) #get input ids and where answer starts
    op = model(torch.tensor(ids).to(model.device), output_hidden_states=True) #puts list of ids into tensor
    hd = op.hidden_states #get hidden layers (1(batchsize), seq_len, hidden_dim --6144 or 4096 ...). The first element is the embedding layer output; is the output of layer 1,
    hds = hd[1][0][-1].clone().detach() #Initializes hds as the last token from the 2nd layer's hidden state
    for i in range(2, len(hd)):
        hds += hd[i][0][-1].clone().detach() #(.clonedetach for making copy and stop tracking gradiants --to isolate safely before changing - Sums the last token's hidden state from each subsequent layer, averages them
    hds = hds / (len(hd) - 1) #so hds is the average last-token hidden state across all layers except layer 0
    
    hds_mean = torch.mean(hd[-1][0][start_at-1:], dim=0) # computes the mean hidden state of the last (top) layer from the answer's start token (start_at-1) to the end.
    assert hds_mean.shape[0] == hd[1][0][-1].shape[-1] #to make sure vector sizes match?
    
    #logit = op.logits #gets logits
    #pl, el = get_pe(logit, ids[0], start_at) #uses logits and ids from q & a to get probability and entropy for answer tokens
    
    return hds.tolist(), hds_mean.tolist()#, pl, el
#Returns token-wise hidden states, mean hidden state, predicted probabilities and entropy

'''logits = outputs.logits
            hidden_states = outputs.hidden_states

            
            # Hidden state vector(s) for new token
            # Aggregate as in your get_hd (stack or sum hidden_states at last position)
            # Example: mean/concat/other, as per your MIND classifier expects
            # Here: last token, all layers
            token_hds = [layer[:, -1, :].clone().detach() for layer in hidden_states[1:]] # list of [1, 3072]
            # Average across layers, like get_hd
            hds_last = torch.mean(torch.stack(token_hds, dim=0), dim=0).squeeze(0)  # shape [3072]

            # If you want to mimic the mean-over-multiple-tokens, and only have one token, just use the same vector
            hds_last_mean = hds_last.clone()  # or, keep a buffer, but per-token, this is typically the same

            # Concatenate to shape [6144]
            hd_for_mind = torch.cat([hds_last, hds_last_mean], dim=-1).cpu().numpy().tolist()
            
            
            '''

def diff_score_per_token(self):
        if not hasattr(self, 'log_rows'):
            self.log_rows = []
        # Start sequence: encode prompt
        input_str = complete_input(self.model_config, self.temp_input)
        tokens = self.tokenizer(
            input_str, truncation=True, max_length=512,
            return_tensors='pt', padding=True, return_attention_mask=True
        )
        input_ids = tokens['input_ids'].to(self.device)
        attention_mask = tokens['attention_mask'].to(self.device)

        output_ids = input_ids.clone()
        stop_generation = False
        generated_tokens = []
        max_new_tokens = 96

        for step in range(max_new_tokens):
            # Feed current prompt+output so far to model with hidden states
            outputs = self.model(
                input_ids=output_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            logits = outputs.logits
            hidden_states = outputs.hidden_states

            # Softmax on last token
            next_token_logits = logits[:, -1, :]
            next_token_probs = torch.softmax(next_token_logits, dim=-1)
            # Sample or argmax for next token
            next_token_id = torch.argmax(next_token_probs, dim=-1, keepdim=True)
            # Or: next_token_id = torch.multinomial(next_token_probs, num_samples=1)

            # Hidden state vector(s) for new token
            # Aggregate as in your get_hd (stack or sum hidden_states at last position)
            # Example: mean/concat/other, as per your MIND classifier expects
            # Here: last token, all layers
            # Suppose hidden_states[1:] are all layers except embedding, for this token, last dim is hidden_dim=3072
            token_hds = [layer[:, -1, :].clone().detach() for layer in hidden_states[1:]] # list of [1, 3072]
            # Average across layers, like get_hd
            hds_last = torch.mean(torch.stack(token_hds, dim=0), dim=0).squeeze(0)  # shape [3072]

            # If you want to mimic the mean-over-multiple-tokens, and only have one token, just use the same vector
            hds_last_mean = hds_last.clone()  # or, keep a buffer, but per-token, this is typically the same

            # Concatenate to shape [6144]
            hd_for_mind = torch.cat([hds_last, hds_last_mean], dim=-1).cpu().numpy().tolist()
            # Now hd_for_mind is list of 6144 floats
            pred_class, hallu_sm = self.mind_model.eval(hd_for_mind)

            # Append generated token for decoding later
            output_ids = torch.cat([output_ids, next_token_id], dim=1)
            generated_tokens.append(next_token_id.item())  # for raw token IDs

            print(f"Token {step}: {hallu_sm}\t\t{pred_class}")
            self.log_rows.append({
                "Token" : step,
                "Score" : hallu_sm,
                "Pred Class": pred_class
            })

        # Finally, decode up to last safe token
        self.temp_output = self.tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
        self.input_list.append(self.temp_input)
        self.output_list.append(self.temp_output)

        self.hd_last, self.hd_last_mean, self.mind_probabilities, self.mind_entropy = mf.get_hd(self.temp_output, input_str) #giving function llm curr output + all prev tok and input string has prompt --seefunc params
        self.total_hd = self.hd_last + self.hd_last_mean
        self.classification, self.hallu_sm = self.mind_model.eval(self.total_hd) #pred_prob = (how conf in if hall or not --use in calc loss)

        self.print_and_write_original_attack()






'''print("temp ids from start to end in mini batch:", self.temp_sample_ids[start:end])
                print("temp ids from start to end in mini batch:", self.temp_sample_ids[start:end].shape)
                #print("target slice start-end, should be mult of same?", self.target_slice[self.target_slice.start - 1:self.target_slice.stop - 1])
                print("mini batch loss shape:", mini_batch_loss.shape)
                print("mini batch loss :", mini_batch_loss)'''



def get_tokenized_ids_batch(answers, qs, tokenizer):
    id_batch = []
    start_at_batch = []
    for answer, q in zip(answers, qs):
        # ... (your current logic per instance) ...
        id1, start_at = get_tokenized_ids(answer, q) # or inline code as above
        id_batch.append(id1[0])  # id1 is a list of one element
        start_at_batch.append(start_at)
    return id_batch, start_at_batch


def get_hd_batch(answers, qs, model, tokenizer):
    ids_batch, start_at_batch = get_tokenized_ids_batch(answers, qs, tokenizer)
    # ids_batch is [batch_size, seq_len]
    tokens_dict = tokenizer.pad(
        {"input_ids": ids_batch},
        padding=True,                  # pad to longest sequence in batch
        return_tensors="pt"            # return PyTorch tensor
    )

    ids_tensor = tokens_dict["input_ids"].to(model.device)  # Now shape [batch_size, max_seq_len]
    op = model(ids_tensor, output_hidden_states=True)      # {hidden_states: [layer, batch, seq_len, hidden_dim]}
    hd = op.hidden_states

    # hds: average over layers, for last token per instance
    # hds_mean: mean for answer tokens from start
    batch_size = ids_tensor.shape[0]
    num_layers = len(hd)
    hidden_dim = hd[-1].shape[-1]
    hds_list = []
    hds_mean_list = []
    for b in range(batch_size):
        hds = hd[b][-1].clone().detach()
        for i in range(2, num_layers):
            hds += hd[i][b][-1].clone().detach()
        hds = hds / (num_layers - 1)
        hds_list.append(hds.tolist())
        # Handle start_at possibly out of bounds, so clamp to valid range
        sa = max(start_at_batch[b] - 1, 0)
        hds_mean = torch.mean(hd[-1][b][sa:], dim=0)
        hds_mean_list.append(hds_mean.tolist())
    return hds_list, hds_mean_list


def no_batching_calc_mind_loss(self, llm_output_mind_loss, input_str_with_candidate):
        
        hd_last, hd_last_mean = get_hd(llm_output_mind_loss, input_str_with_candidate) #TODO #giving it cand string and full input also with perturbed input???(so only include pref+prompt+suff??)
        #should give it string not the temp sample ids maybe in batch         (remove start to end cuz that includes mult candidates in mini batch)
#detoneize
        hds = hd_last + hd_last_mean #add hd states
        
        input_tensor = torch.tensor([hds], dtype = torch.float32).to(self.device) #put then in tensor
        #get logits based on hd states put in tensor
        logits = self.mind_model.model(input_tensor) 
        #put target label in tensor too
        target = torch.tensor([self.mind_target], dtype=torch.long).to(self.device)
        #calc loss with cross entropy
        loss_func = torch.nn.CrossEntropyLoss()

        mind_loss = loss_func(logits, target)

        return mind_loss


def no_batching_mind_forward(self): #figure out which candidate is the best based on losses
        #loss = torch.empty(0, device=self.device) #empty tensor on device 0 -it will hold loss for each candidate

        #mind_losses = torch.empty(0, device=self.device)
        #both_losses = torch.empty(0, device=self.device)
        llm_loss_list = []         # Temporarily store LLM loss per candidate
        mind_loss_list = []

        with tqdm(total=self.batch_size) as pbar:
            pbar.set_description('Processing') #-progress bar u see in terminal output
            #process candidates in mini batches for memory efficiency
            for mini_batch in range(self.mini_batches): #mini_batches is batch_size/mini_batch_size --mini_batch_size is 32   1024/32 = 32
                
                #start/end basically defines range of candidates to process at time in this mini batch
                start = mini_batch*self.mini_batch_size #(will increaase by 32 each time to get to 1024)
                end = min((mini_batch+1)*self.mini_batch_size, self.batch_size)

                #to create target (ground_truth) for each candidate in the mini batch so can compare and compute loss
                targets = self.temp_input_ids[self.target_slice].repeat(end-start, 1) #[mini_batch_size, target_len] -repeats target token ids for each candidate in the mini batch
                #part of Eq. 11? compute logits for each candidate x_adv over and over
                logits = self.model(self.temp_sample_ids[start:end]).logits #[mini_batch_size, seq/input_length, vocab_size] 3rd dim vocab size has logits for index in target output per batch(unormalized predictions for next tok -want target)
              
                #changes order of dim in shape --> [mini_batch_size, vocab_size, seq_len] -so that target slice can be used to get logits for target output
                logits = logits.permute(0, 2, 1) #since pytorch expects this order/shape for cross entropy loss
                
                #Eq. 11
                mini_batch_loss = cross_entropy( #cross entropy loss for each candidate input over all target tokens
                    logits[:, :, self.target_slice.start - 1:self.target_slice.stop - 1],
                    targets, reduction='none'
                ).mean(dim=-1) #-1 means last dim
                #now get mind loss for each candidate (32 at time)
                              

                # Store each LLM loss
                for i in range(end - start):
                    llm_loss_list.append(mini_batch_loss[i].item())      

                for j in range(end-start): #for each cand in mini batch calc mind loss seperately 
                    candidate_ids = self.temp_sample_ids[start + j] # can't do or wont match with start:end processing above? o rif i do start:end its for loss for mult and will get seperated? try this TODO
                    #candidate_ids = self.temp_sample_ids[start:end] #to match with llm loss (not give one at time?? all in this batch should still be sep later tho?)
                    candidate_str = self.tokenizer.decode(candidate_ids, skip_special_tokens=True)
                    full_input_str = complete_input(self.model_config, candidate_str)
                    # Or decode the full input if needed: #TODO ????
                    #would have to add all losses per cand in this batch for mind and THEN add that to minibatchloss

                    #from test()
                    tokens = self.tokenizer(
                        full_input_str, truncation=True, max_length=512,
                        return_tensors='pt', padding=True,  #padding=True for safety changed this section to silence warnings
                        return_attention_mask=True
                    )
                    input_ids = tokens['input_ids'].to(self.device)
                    attention_mask = tokens['attention_mask'].to(self.device)

                    #generate_ids = self.model.generate(input_ids, max_new_tokens=96)
                    generate_ids = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pad_token_id=self.tokenizer.pad_token_id,
                        max_new_tokens=96
                    )

                    self.model.train()
                    temp_output = self.tokenizer.decode(
                        generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
                    )

                    indiv_mind_loss = self.calc_mind_loss(temp_output, full_input_str) # so need to do one by one not in batch but then maybe add losses to list and then add to minibatch loss (later losses seperated to choose min candid)
                    mind_loss_list.append(indiv_mind_loss)
                    #mind_losses = torch.cat([indiv_mind_loss #add each cand mind loss to tensor after processing 32 out of each of 32 batches
                
                #loss = torch.cat([loss, mini_batch_loss.detach()]) #will keep adding mini batch losses (32 at time) to loss tensor --but below we'll add 32 by 32 to m

                 #concatenates losses for each candidate in the mini batch to the loss tensor
                torch.cuda.empty_cache() #clears GPU memory cache --save mem
                pbar.update(end-start) #update output progress bar as each candidate processed in this mini batch
        
        #TODO am i allowed to do this after clear cache??
        '''for j in loss: #should be 1024
            llm_loss = loss[j].item() 
            mind_loss = mind_losses[j].item()
            added_losses = mini_loss + llm_loss #should match shapes
            both_losses = torch.cat(added_losses.detach()]) #add their losses and add to new loss tensor
'''
        #convert lists to tensors for easy addition and pairing
        loss = torch.tensor(llm_loss_list, device=self.device)        #[num_candidates]
        mind_losses = torch.tensor(mind_loss_list, device=self.device) #[num_candidates]same size
        both_losses = loss + mind_losses   
        min_loss, min_index = both_losses.min(dim=-1)
        #Eq 11 (argmax ...)finds the adv prompt candidate with the lowest loss
        #min_loss, min_index = loss.min(dim=-1) #stores min loss value and index of it from the loss tensor. loss is 1D tensor?
        #store loss amount, add to list (only ever used for logging pretty much)
        
        #new combined min loss
        self.temp_loss = min_loss.item() 
        self.loss_list.append(self.temp_loss)

        self.temp_input_ids = self.temp_sample_ids[min_index] #update input ids to this best candidate
        self.temp_input = self.tokenizer.decode( #decode the input ids back to human readable string
            self.temp_input_ids[self.input_slice],
            skip_special_tokens=True, #(check for issue print i saw earlier <|begin of text>| thing)
        )
        if self.model_name == 'internlm':
            ### for internlm, there may be an additional blank space on the left side of the decode string
            self.temp_input = self.temp_input.lstrip()


#TODO am i allowed to do this after clear cache??
        '''for j in loss: #should be 1024
            llm_loss = loss[j].item() 
            mind_loss = mind_losses[j].item()
            added_losses = mini_loss + llm_loss #should match shapes
            both_losses = torch.cat(added_losses.detach()]) #add their losses and add to new loss tensor
'''