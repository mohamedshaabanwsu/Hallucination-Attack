import os, math, torch, pickle
from tqdm import tqdm
from datetime import datetime
from torch.nn.functional import cross_entropy
from config import ModelConfig
from utils import load_model_and_tokenizer, complete_input, extract_model_embedding, complete_input_with_target
from os.path import exists
import mind_functions as mf
import numpy as np
from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE
import pandas as pd
from openpyxl import load_workbook

class Attacker:

    def __init__(self, model_name, init_input, target, device='cuda:0', steps=3000, topk=256, batch_size=1024, mini_batch_size=16, **kwargs):
        try:
            self.model_config = getattr(ModelConfig, model_name)#[0]
        except AttributeError:
            raise NotImplementedError

        self.model_name = model_name
        self.init_input = init_input
        self.target = target
        self.device = device
        self.steps = steps
        self.topk = topk
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.mini_batches = math.ceil(self.batch_size/self.mini_batch_size)
        self.kwargs = kwargs
        self.model, self.tokenizer = load_model_and_tokenizer(
            self.model_config['path'], self.device, False
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.eos_token_id #added this

        #load Mind model
        self.mind_model_name = 'MIND_model'
        
        try:
            self.mind_model_config = getattr(ModelConfig, self.mind_model_name)
        except AttributeError:
            raise NotImplementedError
        
        #self.mind_model, self.mind_tokenizer = mf.mind_load_model_and_tokenizer( #will be none
        #    self.mind_model_config['path'], self.device, False
        #)

        self.mind_model = mf.Model(6144, "/scratch/user/gabriela.nicacio/20250924_164739/best_acc_model.pt")#self.mind_model_config['path'])

        self.mind_target = 0 #to be Non-hall even tho it should be getting hall answer
        self.mind_loss = None
        self.classification = None
        self.pred_prob = 0.0
        self.mind_entropy = None
        self.mind_probabilities = None
        self.hd_last = None
        self.hd_last_mean = None
        self.total_hd = None
        self.mind_loss_list = []
        self.llm_loss_grad = 0
        self.total_loss = 0
        self.hallu_sm = 0
        self.eval_score = None

        self.last_update = None
        self.temp_step = 0
        self.temp_input = self.init_input
        self.temp_output = ''
        self.temp_loss = 1e+9
        self.temp_grad = None
        self.temp_input_ids = None
        self.temp_sample_list = []
        self.temp_sample_ids = None

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

        self.input_str = None
        self.log_rows = [] #for logging results to excel
        self.step_candidate_dfs = []

        self.all_llm_losses = None
        self.all_mind_losses = None
        self.step_filename = "all_candidates_loss_new_seperate_attack_9_26_#2.xlsx"

        self.single_llm_loss = None
        self.single_mind_loss = None
        self.single_llm_loss_before_norm = None
        self.single_mind_loss_before_norm = None
        #print("llm loss that got min total loss:", single_llm_loss)
        #print("mind loss that got min total loss:", single_mind_loss)
        self.min_loss = None
        self.min_index = None
        self.mind_used = False

        self.column_names = [
            "Step",
            "Input",
            "Full Input",
            "Output",
            "Update",                # True/False if update step accepted
            "MIND classification",   # e.g. "Non-Hallucination" or "Hallucination"
            "Binary Class",          # 0 or 1, model binary hallucination call
            "MIND Score",            # self.hallu_sm, MIND model score
            "Score before softmax",
            "Temp Loss",             # Current candidate loss
            "Route Loss",            # Best route loss found
        ]


    def append_table_to_excel(self, filename, df):
        sheet_name='Sheet1'
        if not os.path.isfile(filename):
            # File doesn't exist: create new
            df.to_excel(filename, index=False, sheet_name=sheet_name)
        else:
            def clean_excel_string(s):
                if isinstance(s, str):
                    return ILLEGAL_CHARACTERS_RE.sub('', s)
                return s
            df = df.map(clean_excel_string)
        
            # Append to existing file without direct writer.book assignment
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
                # Just write directly; no need to set writer.book or writer.sheets
                # 'overlay' allows appending to existing sheet without erasing
                startrow = writer.sheets[sheet_name].max_row if sheet_name in writer.sheets else 0
                df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=False, header=False)
                    

    def append_df_to_excel(self, filename, df, sheet_name):
        if not os.path.isfile(filename):
            # If file doesn't exist, write new file
            df.to_excel(filename, sheet_name=sheet_name, index=False)
        else:

            def clean_excel_string(s):
                if isinstance(s, str):
                    return ILLEGAL_CHARACTERS_RE.sub('', s)
                return s
            df = df.map(clean_excel_string)
            # Otherwise, append sheet to existing file
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists="replace") as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
                
    '''
    def print_and_write_attack_with_mind(self):

        if self.classification == 0:
            label = "Non-Hallucination"
        else:
            label = "Hallucination"

        print(f'Step  : {self.temp_step}/{self.steps}\n'
              f'Input : {self.temp_input}\n'
              f'Full Input : {self.input_str}\n'
              f'Output: {self.temp_output}\n'
              f'MIND Classification: {label}\n'
              #f'MIND binaryval: {self.classification}\n'
              f'MIND score: {self.hallu_sm}\n')
            
        self.log_rows.append({
            "Step": self.temp_step,
            "Input": self.temp_input,
            "Full Input": self.input_str,
            "Output": self.temp_output,
            "Update": self.last_update,  # to be set in update()
            "MIND classification": label,  # already computed in your test()
            "Binary Class": self.classification,
            "MIND Score": self.hallu_sm,
            "Score before softmax": self.eval_score,

            "Temp Loss": self.temp_loss,  # make sure this is set before test() each loop
            "Route Loss": self.route_loss,

        })

        df = pd.DataFrame([self.log_rows[-1]], columns=self.column_names)
        self.append_table_to_excel('results__9__#13.xlsx', df)

    def print_and_write_original_attack(self):
        self.log_rows.append({
            "Step": self.temp_step,
            "Input": self.temp_input,
            "Full Input": self.input_str,
            "Output": self.temp_output,
            "Update": self.last_update,  # to be set in update()
            "Temp Loss": self.temp_loss,  # make sure this is set before test() each loop
            "Route Loss": self.route_loss
        })


        df = pd.DataFrame([self.log_rows[-1]], columns=self.column_names)
        self.append_table_to_excel('results__9__#1.xlsx', df)'''

    def log_attack_step(self):

        print(f"Step   : {self.temp_step}/{self.steps}")
        print(f"Input  : {self.temp_input}")
        print(f"Full Input : {self.input_str}")
        print(f"Output : {self.temp_output}")
        print(f"Update : {self.last_update}")
        print(f"Temp Loss: {self.temp_loss}")
        print(f"Route Loss: {self.route_loss}")
        
        if self.mind_used:
            label = "Non-Hallucination" if self.classification == 0 else "Hallucination"
            mind_classification = label
            binary_class = self.classification
            mind_score = self.hallu_sm
            score_before_softmax = self.eval_score

            print(f"MIND Classification: {label}")
            print(f"Binary Class: {self.classification}")
            print(f"MIND Score: {self.hallu_sm}")
            print(f"Score before softmax: {self.eval_score}")

        else:
            mind_classification = None
            binary_class = None
            mind_score = None
            score_before_softmax = None

        if not hasattr(self, 'log_rows'):
            self.log_rows = []

        row = {
            "Step": self.temp_step,
            "Input": self.temp_input,
            "Full Input": self.input_str,
            "Output": self.temp_output,
            "Update": self.last_update,
            "MIND classification": mind_classification,
            "Binary Class": binary_class,
            "MIND Score": mind_score,
            "Score before softmax": score_before_softmax,
            "Temp Loss": self.temp_loss,
            "Route Loss": self.route_loss
        }

        
        self.log_rows.append(row)
        df = pd.DataFrame([row], columns=self.column_names)
        self.append_table_to_excel('results_new_seperate_attack_9_26_#2.xlsx', df)


    def test_just_llm_attack(self):

        self.model.eval()
        self.input_str = complete_input(self.model_config, self.temp_input)

        input_ids = self.tokenizer(
            self.input_str, truncation=True, return_tensors='pt'
        ).input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=96)
        self.model.train()
        self.temp_output = self.tokenizer.decode(
            generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
        )

        self.log_attack_step()

        self.input_list.append(self.temp_input)
        self.output_list.append(self.temp_output)


    def test_both_llm_mind(self):
        self.model.eval()
        self.input_str = complete_input(self.model_config, self.temp_input)

        tokens = self.tokenizer(
            self.input_str, truncation=True, max_length=512,
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
        self.temp_output = self.tokenizer.decode(
            generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        #test MIND #, self.mind_probabilities, self.mind_entropy
        self.hd_last, self.hd_last_mean = mf.get_hd(self.temp_output, self.input_str) #giving function llm output and input string has prompt --seefunc params
        self.total_hd = self.hd_last + self.hd_last_mean
        self.eval_score, self.classification, self.hallu_sm = self.mind_model.eval(self.total_hd) #pred_prob = (how conf in if hall or not --use in calc loss)
        #write to excel

        self.log_attack_step()

        self.input_list.append(self.temp_input)
        self.output_list.append(self.temp_output)


    def slice(self):
        prefix = self.model_config.get('prefix', '')
        prompt = self.model_config.get('prompt', '')
        suffix = self.model_config.get('suffix', '')
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
        temp_str += self.target
        temp_tokens = self.tokenizer(temp_str).input_ids
        self.target_slice = slice(len2, len(temp_tokens))


    def grad(self): #save the gradients (used to help select adv tokens)
        #[vocab_size, embedding_dim]  tokens model recognizes and their embeddings
        model_embed = extract_model_embedding(self.model)
        embed_weights = model_embed.weight 

        #build input (config + current string to perturb + target) to get input_ids
        input_str = complete_input(self.model_config, self.route_input)
        if input_str.endswith(':'):
            input_str += ' '
        input_str += self.target #adds target hallucination to input
        input_ids = self.tokenizer(
            input_str, truncation=True, max_length = 512, return_tensors='pt' #added max len
        ).input_ids[0].to(self.device)
        self.temp_input_ids = input_ids.detach() #creates new tensor with same data but not tracking gradients -memory efficient, like snapshot
        
        # setup and equation 8 
        compute_one_hot = torch.zeros( #initialize w/ 0s for 1 hot matrix [num_modifiable_pos, vocab_size]
            self.input_slice.stop-self.input_slice.start, # modifiable parts of input
            embed_weights.shape[0], 
            dtype=embed_weights.dtype, device=self.device
        )
        compute_one_hot.scatter_(   
            # put 1 for each token at index i each row
            1, input_ids[self.input_slice].unsqueeze(1), 
            #adds another dimension [num_modifiable_positions, vocab_size, 1] i think for compatibility
            torch.ones(
                compute_one_hot.shape[0], 1, device=self.device, dtype=embed_weights.dtype 
            ) 
        )
        #The above: setting up input slice tokens and models vocab so that each token can be replaced by any candidate in the vocab

        #enabling gradients on one hot tensor so that after backprop .grad shows how loss changes wrt each tok repl
        compute_one_hot.requires_grad_() #needed for later & in equ 8  [e_adv - e_x_i] maybe implicit

        #HOLDS embeddings of input tokens (1 per row )
        #unsqueeze adds another dimension, shape: [1, num_mod_pos, embedd_dim]
        compute_embeds = (compute_one_hot @ embed_weights).unsqueeze(0) # --just for model expected input format -no data change

        #get original embeddings of input sequence
        raw_embeds = model_embed(input_ids.unsqueeze(0)).detach() #adds another dimension downward [1, seq_len, emb_dim] detach to disable grad tracking
        concat_embeds = torch.cat([ #holds:
            raw_embeds[:, :self.input_slice.start, :], #embeddings of original prefix
            compute_embeds, #embeddings of current input to perturb
            raw_embeds[:, self.input_slice.stop: , :] #of suffix 
        ], dim=1) #[1, total_seq_len, embed_dim]

        try:
            #replacing input each time with updated conc_emb
            #gets logits -- unnormalized scores/predictions for new output [sequence_length, vocab_size]
            logits = self.model(inputs_embeds=concat_embeds).logits[0] #this removes batch dimension so that 1 cuz dont need
        except AttributeError:
            logits = self.model(input_ids=input_ids.unsqueeze(0), inputs_embeds=concat_embeds)[0] #if that dont work try this way similarly-for diff models
        if logits.dim()>2:
            logits = logits.squeeze() # for if some models have extra dimensions squeeze(remove i think)
        try:
            assert input_ids.shape[0]>=self.target_slice.stop
        except AssertionError:
            self.target_slice = slice(self.target_slice.start, input_ids.shape[0]) #making target slice larger to input len if input is already larger

        #equation 7/8 logp(y~|x)
        compute_logits = logits[self.target_slice.start-1 : self.target_slice.stop-1] #only get logits for corresponding to target output positions only # Indexing with -1 due to offset for next-token prediction
        target = input_ids[self.target_slice] #get target token ids
        # scalar loss value. --how well did #from logits to cross entropy that part of the equ is happening
        llm_loss = cross_entropy(compute_logits, target) #from equ7/8. = -∑logp(˜ y|x) same as argmax log....(minimizing loss vs maximizing likelihood of sucess)
        
        llm_loss.backward() 
        self.llm_loss_grad = llm_loss.item()
        #backpropagate loss to get gradients wrt one-hot encoding of tokens 
        #NEED to get gradients that will help get tokens that will cause hallucination but that MIND will detect as non hall
        
        #finally get ∇e_xi  eq 8
        self.temp_grad = compute_one_hot.grad.detach() #detach no need track gradients anymore, saving it
        #[num_mod_pos, vocab_size] -holds gradients(sensitvity of loss) for each token in vocab for each modifiable position


    def sample(self):#equation 9 :  non literal algoritmic step. Whole set of adv prompt candidates. But are sampled from in equ 10
        self.temp_sample_list = []
        #(values have actual val of topk tokens to replace) (indices are where located in the tensor they are stored in)
        #gets candidate token replacement set of topk (256) per modifiable position
        values, indices = torch.topk(self.temp_grad, k=self.topk, dim=1) # Eq. 8 Topk([eadv−exi]⊺∇exi logp(˜ y|x)) gets top k tokens per pos i with largest neg gradients (tokens that by replacing decrease loss most) 
       
        #Eq. 10 random sampling B candidates from all top-k positions
        #shuffles those candidates   #top repl cand per pos * # modif tok pos = total # candidates
         #THEN: adds to list selected batch_size(1024) random samples for each token to repl -->so expected to have at least 1024 adv tokens total --not per token??
        sample_indices = torch.randperm(self.topk * self.temp_grad.shape[0])[:self.batch_size].tolist()
        #randperm(256*1) 0-n                         
        
        #to get 1024 combinations of new promptsorigonal input replicated batch size times with only 1 token changed per sequence
        #for each sample index, finds the position in the original input to replace and the ID of adv token to replace it with
        for i in range(self.batch_size): #so 1024 pairs of (pos, pos_index) 
            pos = sample_indices[i] // self.topk #div by 256 gets index from 0-1024 of original token index
            pos_index = indices[pos][sample_indices[i] % self.topk].item()  #specific index or ID of adv token to replace with
            self.temp_sample_list.append((pos, pos_index)) #adds pair to list
        pos_list, pos_index_list = zip(*self.temp_sample_list) #seems to make the pairs into two lists: positions to perturb and other with adv token indices
        pos_tensor = torch.tensor(pos_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device) #put all indices of tokens to repl into a 1D tensor [batch_size](for compatibiltiy later)
        pos_tensor += self.input_slice.start #adds input slice start index # to each repl tok position so that it is relative to the whole input sequence (includes prefix + suffix..)
        pos_index_tensor = torch.tensor(pos_index_list, dtype=self.temp_input_ids.dtype, device=self.temp_input_ids.device) #again make 1D tensor of [batch_size] but of candidate adv token IDs
        
        sample_ids = self.temp_input_ids.repeat(self.batch_size, 1) #makes batch size copies of current input ids
        sample_ids[range(self.batch_size), pos_tensor] = pos_index_tensor # for each batch (row) replace that token pos with adv ones
        # eq.9/10 this is now prompt candidate set -- matrix with batch size options of X~ prompts with a single differing token each
        self.temp_sample_ids = sample_ids #[batch_size, input_len]   
       
            
    def calc_loss_mind(self, candidate_outputs, candidate_full_inputs):
        hd_last_batch, hd_last_mean_batch = mf.get_hd_batch(candidate_outputs, candidate_full_inputs, self.model, self.tokenizer)

        # average hd_last_batch over tokens dim to get vector [3072]
        hd_last_avg = [np.mean(hd_last, axis=0) for hd_last in hd_last_batch]

        # concatenate hd_last_avg and hd_last_mean to get  vector
        combined = [np.concatenate([avg, hd_mean]) for avg, hd_mean in zip(hd_last_avg, hd_last_mean_batch)]

        hds_tensor = torch.tensor(np.array(combined), dtype=torch.float32).to(self.device)  # [batch_size, 6144]

        logits = self.mind_model.model(hds_tensor)  # [batch_size, n_class]
        #print(logits) for this batch just to see one step or sm
        targets = torch.full((hds_tensor.shape[0],), self.mind_target, dtype=torch.long).to(self.device) #[batch_size] actually mini batch size
        #print(targets) 

        mind_losses = torch.nn.functional.cross_entropy(logits, targets, reduction='none')  # [batch_size]
        #print("MIND losses in this batch = ", mind_losses)

        #print if were to do softmax of logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)  # [batch_size, n_class]
        #print("Mind logts softmax = ", probabilities)
        #add all these values to columns in excel
        
        return mind_losses, logits, probabilities, targets

    def forward_original(self):
        loss = torch.empty(0, device=self.device)
        with tqdm(total=self.batch_size) as pbar:
            pbar.set_description('Processing')
            for mini_batch in range(self.mini_batches):
                start = mini_batch*self.mini_batch_size
                end = min((mini_batch+1)*self.mini_batch_size, self.batch_size)
                targets = self.temp_input_ids[self.target_slice].repeat(end-start, 1)
                logits = self.model(self.temp_sample_ids[start:end]).logits
                logits = logits.permute(0, 2, 1)
                mini_batch_loss = cross_entropy(
                    logits[:, :, self.target_slice.start - 1:self.target_slice.stop - 1],
                    targets, reduction='none'
                ).mean(dim=-1)
                loss = torch.cat([loss, mini_batch_loss.detach()])
                torch.cuda.empty_cache()
                pbar.update(end-start)

        min_loss, min_index = loss.min(dim=-1)
        self.temp_loss = min_loss.item()
        self.loss_list.append(self.temp_loss)

        self.temp_input_ids = self.temp_sample_ids[min_index]
        self.temp_input = self.tokenizer.decode(
            self.temp_input_ids[self.input_slice],
            skip_special_tokens=True,
        )
        if self.model_name == 'internlm':
            ### for internlm, there may be an additional blank space on the left side of the decode string
            self.temp_input = self.temp_input.lstrip()

    def forward_with_mind(self):
        
        candidate_outputs = []      # Store generated outputs for each candidate
        candidate_full_inputs = []  # Store full input strings for each candidate
        all_full_inputs_no_target = []
        all_candidate_outputs = []
        all_perturbed_inputs = []
        all_targets = []
        all_logits = []
        all_probabilities = []

        with tqdm(total=self.batch_size) as pbar:
            pbar.set_description('Processing')
            for mini_batch in range(self.mini_batches):
                start = mini_batch * self.mini_batch_size
                end = min((mini_batch + 1) * self.mini_batch_size, self.batch_size)

                # --- 1. LLM Loss (already batched/vectorized) ---
                targets = self.temp_input_ids[self.target_slice].repeat(end-start, 1)
                #dont need these below huh was for llm loss
                logits = self.model(self.temp_sample_ids[start:end]).logits
                logits = logits.permute(0, 2, 1)
                mini_batch_loss = cross_entropy(
                    logits[:, :, self.target_slice.start - 1: self.target_slice.stop - 1],
                    targets, reduction='none'
                ).mean(dim=-1)         # [mini_batch_size]

                # Step 1: Decode all candidate input IDs to strings (prompt strings)
                batch_candidate_ids = self.temp_sample_ids[start:end]  # shape [mini_batch_size, seq_len]
                
                candidate_prompts = self.tokenizer.batch_decode(
                    batch_candidate_ids, skip_special_tokens=True
                )
                
                #should print 8 candidate perturbed prompts
                #print("Candidate_Prompts = ", candidate_prompts)
                
                candidate_full_inputs = [complete_input(self.model_config, prompt) for prompt in candidate_prompts]
                # Step 2: For each candidate in the batch

                #print("Candidate_full_inputs = ", candidate_full_inputs) #should print 8 cnadidates with perturbed input + pre + suf + prompt

                # Step 3: Batch tokenize full input strings for generation
                tokens = self.tokenizer(
                    candidate_full_inputs,
                    return_tensors="pt",
                    padding=True,          # pad to max length in the batch automatically
                    truncation=True,
                    max_length=512
                ).to(self.device)

                # Step 4: Batch generate outputs for all candidates together
                generate_ids = self.model.generate(
                    input_ids=tokens["input_ids"],
                    attention_mask=tokens["attention_mask"],
                    max_new_tokens=96,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                # Step 5: Decode generated outputs per candidate from batch output
                candidate_outputs = [
                    self.tokenizer.decode(output_ids[tokens["input_ids"].shape[-1] :], skip_special_tokens=True)
                    for output_ids in generate_ids
                ]

                # --- 3. Batched Mind Loss ---
                mind_losses, logits, probabilities, targets = self.calc_loss_mind(candidate_outputs, candidate_full_inputs)  # should return a tensor or list of losses, [mini_batch_size]
                # If it's already a tensor, use as-is, otherwise convert:
                if not isinstance(mind_losses, torch.Tensor):
                    mind_losses = torch.tensor(mind_losses, device=self.device)

                all_targets.extend(targets.detach().cpu().tolist())
                all_logits.extend(logits.detach().cpu().numpy().tolist())
                all_probabilities.extend(probabilities.detach().cpu().numpy().tolist())

                # --- 4. Append for later selection ---
                if mini_batch == 0: #when first mini batch initialized tensors
                  
                    self.all_mind_losses = mind_losses.detach().clone()

                else:
                     #adds each cand batch losses to tensors after eaach mini batch
                    self.all_mind_losses = torch.cat([self.all_mind_losses, mind_losses.detach().clone()])

                # batch_candidate_ids is [mini_batch_size, seq_len]
                perturbed_inputs = [
                    self.tokenizer.decode(batch_candidate_ids[i, self.input_slice], skip_special_tokens=True)
                    for i in range(batch_candidate_ids.size(0))
                ]
                full_inputs_no_target = [
                    complete_input(self.model_config, pert_inp) for pert_inp in perturbed_inputs
                ]
                # candidate_outputs already built for each mini batch

                all_perturbed_inputs.extend(perturbed_inputs)
                all_full_inputs_no_target.extend(full_inputs_no_target)
                all_candidate_outputs.extend(candidate_outputs)

                torch.cuda.empty_cache()
                pbar.update(end-start)

    
        min_loss, min_index = self.all_mind_losses.min(dim=-1)
        
        self.single_mind_loss = self.all_mind_losses[min_index].item()

        self.min_loss = min_loss.item()
        self.min_index = min_index.item()

        mind_losses = self.all_mind_losses.detach().cpu().tolist()

        #REMOVED LLM loss for candidates WANT TO OBSERVE ANYWAY????
        df_candidates = pd.DataFrame({
            "Perturbed Input": all_perturbed_inputs,
            "Full Input (no target)": all_full_inputs_no_target,
            "Target Output": all_candidate_outputs,             
            "MIND Loss": mind_losses,
            "MIND Target": all_targets,
            "MIND Logits": all_logits,
            "MIND Softmax": all_probabilities
        })
        
        sheet_name = f"Step_{self.temp_step}" # Or whatever your step number is
        self.append_df_to_excel(self.step_filename, df_candidates, sheet_name)

        # Store this df for later saving OR immediately write to Excel (recommended: store for later)
        #self.step_candidate_dfs.append(df_candidates)  # step_candidate_dfs = []

        self.temp_loss = min_loss.item()
        self.loss_list.append(self.temp_loss)

        # Select corresponding input/output for min_index
        self.temp_input_ids = self.temp_sample_ids[min_index]
        self.temp_input = self.tokenizer.decode(
            self.temp_input_ids[self.input_slice],
            skip_special_tokens=True
        )
        if self.model_name == 'internlm':
            self.temp_input = self.temp_input.lstrip()

    def update(self):
        update_strategy = self.kwargs.get('update_strategy', 'strict')
        is_update = False

        if update_strategy == 'strict':
            if self.temp_loss < self.route_loss:
                is_update = True
        elif update_strategy == 'gaussian':
            gap_step = min(self.temp_step - self.route_step_list[-1], 20)
            
            if (self.temp_loss/self.route_loss-1)*100/gap_step <= torch.randn(1)[0].abs():
                is_update = True

        self.last_update = is_update
        print(f'Temp Loss: {self.temp_loss}\tRoute Loss: {self.route_loss}\nUpdate:', 'True' if is_update else 'False', '\n')

        if is_update:
            self.route_step_list.append(self.temp_step)
            self.route_input = self.temp_input
            self.route_input_list.append(self.route_input)
            self.route_loss = self.temp_loss
            self.route_loss_list.append(self.route_loss)
            self.route_output_list.append(self.temp_output)


    def pre(self):
        #self.test()
        self.test_just_llm_attack() 
        print('='*128,'\n')
        self.route_step_list.append(self.temp_step)
        self.route_input_list.append(self.temp_input)
        self.route_output_list.append(self.temp_output)
        self.route_loss_list.append(self.route_loss)
        self.temp_step+=1


    def save(self):
        save_dir = self.kwargs.get('save_dir', './results')
        os.makedirs(save_dir, exist_ok=True)
        save_dict = {
            'model_name': self.model_name,
            'init_input': self.init_input,
            'target': self.target,
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
        with open(os.path.join(save_dir, pkl_name), mode='wb') as f:
            pickle.dump(save_dict, f)

       
    def run(self):
        self.pre()
        early_stop = self.kwargs.get('early_stop', False)
        while self.temp_step <= self.steps:

            if self.temp_output != self.target:
                self.mind_used = False
                print("Trying to get to Target String\n")
                self.slice()
                self.grad()
                self.sample()
                self.forward_original()
                self.test_just_llm_attack() 
                self.update()
            else:
                self.mind_used = True
                print("Perturbed String got Target String, Attempting to get MIND target now\n")
                self.slice()
                self.grad()
                self.sample()
                self.forward_with_mind()
                self.test_both_llm_mind() 
                self.update()

            self.temp_step += 1

            if (early_stop and self.temp_output == self.target) and (self.classification == self.mind_target): #add that mind needs to have its target too in order to get early stop
                break

        is_save = self.kwargs.get('is_save', False)
        if is_save:
            self.save()
