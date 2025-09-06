import os, math, torch, pickle
from tqdm import tqdm
from datetime import datetime
from torch.nn.functional import cross_entropy
from config import ModelConfig
from utils import load_model_and_tokenizer, complete_input, extract_model_embedding


class Attacker:

    def __init__(self, model_name, init_input, target, device='cuda:0', steps=768, topk=256, batch_size=1024, mini_batch_size=16, **kwargs):
        try:
            self.model_config = getattr(ModelConfig, model_name)[0]
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
        print("batch_size:", self.batch_size)
        print("mini_batches:", self.mini_batches)
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


    def test(self):
        self.model.eval()
        input_str = complete_input(self.model_config, self.temp_input)
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids.to(self.device)
        generate_ids = self.model.generate(input_ids, max_new_tokens=96)
        self.model.train()
        self.temp_output = self.tokenizer.decode(
            generate_ids[0][input_ids.shape[-1]:], skip_special_tokens=True
        )
        print(f'Step  : {self.temp_step}/{self.steps}\n'
              f'Input : {self.temp_input}\n'
              f'Output: {self.temp_output}')

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
        print("input slice: ", self.input_slice)
        print("target slice: ", self.target_slice)


    def grad(self): #save the gradients (used to help select adv tokens)
        #[vocab_size, embedding_dim]  tokens model recognizes and their embeddings
        model_embed = extract_model_embedding(self.model)
        embed_weights = model_embed.weight 
       
        print("embed_weights shape: ", embed_weights.shape)
        #build input (config + current string to perturb + target) to get input_ids
        input_str = complete_input(self.model_config, self.route_input)
        if input_str.endswith(':'):
            input_str += ' '
        input_str += self.target #adds target hallucination to input
        input_ids = self.tokenizer(
            input_str, truncation=True, return_tensors='pt'
        ).input_ids[0].to(self.device)
        self.temp_input_ids = input_ids.detach() #creates new tensor with same data but not tracking gradients -memory efficient, like snapshot
        print("input string: ", input_str)
        print("input_ids shape: ", self.temp_input_ids.shape)
        # setup and equation 8 
        compute_one_hot = torch.zeros( #initialize w/ 0s for 1 hot matrix [num_modifiable_pos, vocab_size]
            self.input_slice.stop-self.input_slice.start, # modifiable parts of input
            embed_weights.shape[0], 
            dtype=embed_weights.dtype, device=self.device
        )
        print("compute_one_hot shape: ", compute_one_hot.shape)
        
        compute_one_hot.scatter_(   
            # put 1 for each token at index i each row
            1, input_ids[self.input_slice].unsqueeze(1), 
            #adds another dimension [num_modifiable_positions, vocab_size, 1] i think for compatibility
            torch.ones(
                compute_one_hot.shape[0], 1, device=self.device, dtype=embed_weights.dtype 
            ) 
        )
        print("compute_one_hot shape after scatter: ", compute_one_hot.shape)
        #The above: setting up input slice tokens and models vocab so that each token can be replaced by any candidate in the vocab
        #enabling gradients on one hot tensor so that after backprop .grad shows how loss changes wrt each tok repl
        compute_one_hot.requires_grad_() #needed for later & in equ 8  [e_adv - e_x_i] maybe implicit

        #HOLDS embeddings of input tokens (1 per row )
        #unsqueeze adds another dimension, shape: [1, num_mod_pos, embedd_dim]
        compute_embeds = (compute_one_hot @ embed_weights).unsqueeze(0) # --just for model expected input format -no data change
        print("compute_embeds[0][0]:", compute_embeds[0][0])
        print("compute_embeds shape:", compute_embeds.shape)
        '''
        #many e_adv chosen for each token based on how if replacing i with j(token in vocab) would decrease loss 
        #sums up inner product between the embedding difference (ej - exi) --like in equ
        # equ 8 : gets e_adv(many) for each positions (embedding of adv token) "T_adv represented as 1 hot vectors, embedded to form e_adv"
        #[e_adv - e_x_i]^T eq. 8 (not literally taking difference & transposes it)'''

        #get original embeddings of input sequence
        raw_embeds = model_embed(input_ids.unsqueeze(0)).detach() #adds another dimension downward [1, seq_len, emb_dim] detach to disable grad tracking
        print("raw_embeds[0][0]:", raw_embeds[0][0])
        print("raw_embeds shape:", raw_embeds.shape)
        concat_embeds = torch.cat([ #holds:
            raw_embeds[:, :self.input_slice.start, :], #embeddings of original prefix
            compute_embeds, #embeddings of current input to perturb
            raw_embeds[:, self.input_slice.stop: , :] #of suffix 
        ], dim=1) #[1, total_seq_len, embed_dim]
        print("concat_embeds[0][0]:", concat_embeds[0][0])
        print("concat_embeds shape:", concat_embeds.shape)
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
        print("logits[0]:", logits[0])
        print("logits shape:", logits.shape)
        #equation 7/8 logp(y~|x)
        compute_logits = logits[self.target_slice.start-1 : self.target_slice.stop-1] #only get logits (prob/score of a tok in vocab being next one)for corresponding to target output positions only # Indexing with -1 due to offset for next-token prediction
        print("compute_logits[0]:", compute_logits[0])
        print("compute_logits shape:", compute_logits.shape)
        target = input_ids[self.target_slice] #get target token ids
        print("target[0]:", target[0])
        print("target shape:", target.shape)
        # scalar loss value. --how well did
        loss = cross_entropy(compute_logits, target) #from equ7/8. = -∑logp(˜ y|x) same as argmax log....(minimizing loss vs maximizing likelihood of sucess)
        print("loss:", loss.item())
        #from logits to cross entropy that part of the equ is happening
        loss.backward() #backpropagate loss to get gradients wrt one-hot encoding of tokens 

        #finally get ∇e_xi  eq 8
        self.temp_grad = compute_one_hot.grad.detach() #detach no need track gradients anymore, saving it
        #[num_mod_pos, vocab_size] -holds gradients(sensitvity of loss) for each token in vocab for each modifiable position
        print("self.temp_grad[0]:", self.temp_grad[0])
        print("self.temp_grad shape:", self.temp_grad.shape)

    def sample(self):#equation 9 :  non literal algoritmic step. Whole set of adv prompt candidates. But are sampled from in equ 10
        self.temp_sample_list = []
        #(values have actual val of topk tokens to replace) (indices are where located in the tensor they are stored in)
        #gets candidate token replacement set of topk (256) per modifiable position
        values, indices = torch.topk(self.temp_grad, k=self.topk, dim=1) # Eq. 8 Topk([eadv−exi]⊺∇exi logp(˜ y|x)) gets top k tokens per pos i with largest neg gradients (tokens that by replacing decrease loss most) 
        print("values[0]:", values[0])
        print("indices[0]:", indices[0])
        print("values shape:", values.shape)
        print("indices shape:", indices.shape)
        #Eq. 10 random sampling B candidates from all top-k positions
        #shuffles those candidates   #top repl cand per pos * # modif tok pos = total # candidates
         #THEN: adds to list selected batch_size(1024) random samples for each token to repl -->so expected to have at least 1024 adv tokens total --not per token??
        sample_indices = torch.randperm(self.topk * self.temp_grad.shape[0])[:self.batch_size].tolist()
        print("sample_indices:", sample_indices[:3]) 
        print("sample_indices len:", len(sample_indices))
        #randperm(256*1) 0-n                         what shape i had printed: [1,6144] (see how messed up cuz class target is 0 or 1)
        
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
        print("pos_tensor shape:", pos_tensor.shape)
        print("pos_index_tensor shape:", pos_index_tensor.shape)
        print("pos_tensor[:3]:", pos_tensor[:3])
        print("pos_index_tensor[:3]:", pos_index_tensor[:3])
        sample_ids = self.temp_input_ids.repeat(self.batch_size, 1) #makes batch size copies of current input ids
        sample_ids[range(self.batch_size), pos_tensor] = pos_index_tensor # for each batch (row) replace that token pos with adv ones
        # eq.9/10 this is now prompt candidate set -- matrix with batch size options of X~ prompts with a single differing token each
        print("sample_ids[0]:", sample_ids[0])
        print("sample_ids shape:", sample_ids.shape)
        self.temp_sample_ids = sample_ids #[batch_size, input_len] 
        print("temp_sample_ids[0]:", self.temp_sample_ids[0])
        print("temp_sample_ids shape:", self.temp_sample_ids.shape)


    def forward(self): #figure out which candidate is the best based on losses
        loss = torch.empty(0, device=self.device) #empty tensor on device 0 -it will hold loss for each candidate
        
        with tqdm(total=self.batch_size) as pbar:
            pbar.set_description('Processing') #-progress bar u see in terminal output
            #process candidates in mini batches for memory efficiency
            for mini_batch in range(self.mini_batches): #mini_batches is batch_size/mini_batch_size --mini_batch_size is 16   1024/16
                
                #start/end basically defines range of candidates to process at time in this mini batch
                start = mini_batch*self.mini_batch_size #(1024/16) * 16 = 1024
                end = min((mini_batch+1)*self.mini_batch_size, self.batch_size)
                print(f"Mini batch {mini_batch}: start={start}, end={end}")

                #to create target (ground_truth) for each candidate in the mini batch so can compare and compute loss
                targets = self.temp_input_ids[self.target_slice].repeat(end-start, 1) #[mini_batch_size, target_len] -repeats target token ids for each candidate in the mini batch
                #part of Eq. 11? compute logits for each candidate x_adv over and over
                print("targets[0]:", targets[0])
                print("targets shape:", targets.shape) 
                logits = self.model(self.temp_sample_ids[start:end]).logits #[mini_batch_size, seq/input_length, vocab_size] 3rd dim vocab size has logits for index in target output per batch(unormalized predictions for next tok -want target)
                
                #changes order of dim in shape --> [mini_batch_size, vocab_size, seq_len] -so that target slice can be used to get logits for target output
                logits = logits.permute(0, 2, 1) #since pytorch expects this order/shape for cross entropy loss
                print("logits.shape:", logits.shape)
                print("logits[0, :, :5]:", logits[0, :, :5])
                #Eq. 11
                mini_batch_loss = cross_entropy( #cross entropy loss for each candidate input over all target tokens
                    logits[:, :, self.target_slice.start - 1:self.target_slice.stop - 1],
                    targets, reduction='none'
                ).mean(dim=-1) #-1 means last dim
                print("mini_batch_loss[:5]:", mini_batch_loss[:5])
                print("mini_batch_loss shape:", mini_batch_loss.shape)
                loss = torch.cat([loss, mini_batch_loss.detach()]) #concatenates losses for each candidate in the mini batch to the loss tensor
                torch.cuda.empty_cache() #clears GPU memory cache --save mem
                pbar.update(end-start) #update output progress bar as each candidate processed in this mini batch
        
        #Eq 11 (argmax ...)finds the adv prompt candidate with the lowest loss
        min_loss, min_index = loss.min(dim=-1) #stores min loss value and index of it from the loss tensor. loss is 1D tensor?
        #store loss amount, add to list (only ever used for logging pretty much)
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


    def update(self):
        update_strategy = self.kwargs.get('update_strategy', 'strict')

        is_update = False
        if update_strategy == 'strict': #IN MAIN: gaussian stragety used --why not strict? maybe better
            if self.temp_loss<self.route_loss: 
                is_update = True
        elif update_strategy == 'gaussian':#current loss must be lower than prev loss
            gap_step = min(self.temp_step - self.route_step_list[-1], 20)
            if (self.temp_loss/self.route_loss-1)*100/gap_step <= torch.randn(1)[0].abs():
                is_update = True

        print(f'Temp Loss: {self.temp_loss}\t'
              f'Route Loss: {self.route_loss}\n'
              f'Update:', 'True' if is_update else 'False', '\n')

        if is_update:
            self.route_step_list.append(self.temp_step)
            self.route_input = self.temp_input
            self.route_input_list.append(self.route_input)
            self.route_loss = self.temp_loss
            self.route_loss_list.append(self.route_loss)
            self.route_output_list.append(self.temp_output)


    def pre(self):
        self.test()
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
            self.slice()
            self.grad()
            self.sample()
            self.forward()
            self.test()
            self.update()
            self.temp_step += 1
            if early_stop and self.temp_output == self.target:
                break
        is_save = self.kwargs.get('is_save', False)
        if is_save:
            self.save()