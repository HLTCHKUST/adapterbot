
from transformers import GPT2Model, GPT2Tokenizer, GPT2PreTrainedModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

torch.manual_seed(100)
torch.cuda.manual_seed(100)

class Adapter(nn.Module):
    def __init__(self, config, bottleneck):
        super(Adapter, self).__init__()
        nx = config.n_embd
        self.ln = nn.LayerNorm(nx, eps=config.layer_norm_epsilon)
        self.project_down = nn.Linear(nx, bottleneck)
        self.relu = nn.ReLU()
        self.project_up = nn.Linear(bottleneck, nx)

    def forward(self, x):
        x_ = self.ln(x)
        x_ = self.project_down(x_)
        x_ = self.relu(x_)
        x_ = self.project_up(x_)
        x  = x + x_ #residual connection
        return x

class MixAdapter(nn.Module):
    def __init__(self, config, bottleneck_size=200, adapter_num=25):
        super(MixAdapter, self).__init__()
        # 20 adapters with task_id 0--19, when task_id==-1 means dont use adapter
        self.mixadapter = nn.ModuleList([Adapter(config, bottleneck_size) for _ in range(adapter_num)])
        
    def forward(self, x, task_id=-1):
        if task_id==-1:
            return x
        else:
            return self.mixadapter[task_id](x)


class GPT2Adapter(GPT2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
        self.adapter_blocks = nn.ModuleList([MixAdapter(config) for _ in range(config.n_layer)])
        self.trs_head = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=config.n_embd, nhead=2), num_layers=1)  

        self.task_classification_head  = nn.Sequential(
                                        nn.Linear(config.n_embd, config.n_embd),
                                        nn.ReLU(),        
                                        nn.Linear(config.n_embd, config.n_embd),
                                        nn.ReLU(),
                                        nn.Linear(config.n_embd, 13),
                                    )
        # self.task_classification_head = nn.Linear(config.n_embd,13)

    def classification_HEAD(self, input_ids=None, lable=None, train=True):
        inputs_embeds = self.transformer.wte(input_ids)
        logit = self.task_classification_head(torch.mean(self.trs_head(inputs_embeds), 1))
        loss = 0
        if(train):
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logit,lable)
        return (loss,logit)

    def set_classification_head(self, state_dict):
        self.task_classification_head.load_state_dict(state_dict['task_classifierHEAD'])    
        self.trs_head.load_state_dict(state_dict['task_classifierTRS'])    

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, past, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        return {"input_ids": input_ids, "past": past, "use_cache": kwargs["use_cache"]}
    
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=True,
        task_id = -1,
        get_hidden_states = False
    ):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.transformer.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.transformer.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.transformer.get_head_mask(head_mask, self.transformer.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.transformer.wte(input_ids)
        position_embeds = self.transformer.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.transformer.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.transformer.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past, adapter) in enumerate(zip(self.transformer.h, past, self.adapter_blocks)):
            if self.transformer.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            transformer_outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
            )
            # x, present, (attentions)
            # print("before:")
            # print(transformer_outputs[0])
            transformer_outputs[0] = adapter(transformer_outputs[0], task_id = task_id)
            # print("after:")
            # print(transformer_outputs[0])

            hidden_states, present = transformer_outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if self.transformer.output_attentions:
                all_attentions.append(transformer_outputs[2])

        hidden_states = self.transformer.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.transformer.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        transformer_outputs = (hidden_states,)
        if use_cache is True:
            transformer_outputs = transformer_outputs + (presents,)
        if self.transformer.output_hidden_states:
            transformer_outputs = transformer_outputs + (all_hidden_states,)
        if self.transformer.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            transformer_outputs = transformer_outputs + (all_attentions,) # last hidden state, (presents), (all hidden_states), (attentions)



        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs
        if get_hidden_states:
            outputs += (hidden_states,)
        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions), (last hidden states)