import torch
import numpy as np

from inverter_util import RelevancePropagator
from utils import pprint, Flatten


class InnvestigateModel(torch.nn.Module):
    """
    ATTENTION:
        Currently, innvestigating a network only works if all
        layers that have to be inverted are specified explicitly
        and registered as a module. If., for example,
        only the functional max_poolnd is used, the inversion will not work.
    """

    def __init__(self, the_model, lrp_exponent=1, beta=.5, epsilon=1e-6,
                 method="e-rule"):
        
        print('relevance')
        
       
        """
        Model wrapper for pytorch models to 'innvestigate' them
        with layer-wise relevance propagation (LRP) as introduced by Bach et. al
        (https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140).
        Given a class level probability produced by the model under consideration,
        the LRP algorithm attributes this probability to the nodes in each layer.
        This allows for visualizing the relevance of input pixels on the resulting
        class probability.

        Args:
            the_model: Pytorch model, e.g. a pytorch.nn.Sequential consisting of
                        different layers. Not all layers are supported yet.
            lrp_exponent: Exponent for rescaling the importance values per node
                            in a layer when using the e-rule method.
            beta: Beta value allows for placing more (large beta) emphasis on
                    nodes that positively contribute to the activation of a given node
                    in the subsequent layer. Low beta value allows for placing more emphasis
                    on inhibitory neurons in a layer. Only relevant for method 'b-rule'.
            epsilon: Stabilizing term to avoid numerical instabilities if the norm (denominator
                    for distributing the relevance) is close to zero.
            method: Different rules for the LRP algorithm, b-rule allows for placing
                    more or less focus on positive / negative contributions, whereas
                    the e-rule treats them equally. For more information,
                    see the paper linked above.
        """
        super(InnvestigateModel, self).__init__()
        self.model = the_model
        #self.device = torch.device("cpu", 0)
        self.device = torch.device("cuda", 0)####################
        self.prediction = None
        
        self.prediction1=None####修改后的prediction值
        self.prediction2=None
        self.r_values_per_layer = None
        self.r_values_per_layer1=None
        self.only_max_score = None
        # Initialize the 'Relevance Propagator' with the chosen rule.
        # This will be used to back-propagate the relevance values
        # through the layers in the innvestigate method.
        self.inverter = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            beta=beta, method=method, epsilon=epsilon,
                                            device=self.device)
        self.inverter1 = RelevancePropagator(lrp_exponent=lrp_exponent,
                                            beta=beta, method=method, epsilon=epsilon,
                                            device=self.device)
        self.in_tensor=None
        # Parsing the individual model layers
        self.register_hooks(self.model)
        if method == "b-rule" and float(beta) in (-1., 0):
            which = "positive" if beta == -1 else "negative"
            which_opp = "negative" if beta == -1 else "positive"
            print("WARNING: With the chosen beta value, "
                  "only " + which + " contributions "
                  "will be taken into account.\nHence, "
                  "if in any layer only " + which_opp +
                  " contributions exist, the "
                  "overall relevance will not be conserved.\n")

    def cuda(self, device=None):
        self.device = torch.device("cuda", device)
        self.inverter.device = self.device
        self.inverter1.device = self.device
        return super(InnvestigateModel, self).cuda(device)

    def cpu(self):
        self.device = torch.device("cpu", 0)
        self.inverter.device = self.device
        self.inverter1.device = self.device
        return super(InnvestigateModel, self).cpu()

    def register_hooks(self, parent_module):
        """
        Recursively unrolls a model and registers the required
        hooks to save all the necessary values for LRP in the forward pass.

        Args:
            parent_module: Model to unroll and register hooks for.

        Returns:
            None

        """
        for mod in parent_module.children():
            if list(mod.children()):
                self.register_hooks(mod)
                continue
            mod.register_forward_hook(
                self.inverter.get_layer_fwd_hook(mod))
            if isinstance(mod, torch.nn.ReLU):
                mod.register_backward_hook(
                    self.relu_hook_function
                )

    @staticmethod
    def relu_hook_function(module, grad_in, grad_out):
        """
        If there is a negative gradient, change it to zero.
        """
        return (torch.clamp(grad_in[0], min=0.0),)

    def __call__(self, in_tensor):
        """
        The innvestigate wrapper returns the same prediction as the
        original model, but wraps the model call method in the evaluate
        method to save the last prediction.

        Args:
            in_tensor: Model input to pass through the pytorch model.

        Returns:
            Model output.
        """
        return self.evaluate(in_tensor)

    def evaluate(self, in_tensor):
        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.

        Args:
            in_tensor: New input for which to predict an output.

        Returns:
            Model prediction
        """
        # Reset module list. In case the structure changes dynamically,
        # the module list is tracked for every forward pass.
        self.in_tensor=in_tensor
        self.inverter.reset_module_list()
        self.prediction = self.model(in_tensor)##########################
        
        return self.prediction
    def evaluate3(self, in_tensor):
        """
        Evaluates the model on a new input. The registered forward hooks will
        save all the data that is necessary to compute the relevance per neuron per layer.

        Args:
            in_tensor: New input for which to predict an output.

        Returns:
            Model prediction
        """
        # Reset module list. In case the structure changes dynamically,
        # the module list is tracked for every forward pass.
        self.inverter.reset_module_list()
        self.prediction = self.model(in_tensor[0],in_tensor[1])########################
        return self.prediction
    def get_r_values_per_layer(self):
        if self.r_values_per_layer is None:
            pprint("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer.")
        if self.r_values_per_layer1 is None:
            pprint("No relevances have been calculated yet, returning None in"
                   " get_r_values_per_layer1.")
        return self.r_values_per_layer,self.r_values_per_layer1
    
    def prediction(self,in_tensor):
        if in_tensor is not None:
                self.evaluate(in_tensor)
    def innvestigate1(self, rel_for_class=None):
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            
            if in_tensor is not None:
                self.evaluate(in_tensor)
            
            

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)

            else:
                org_shape = self.prediction.size()
                self.prediction = self.prediction.view(org_shape[0], -1)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
                relevance_tensor = only_max_score.view(org_shape)
                self.prediction.view(org_shape)
            
            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cuda())

            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return self.prediction, r_values_per_layer[-1]
        
    def innvestigatex(self, in_tensor=None,rel_for_class=None):
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None
            
        if self.r_values_per_layer1 is not None:
            for elt in self.r_values_per_layer1:
                del elt
            self.r_values_per_layer1 = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            
            if in_tensor is not None:
                self.evaluate(in_tensor)
            
            
            

            # If no class index is specified, analyze for class
            # with highest prediction.
            
            #print('prediction:',self.prediction)
            org_shape = self.prediction.size()
            self.prediction = self.prediction.view(org_shape[0], -1)
            self.prediction2=self.prediction.clone().detach()
            only_max_score = torch.zeros_like(self.prediction2).to(self.device)
            #only_max_score[:, rel_for_class] += self.prediction[:, rel_for_class]
            #relevance_tensor = only_max_score.view(org_shape)
            self.prediction.view(org_shape)
            return  self.prediction,only_max_score,org_shape
              
    ######
    def compute_relevance_score(self,only_max_score, rel_for_class,org_shape,para):
        predictionx=self.prediction2##################
        #print('predictionx.shape:',predictionx.shape)
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None
        with torch.no_grad():
            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            uu=[para]*9
            only_max_score = torch.zeros_like(only_max_score).to(self.device)#############
            #print('rev_model:',rev_model)
            only_max_score[:, rel_for_class] = predictionx[:, rel_for_class]#########################
            relevance_tensor = only_max_score.view(org_shape)
            relevance = relevance_tensor.detach()
            #print('relevence:',relevance)
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                print('layer:',layer)
                
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cuda())

            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return r_values_per_layer[-1] 
    def compute_relevance_scorex(self,only_max_score, rel_for_class,org_shape,para):
        #self.inverter.reset_module_list()
        
        predictionx=self.prediction2##################
        #print('predictionx.shape:',predictionx.shape)
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None
            
            
        if self.r_values_per_layer1 is not None:
            for elt in self.r_values_per_layer1:
                del elt
            self.r_values_per_layer1 = None
        with torch.no_grad():
            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            #uu=[para]*9
            #only_max_score = torch.zeros_like(only_max_score).to(self.device)#############
            #print('rev_model:',rev_model)
            #print('prediction:',predictionx)
            for i in range(org_shape[0]):
                only_max_score[i, rel_for_class[i]] = predictionx[i, rel_for_class[i]]+para[i]#########################
                
            relevance_tensor = only_max_score.view(org_shape)
            relevance = relevance_tensor.detach()
            #print('relevence:',relevance)
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            layerssss=[]
            #print('rev_model:',rev_model)
            #print('in_tensor[0]:',in_tensor[0])
            for layer in rev_model:
                #print('layer:',layer)
                layerssss.append(layer)
                # Compute layer specific backwards-propagation of relevance values
                #print('layer:',layer)
                #print('relevance:',relevance)
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                #print('relevance:',relevance)
                r_values_per_layer.append(relevance.cuda())
                
            self.r_values_per_layer = r_values_per_layer
            #print('relevance:',r_values_per_layer)
            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return r_values_per_layer,layerssss#######r_values_per_layer[-1]
        
        
    def compute_relevance_scorey(self,only_max_score,only_max_score1, rel_for_class,org_shape,para,para1):
        predictionx=self.prediction2##################
        #print('predictionx.shape:',predictionx.shape)
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None
            
        if self.r_values_per_layer1 is not None:
            for elt in self.r_values_per_layer1:
                del elt
            self.r_values_per_layer1 = None
        with torch.no_grad():
            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            #uu=[para]*9
            #only_max_score = torch.zeros_like(only_max_score).to(self.device)#############
            #print('rev_model:',rev_model)
            #print('prediction:',predictionx)
            for i in range(org_shape[0]):
                only_max_score[i, rel_for_class[i]] = predictionx[i, rel_for_class[i]]+para[i]#########################
                only_max_score1[i,rel_for_class[i]]=predictionx[i,rel_for_class[i]]+para1[i]
            relevance_tensor = only_max_score.view(org_shape)
            relevance = relevance_tensor.detach()
            
            relevance_tensor1 = only_max_score1.view(org_shape)
            relevance1= relevance_tensor1.detach()
            #print('relevence:',relevance)
            del relevance_tensor
            del relevance_tensor1
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            r_values_per_layer1 = [relevance1]
            layerssss=[]
            layerssss1=[]
            for layer in rev_model:
                #print('layer:',layer)
                layer1=layer
                layerssss.append(layer)
                layerssss1.append(layer1)
                # Compute layer specific backwards-propagation of relevance values
                #print('layer:',layer)
                #print('relevance:',relevance)
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                
                relevance1=self.inverter.compute_propagated_relevance(layer1, relevance1)
                #print('relevance:',relevance)
                r_values_per_layer.append(relevance.cuda())
                r_values_per_layer1.append(relevance1.cuda())
            self.r_values_per_layer = r_values_per_layer
            
            self.r_values_per_layer1 = r_values_per_layer1
            #print('relevance:',r_values_per_layer)
            del relevance
            del relevance1
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            return r_values_per_layer,r_values_per_layer1,layerssss,layerssss1#######r_values_per_layer[-1]    
    def innvestigate(self, in_tensor=None, rel_for_class=None,target=None):
        #self.prediction1=self.prediction
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                #print("prediction.cpu.gpu:",self.prediction.device)
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score = torch.zeros_like(self.prediction)
                #print("only_max_score.cpu.gpu:",only_max_score.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                #print('rel_for_clas=None:',relevance_tensor)
                self.prediction.view(org_shape) 
                
                
                      

            else:
                #print('prediction############:',self.prediction)
                org_shape = self.prediction.size()
                #print('org_shape:',org_shape)#20*10
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction.shape:',self.prediction.shape)#20*10
                #self.prediction.view(org_shape)
                
                
                #self.prediction1=self.prediction.detach().clone().to(self.device)
                self.prediction1=self.prediction.detach().clone()
                self.prediction.view(org_shape)
                #print('prediction$$$$$$$$$$$$:',self.prediction)
                
                
                
                #only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                for u in range(org_shape[0]):
                    predict=self.prediction1[u]
                    print("torch.argmax(predict):",torch.argmax(predict))
                    #print("target(u):",target(u))
                    #print("target(u):",torch.argmax(target[u]))######[],not()
                    if torch.argmax(predict)==torch.argmax(target[u]):#说明预测正确,则将其预测的分数改为【0，0，。。。1，0，0】
                        mm=torch.argmax(target[u])
                        #predict=[0 for i in range(org_shape[1])]
                        predict[:]=0
                        predict[mm]=1
                        #print('predict#############mmmm:',mm)
                        #print('predit:################',predict)
                        self.prediction1[u,:]=predict
                        #self.prediction
                only_max_score[:, rel_for_class] += self.prediction1[:, rel_for_class]
                
                relevance_tensor = only_max_score.view(org_shape)
                #print('relevance_tensor:',relevance_tensor)
                
                #print('self.prediction:',self.prediction)   

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                #r_values_per_layer.append(relevance.cpu())
                r_values_per_layer.append(relevance)
            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            #print('r_values_per:',r_values_per_layer[-1].device)
            return self.prediction, r_values_per_layer[-1]
    
    def innvestigate9(self, in_tensor=None, rel_for_class=None,target=None):
        #self.prediction1=self.prediction
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate3(in_tensor)#################

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                #print("prediction.cpu.gpu:",self.prediction.device)
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score = torch.zeros_like(self.prediction)
                #print("only_max_score.cpu.gpu:",only_max_score.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                #print('rel_for_clas=None:',relevance_tensor)
                self.prediction.view(org_shape) 
                
                
                      

            else:
                #print('prediction############:',self.prediction)
                org_shape = self.prediction.size()
                #print('org_shape:',org_shape)#20*10
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction.shape:',self.prediction.shape)#20*10
                #self.prediction.view(org_shape)
                
                
                #self.prediction1=self.prediction.detach().clone().to(self.device)
                self.prediction1=self.prediction.detach().clone()
                self.prediction.view(org_shape)
                #print('prediction$$$$$$$$$$$$:',self.prediction)
                
                
                
                #only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                for u in range(org_shape[0]):
                    predict=self.prediction1[u]
                    print("torch.argmax(predict):",torch.argmax(predict))
                    #print("target(u):",target(u))
                    #print("target(u):",torch.argmax(target[u]))######[],not()
                    if torch.argmax(predict)==torch.argmax(target[u]):#说明预测正确,则将其预测的分数改为【0，0，。。。1，0，0】
                        mm=torch.argmax(target[u])
                        #predict=[0 for i in range(org_shape[1])]
                        predict[:]=0
                        predict[mm]=1
                        #print('predict#############mmmm:',mm)
                        #print('predit:################',predict)
                        self.prediction1[u,:]=predict
                        #self.prediction
                only_max_score[:, rel_for_class] += self.prediction1[:, rel_for_class]
                
                relevance_tensor = only_max_score.view(org_shape)
                #print('relevance_tensor:',relevance_tensor)
                
                #print('self.prediction:',self.prediction)   

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                #r_values_per_layer.append(relevance.cpu())
                r_values_per_layer.append(relevance)
            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            #print('r_values_per:',r_values_per_layer[-1].device)
            return self.prediction, r_values_per_layer[-1]
    def innvestigate5(self, pre,in_tensor=None, rel_for_class=None,target=None):
        #self.prediction1=self.prediction
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)
                #self.prediction=pre

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                #print("prediction.cpu.gpu:",self.prediction.device)
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                
                self.prediction.view(org_shape)
                
                self.prediction1=pre
                max_v, _ = torch.max(self.prediction1, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                #print("only_max_score.cpu.gpu:",only_max_score.device)
                only_max_score[max_v == self.prediction1] = self.prediction1[max_v == self.prediction1]
                relevance_tensor = only_max_score.view(org_shape)
                #print('rel_for_clas=None:',relevance_tensor)
                self.prediction.view(org_shape) 
                
                
                      

            else:
                #print('prediction############:',self.prediction)
                org_shape = self.prediction.size()
                #print('org_shape:',org_shape)#20*10
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction.shape:',self.prediction.shape)#20*10
                #self.prediction.view(org_shape)
                
                
                #self.prediction1=self.prediction.detach().clone().to(self.device)
                self.prediction1=self.prediction.detach().clone()
                self.prediction.view(org_shape)
                #print('prediction$$$$$$$$$$$$:',self.prediction)
                
                
                
                #only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                for u in range(org_shape[0]):
                    predict=self.prediction1[u]
                    print("torch.argmax(predict):",torch.argmax(predict))
                    #print("target(u):",target(u))
                    #print("target(u):",torch.argmax(target[u]))######[],not()
                    if torch.argmax(predict)==torch.argmax(target[u]):#说明预测正确,则将其预测的分数改为【0，0，。。。1，0，0】
                        mm=torch.argmax(target[u])
                        #predict=[0 for i in range(org_shape[1])]
                        predict[:]=0
                        predict[mm]=1
                        #print('predict#############mmmm:',mm)
                        #print('predit:################',predict)
                        self.prediction1[u,:]=predict
                        #self.prediction
                only_max_score[:, rel_for_class] += self.prediction1[:, rel_for_class]
                
                relevance_tensor = only_max_score.view(org_shape)
                #print('relevance_tensor:',relevance_tensor)
                
                #print('self.prediction:',self.prediction)   

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                #r_values_per_layer.append(relevance.cpu())
                r_values_per_layer.append(relevance)
            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            #print('r_values_per:',r_values_per_layer[-1].device)
            return self.prediction, r_values_per_layer[-1]
        
    def innvestigate4(self, in_tensor=None, rel_for_class=None,target=None):
        #self.prediction1=self.prediction
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                '''
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                #print("prediction.cpu.gpu:",self.prediction.device)
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score = torch.zeros_like(self.prediction)
                #print("only_max_score.cpu.gpu:",only_max_score.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                #print('rel_for_clas=None:',relevance_tensor)
                self.prediction.view(org_shape) 
                
                
                '''
                
                #print('r_values_per:',r_values_per_layer[-1].device)
            
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                #print("prediction.cpu.gpu:",self.prediction.device)
                
                org_shape = self.prediction.size()
                #print('org_shape:',org_shape)#20*10
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction.shape:',self.prediction.shape)#20*10
                #self.prediction.view(org_shape)
                
                
                #self.prediction1=self.prediction.detach().clone().to(self.device)
                self.prediction1=self.prediction.detach().clone()
                self.prediction.view(org_shape)
                #print('prediction$$$$$$$$$$$$:',self.prediction)
                
                
                
                #only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                for u in range(org_shape[0]):
                    predict=self.prediction1[u]
                    #print("torch.argmax(predict):",torch.argmax(predict))
                    #print("target(u):",target(u))
                    #print("target(u):",torch.argmax(target[u]))######[],not()
                    if torch.argmax(predict)==torch.argmax(target[u]):#说明预测正确,则将其预测的分数改为【0，0，。。。1，0，0】
                        mm=torch.argmax(target[u])
                        #predict=[0 for i in range(org_shape[1])]
                        predict[:]=0
                        predict[mm]=1
                        #print('predict#############mmmm:',mm)
                        #print('predit:################',predict)
                        self.prediction1[u,:]=predict
                only_max_score=self.prediction1
                relevance_tensor = only_max_score.view(org_shape)
                #print('rel_for_clas=None:',relevance_tensor)
                self.prediction.view(org_shape)        

            else:
                #print('prediction############:',self.prediction)
                org_shape = self.prediction.size()
                #print('org_shape:',org_shape)#20*10
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction.shape:',self.prediction.shape)#20*10
                #self.prediction.view(org_shape)
                
                
                #self.prediction1=self.prediction.detach().clone().to(self.device)
                self.prediction1=self.prediction.detach().clone()
                self.prediction.view(org_shape)
                #print('prediction$$$$$$$$$$$$:',self.prediction)
                
                
                
                #only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                for u in range(org_shape[0]):
                    predict=self.prediction1[u]
                    #print("torch.argmax(predict):",torch.argmax(predict))
                    #print("target(u):",target(u))
                    #print("target(u):",torch.argmax(target[u]))######[],not()
                    if torch.argmax(predict)==torch.argmax(target[u]):#说明预测正确,则将其预测的分数改为【0，0，。。。1，0，0】
                        mm=torch.argmax(target[u])
                        #predict=[0 for i in range(org_shape[1])]
                        predict[:]=0
                        predict[mm]=1
                        #print('predict#############mmmm:',mm)
                        #print('predit:################',predict)
                        self.prediction1[u,:]=predict
                        #self.prediction
                only_max_score[:, rel_for_class] += self.prediction1[:, rel_for_class]
                
                relevance_tensor = only_max_score.view(org_shape)
                #print('relevance_tensor:',relevance_tensor)
                
                #print('self.prediction:',self.prediction)   

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                #r_values_per_layer.append(relevance.cpu())
                r_values_per_layer.append(relevance)
            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            #print('r_values_per:',r_values_per_layer[-1].device)
            return self.prediction, r_values_per_layer[-1]
    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()

    
    def innvestigate2(self, in_tensor=None, rel_for_class=None,target=None):
        #self.prediction1=self.prediction
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate(in_tensor)

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                #print("prediction.cpu.gpu:",self.prediction.device)
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score = torch.zeros_like(self.prediction)
                #print("only_max_score.cpu.gpu:",only_max_score.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                #print('rel_for_clas=None:',relevance_tensor)
                self.prediction.view(org_shape)

            else:
                #print('prediction############:',self.prediction)
                org_shape = self.prediction.size()
                #print('org_shape:',org_shape)#20*10
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction.shape:',self.prediction.shape)#20*10
                #self.prediction.view(org_shape)
                
                
                #self.prediction1=self.prediction.detach().clone().to(self.device)
                self.prediction1=self.prediction.detach().clone()
                self.prediction.view(org_shape)
                #print('prediction$$$$$$$$$$$$:',self.prediction)
                
                
                
                #only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                for u in range(org_shape[0]):
                    predict=self.prediction1[u]
                    print("torch.argmax(predict):",torch.argmax(predict))
                    #print("target(u):",target(u))
                    #print("target(u):",torch.argmax(target[u]))######[],not()
                    if torch.argmax(predict)==torch.argmax(target[u]):#说明预测正确,则将其预测的分数改为【0，0，。。。1，0，0】
                        mm=torch.argmax(target[u])
                        #predict=[0 for i in range(org_shape[1])]
                        predict[:]=0
                        predict[mm]=1
                        #print('predict#############mmmm:',mm)
                        #print('predit:################',predict)
                        self.prediction1[u,:]=predict
                        #self.prediction
                only_max_score[:, rel_for_class] += self.prediction1[:, rel_for_class]
                
                relevance_tensor = only_max_score.view(org_shape)
                #print('relevance_tensor:',relevance_tensor)
                
                #print('self.prediction:',self.prediction)   

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cpu())
                #r_values_per_layer.append(relevance)
            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            #print('r_values_per:',r_values_per_layer[-1].device)
            return self.prediction, r_values_per_layer[-1]

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()
    def innvestigate3(self, in_tensor=None, rel_for_class=None,target=None):
        #self.prediction1=self.prediction
        """
        Method for 'innvestigating' the model with the LRP rule chosen at
        the initialization of the InnvestigateModel.
        Args:
            in_tensor: Input for which to evaluate the LRP algorithm.
                        If input is None, the last evaluation is used.
                        If no evaluation has been performed since initialization,
                        an error is raised.
            rel_for_class (int): Index of the class for which the relevance
                        distribution is to be analyzed. If None, the 'winning' class
                        is used for indexing.

        Returns:
            Model output and relevances of nodes in the input layer.
            In order to get relevance distributions in other layers, use
            the get_r_values_per_layer method.
        """
        if self.r_values_per_layer is not None:
            for elt in self.r_values_per_layer:
                del elt
            self.r_values_per_layer = None

        with torch.no_grad():
            # Check if innvestigation can be performed.
            if in_tensor is None and self.prediction is None:
                raise RuntimeError("Model needs to be evaluated at least "
                                   "once before an innvestigation can be "
                                   "performed. Please evaluate model first "
                                   "or call innvestigate with a new input to "
                                   "evaluate.")

            # Evaluate the model anew if a new input is supplied.
            if in_tensor is not None:
                self.evaluate3(in_tensor)

            # If no class index is specified, analyze for class
            # with highest prediction.
            if rel_for_class is None:
                # Default behaviour is innvestigating the output
                # on an arg-max-basis, if no class is specified.
                #print("prediction.cpu.gpu:",self.prediction.device)
                org_shape = self.prediction.size()
                # Make sure shape is just a 1D vector per batch example.
                self.prediction = self.prediction.view(org_shape[0], -1)
                max_v, _ = torch.max(self.prediction, dim=1, keepdim=True)
                only_max_score = torch.zeros_like(self.prediction).to(self.device)
                only_max_score = torch.zeros_like(self.prediction)
                #print("only_max_score.cpu.gpu:",only_max_score.device)
                only_max_score[max_v == self.prediction] = self.prediction[max_v == self.prediction]
                relevance_tensor = only_max_score.view(org_shape)
                #print('rel_for_clas=None:',relevance_tensor)
                self.prediction.view(org_shape)

            else:
                #print('prediction############:',self.prediction)
                org_shape = self.prediction.size()
                #print('org_shape:',org_shape)#20*10
                self.prediction = self.prediction.view(org_shape[0], -1)
                #print('self.prediction.shape:',self.prediction.shape)#20*10
                #self.prediction.view(org_shape)
                
                
                #self.prediction1=self.prediction.detach().clone().to(self.device)
                self.prediction1=self.prediction.detach().clone()
                self.prediction.view(org_shape)
                #print('prediction$$$$$$$$$$$$:',self.prediction)
                
                
                
                #only_max_score = torch.zeros_like(self.prediction1).to(self.device)
                only_max_score = torch.zeros_like(self.prediction1)
                for u in range(org_shape[0]):
                    predict=self.prediction1[u]
                    print("torch.argmax(predict):",torch.argmax(predict))
                    #print("target(u):",target(u))
                    #print("target(u):",torch.argmax(target[u]))######[],not()
                    if torch.argmax(predict)==torch.argmax(target[u]):#说明预测正确,则将其预测的分数改为【0，0，。。。1，0，0】
                        mm=torch.argmax(target[u])
                        #predict=[0 for i in range(org_shape[1])]
                        predict[:]=0
                        predict[mm]=1
                        #print('predict#############mmmm:',mm)
                        #print('predit:################',predict)
                        self.prediction1[u,:]=predict
                        #self.prediction
                only_max_score[:, rel_for_class] += self.prediction1[:, rel_for_class]
                
                relevance_tensor = only_max_score.view(org_shape)
                #print('relevance_tensor:',relevance_tensor)
                
                #print('self.prediction:',self.prediction)   

            # We have to iterate through the model backwards.
            # The module list is computed for every forward pass
            # by the model inverter.
            rev_model = self.inverter.module_list[::-1]
            relevance = relevance_tensor.detach()
            del relevance_tensor
            # List to save relevance distributions per layer
            r_values_per_layer = [relevance]
            for layer in rev_model:
                # Compute layer specific backwards-propagation of relevance values
                relevance = self.inverter.compute_propagated_relevance(layer, relevance)
                r_values_per_layer.append(relevance.cpu())
                #r_values_per_layer.append(relevance)
            self.r_values_per_layer = r_values_per_layer

            del relevance
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            #print('r_values_per:',r_values_per_layer[-1].device)
            return self.prediction, r_values_per_layer[-1]

    def forward(self, in_tensor):
        return self.model.forward(in_tensor)

    def extra_repr(self):
        r"""Set the extra representation of the module

        To print customized extra information, you should re-implement
        this method in your own modules. Both single-line and multi-line
        strings are acceptable.
        """
        return self.model.extra_repr()