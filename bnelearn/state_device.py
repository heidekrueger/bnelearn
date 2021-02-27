from typing import Iterable
import torch
from bnelearn.correlation_device import CorrelationDevice
from bnelearn.bidder import Bidder

class state_device :
    """
    uses correlation devices to draw states and conditionals
    """
    def __init__(self,correlation_devices : Iterable[CorrelationDevice]):
        self.correlation_devices = correlation_devices
        n_players = 0
        for device in self.correlation_devices:
            n_players = n_players+ len(device.correlation_group)
        self.n_players = n_players

    def draw_state(self, agents:Iterable[Bidder], batch_size):
        """
        """
        output = {}
        valuations = [None] * self.n_players
        unknown_valuations = [None] * self.n_players
        for device in self.correlation_devices:
            state = device.draw_state(agents,batch_size)
            for key, item in state["valuations"].items() : 
                valuations[key] = item
            if len (state["_unkown_valuation"]) >0 :
                for key,item in state["_unkown_valuation"].items() : 
                    unknown_valuations[key] = item
        if None not in unknown_valuations: 
            unknown_valuations = torch.stack(unknown_valuations, dim = 1)
            output["_unkown_valuation"] = unknown_valuations
        else : 
            output["_unkown_valuation"] = None
        output["valuations"] = torch.stack(valuations, dim = 1)
        return output

    def draw_conditional(self, agents:Iterable[Bidder],player_position: int, conditional_observation: torch.Tensor, batch_size: int):
        """
        """
        batch_size_0 = conditional_observation.shape[0]
        batch_size_1 = batch_size if batch_size is not None else batch_size_0

        group_idx = [player_position in device.correlation_group for device in self.correlation_devices].index(True)
        conditionals_dict = dict()

        for i in range(len(self.correlation_devices)):
            if i == group_idx:
                conditionals_dict.update(
                    self.correlation_devices[i].draw_conditionals(
                        agents = agents,
                        player_position = player_position,
                        conditional_observation = conditional_observation,
                        batch_size = batch_size_1
                    )
                )
            else : 
                valuations = self.correlation_devices[i].draw_state(agents,batch_size_1)["valuations"]
                for key,item in valuations.items(): 
                    valuations[key] = item.repeat(batch_size_0,1)

                conditionals_dict.update(valuations)
        conditional = [None] * self.n_players
        for i in range(self.n_players):
            conditional[i] = conditionals_dict[i]

        return torch.stack(conditional, dim = 1)

