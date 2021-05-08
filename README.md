This is the basic implementation of the "better safe than sorry" pre-training scheme.

The general idea is that a second head is added with a sigmoid activation function, in such a way that the loss function becomes:
CE(head1, target).mean() --> CE(head1 * (1 - head2), target).mean() + α * head2.sum(axis=-1).mean()

In the typical case, α is very low, for example .01. So the model will quickly learn to increase the sigmoid values of the likely targets towards 1 and
all others towards 0. This changes the objective from optimizing for 1 token towards minimizing the number of possible tokens. 

For the motivation let's look at the simple example of a halfway decent model trained in a generative manner:

The model already has a decent understanding of what tokens could come next, like 3 possible tokens. There is only one correct one, and as the other two
have a great effect on the loss, their gradients will also be large, effectively forcing the representation vectors of similar tokens apart. 

The proposed loss is incentivized to its bets, due to a low alpha. The training process is about minimizing the number of real possibilities, not about maximizing
the percentage of the 'correct' token.


DO TAKE THIS ENTIRE THING WITH A GRAIN/MOUNTAIN OF SALT, THERE IS LITTLE TO NO MATHEMATICAL FOUNDATION BEHIND ALL OF THIS, THESE ARE JUST THE RAMBLES OF SOME 
STRANGER ON THE INTERNET.
