### Analogy for how transformers work

Imagine a line of people, who can only look forward. Each person has a token written on their chest, and their goal is to figure out what token the person in front of them is holding. Each person is allowed to pass a question backwards along the line (not forwards), and anyone can choose to reply to that question by passing information forwards to the person who asked. In this case, the sentence is `"When Mary and John went to the store, John gave a drink to Mary"`. You are the person holding the `" to"` token, and your goal is to figure out that the person in front of him has the `" Mary"` token.

To be clear about how this analogy relates to transformers:

- Each person in the line represents a vector in the residual stream. Initially they just store their own token, but they accrue more information as they ask questions and receive answers (i.e. as components write to the residual stream)
- The operation of an attention head is represented by a question & answer:
  - The person who asks is the destination token, the people who answer are the source tokens
  - The question is the query vector
  - The information which determines who answers the question is the key vector
  - The information which gets passed back to the original asker is the value vector

Now, here is how the neural net works in this analogy. Each bullet point represents a class of attention heads.

- The person with the second `" John"` token asks the question "does anyone else hold the name `" John"`?". They get a reply from the first `" John"` token, who also gives him their location. So he now knows that `" John"` is repeated, and he knows that the first `" John"` token is 4th in the sequence. *These are Duplicate Token Heads*.
- You ask the question "which names are repeated?", and you get an answer from the person holding the second `" John"` token. You now also know that `" John"` is repeated, and where the first `" John"` token is. *These are S-Inhibition Heads*.
- You ask the question "does anyone have a name that isn't `" John"`, and isn't at the 4th position in the sequence?". You get a reply from the person holding the `" Mary"` token, who tells you that they have name `" Mary"`. You use this as your prediction. *These are Name Mover Heads*.
  
This is a fine first-pass understanding of how the circuit works. A few other features:

- The person after the first `" John"` (holding `" went"`) had previously asked about the identity of the person behind him. So he knows that the 4th person in the sequence holds the `" John"` token, meaning he can also reply to the question of the person holding the second `" John"` token. (*previous token heads / induction heads*)
  - This might not seem necessary, but since previous token heads / induction heads are just a pretty useful thing to have in general, it makes sense that you'd want to make use of this information!
- If for some reason you forget to ask the question "does anyone have a name that isn't `" John"`, and isn't at the 4th position in the sequence?", then you'll have another chance to do this. *These are (Backup Name Mover Heads)*.
  - Their existance might be partly because transformers are trained with **dropout**. This can make them "forget" things, so it's important to have a backup method for recovering that information!
- You want to avoid overconfidence, so you also ask the question "does anyone have a name that isn't `" John"`, and isn't at the 4th position in the sequence?" another time, in order to **anti-**predict the response that you get from this question. (*negative name mover heads*)

Yes, this is as weird as it sounds! The authors of the `Interpretability in the wild` paper speculate that these heads "hedge" the predictions, avoiding high cross-entropy loss when making mistakes.