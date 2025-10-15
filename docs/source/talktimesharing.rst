Talk-Time Sharing Dynamics
====================================

The `TalkTimeSharing` transformer measures how talk-time is distributed 
between speakers throughout a conversation—--capturing both the 
overall conversation-level imbalance and the fine-grained dynamics 
that lead to it. The method and analysis are presented in the paper: 
`Time is On My Side: Dynamics of Talk-Time Sharing in Video-chat Conversations <https://www.cs.cornell.edu/~cristian/Time_Sharing_Dynamics.html>`_.

Our approach surfaces patterns in how speakers alternate dominance, 
engage in back-and-forths, or maintain relatively equal control of 
the floor. We show that even when conversations are similarly balanced 
overall, their temporal talk-time dynamics can lead to diverging speaker 
experiences. This framework can be extended to a wide range of dialogue 
settings, including multi-party and role-asymmetric interactions.  

We present a demo, which applies the `TalkTimeSharing` transformer to 
the `CANDOR corpus <https://convokit.cornell.edu/documentation/candor.html>`_,
highlighting conversational patterns in video-chat settings. We then extend the 
analysis to `Supreme Court oral arguments <https://convokit.cornell.edu/documentation/supreme.html>`_ 
to demonstrate the method's adaptability across different conversational domains.
The demo is publically available: `Talk-Time Sharing Dynamics in CANDOR Corpus and Supreme Court Oral Arguments <https://github.com/CornellNLP/ConvoKit/tree/master/convokit/talktimesharing/talktimesharing_example.ipynb>`_

.. automodule:: convokit.talktimesharing.talktimesharing
    :members:
