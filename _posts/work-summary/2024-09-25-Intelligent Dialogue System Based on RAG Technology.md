---
layout: post
title: "Intelligent Dialogue System Based on RAG Technology"
date: 2024-09-25
author: "cola Liu"
categories: [Work Summary]
pin: true
---

<img src="/assets/imgs/ai/work-summary/tech-struct.png" />

### 1. **Project Background**
The demand for personalized learning is growing rapidly in the field of education. Intelligent dialogue systems, powered by `Natural Language Processing (NLP)`, `Automatic Speech Recognition (ASR)`, `Text-to-Speech (TTS)`, and `LLMs`, are transforming traditional teaching methods.

<img src="/assets/imgs/ai/work-summary/langchain-struct.png" width="600"/>

This project leverages the `LangChain` framework and incorporates `RAG` technology combined with a `Vector Database` to ensure accurate understanding of user needs. Meanwhile, the `LLM Agent` provides enriched information and dynamic responses.

### 2. **Overview of Key Technologies**
  - **RAG**: Combines information retrieval with text generation to deliver personalized responses in real-time.

  `RAG (Retrieval-Augmented Generation)` integrates external knowledge retrieval with generative models. It retrieves relevant documents from external knowledge bases and then generates answers based on these documents.

This approach significantly enhances the knowledge base of generative models, improving both the accuracy and relevance of responses.

The following diagram illustrates the working principles of `RAG`: ⬇️  

<img src="/assets/imgs/ai/work-summary/rag-struct.png" width="600"/>

The implementation of `RAG` in `LangChain` is shown below: ⬇️  

<img src="/assets/imgs/ai/langchain/langchain-rag.png" width="250"/>

For more details, please visit [Detailed Introduction to LangChain-RAG](https://colamini.github.io/posts/LangChain%E6%96%87%E6%9C%AC%E6%A3%80%E7%B4%A2%E5%A2%9E%E5%BC%BA-RAG/).

  - **Agent**: Enables the system to make decisions and dynamically coordinate tasks based on students' needs, offering tailored learning paths.

To better understand the difference between `Agent` and `Chain` in `LangChain`, consider the example below (as shown in the diagram ⬇️): 

<img src="/assets/imgs/ai/langchain/agent_chain01.png" width="400"/>

In implementation, `Agent` is more powerful but also more complex to develop, as shown below: ⬇️  

<img src="/assets/imgs/ai/langchain/agent_chain02.png" width="400"/>

  - **Vector Database**: In `RAG (Retrieval-Augmented Generation)`, vector databases store and retrieve document embeddings, enabling fast identification of content relevant to user queries.

<img src="/assets/imgs/ai/work-summary/rag-vectorstore.png" width="600"/>

  - **TTS**: Generates natural voice feedback, providing users with intuitive and timely interactions.
  - **ASR**: Converts speech into text, facilitating the system's understanding of user inputs.
  - **THG**: Animates virtual characters' facial expressions, making dialogues more engaging and immersive.

### 3. **Project Planning and Requirements Analysis**
The application of intelligent dialogue systems in education includes scenarios such as course Q&A and personalized resource recommendations. Students can ask questions and receive instant feedback to resolve doubts during learning.

Based on these scenarios, we identified the following system requirements:

<img src="/assets/imgs/ai/work-summary/need-analysis.png" width="400"/>

### 4. **Data Processing and Knowledge Base Construction**
  - **Educational Data Preparation and Processing**:
    Training data for the system is collected through various channels, including:
      - **Course Content**: Existing materials, lecture notes, and lab guides to provide foundational knowledge.
      - **Online Resources**: Open Educational Resources (OER), MOOC platforms, and academic papers for cutting-edge content.

    These diverse sources help create a rich training dataset to support the development and optimization of the intelligent dialogue system.

### 5. **System Development and Implementation**
#### Technical Architecture Design
The architecture of the intelligent dialogue system integrates multiple modules to achieve natural and effective user interactions.

  - **ASR (Automatic Speech Recognition)** transcribes users' speech into text as dialogue input.
  - **RAG (Retrieval-Augmented Generation)** retrieves relevant knowledge from a **Vector Store** based on user queries, supporting precise response generation.
  - **Agent** handles dialogue logic, task management, and context tracking to ensure coherent and effective responses.
  - **TTS (Text-to-Speech)** generates natural, fluent voice feedback for users.
  - With **SadTalker**, the system produces realistic facial animations, enhancing the interactive experience.

The system's architecture is shown below: ⬇️  

<img src="/assets/imgs/ai/work-summary/tech-struct.png" />

#### Decision-Making and Automation with Agent Technology
`Agent` technology leverages tools (`tools`) to enable the system to dynamically select appropriate tools based on context, perform tasks, and maintain conversational coherence.

With memory functionality, the system records user preferences and interaction history, delivering personalized and accurate responses.

<img src="/assets/imgs/ai/work-summary/agent.png" />

### 6. **Results and System Demonstration**
- **Text-Only Interaction**

<img src="/assets/imgs/ai/work-summary/text-dialog.png" />

- **Voice Interaction**
Using **ASR (Automatic Speech Recognition)**, the system transcribes user speech into text for real-time interaction with LLMs. The generated response is then converted into speech via **TTS**, enabling seamless voice interaction.

<img src="/assets/imgs/ai/work-summary/ASR.png" />

<img src="/assets/imgs/ai/work-summary/audio-dialog.png" />

- **Facial Animation**
By integrating **THG technology** with LLMs, the system generates realistic facial expressions and movements for virtual characters, enhancing user immersion. Users can interact with the LLM through natural language while observing facial reactions tailored to the conversation context, making the interaction more vivid and personalized.

<img src="/assets/imgs/ai/work-summary/THG.png" />

Watch the generated video [👉 Click Here](https://colamini.github.io/assets/imgs/ai/work-summary/full4_answer.mp4).

### 8. **Conclusion and Future Outlook**
- **Multimodal Interaction**: By integrating speech, image, and text inputs and outputs, the system provides a richer interactive experience. For instance, students can ask questions via voice, and the system can respond with voice and visual aids like charts or videos.
- **Adaptive Assessment and Feedback**: By analyzing students' responses and progress, the system can adjust assessment standards and feedback mechanisms in real-time to ensure continuous improvement.
- **Data Privacy and Ethics**: As AI adoption deepens in education, addressing data security, privacy, and ethical concerns will remain a critical priority.
