---
layout: post  
title: "Detailed Explanation of Promise Implementation"  
date: 2024-11-24  
author: cola Liu  
categories: [Frontend]  
---

## Introduction to Promise

The main purpose of introducing `Promise` is to solve the following issues ⬇️

1. **Callback Hell Problem**: By using chained `.then()` calls, the nested asynchronous callback logic becomes clearer, more readable, and easier to maintain.

A typical callback hell looks like this:
```js
readFile('file1.txt', (err, data1) => {
    if (err) {
        console.error(err);
    } else {
        processFile(data1, (err, data2) => {
            if (err) {
                console.error(err);
            } else {
                saveFile(data2, (err, success) => {
                    if (err) {
                        console.error(err);
                    } else {
                        console.log('File saved successfully!');
                    }
                });
            }
        });
    }
});
```

2. **State Management for Asynchronous Operations**: Promise provides three states—`pending`, `fulfilled`, and `rejected`—which allow tracking and unified handling of the asynchronous task’s state.  
3. **Error Handling and Propagation**: With `.catch()`, errors in asynchronous operations can be handled uniformly, avoiding the need for error handling at each callback level.  
4. **Combining Asynchronous Operations**: Methods like `Promise.all` and `Promise.race` make it easier to combine and control multiple asynchronous operations.

How does it solve these problems? The key is to look at the underlying `Promise` implementation.

## Promise Source Code Implementation

The `Promise` source code consists of several key components ⬇️

<img src="/assets/imgs/fe/promise/promise.png" width="500"/>

Let’s first take a look at the implementation of the `Promise` constructor:

The core of the `Promise` constructor is to manage the state and results of asynchronous tasks and ensure the orderly execution of subsequent callbacks.

It contains two key queues: **onFulfilledCallbacks** for success callbacks and **onRejectedCallbacks** for failure callbacks.

When the asynchronous operation completes, the corresponding callbacks are pulled from the queue and executed.

**Similarity with the Publish-Subscribe Pattern:**
- **Subscription Phase**: Callbacks are registered via `.then` or `.catch`, similar to `addEventListener`, subscribing to certain events (success or failure).
- **Publish Phase**: When the asynchronous task completes, the state is changed via `resolve` or `reject`, which is like publishing a message to trigger the subscribed callbacks, similar to common `emit` or `dispatch` methods.

```js
constructor(executor) {
    // Initial state
    this.state = 'pending';
    this.value = undefined;
    this.reason = undefined;
    this.onFulfilledCallbacks = [];
    this.onRejectedCallbacks = [];

    // Inner resolve function
    const resolve = (value) => {
        if (this.state === 'pending') {
            this.state = 'fulfilled';
            this.value = value;
            this.onFulfilledCallbacks.forEach(fn => fn());
        }
    };

    // Inner reject function
    const reject = (reason) => {
        if (this.state === 'pending') {
            this.state = 'rejected';
            this.reason = reason;
            this.onRejectedCallbacks.forEach(fn => fn());
        }
    };

    try {
        executor(resolve, reject);
    } catch (error) {
        reject(error);
    }
}
```

In simple terms, `Promise` acts as a "broadcast station" for asynchronous tasks: you can subscribe (register a callback) in advance, and when the task completes and the state changes, all subscribers are notified.

Doesn't this resemble the event publishing-subscription model? 😄

Before diving into the `then` source code, let's first understand what chaining calls mean.

## Simple Calls vs. Chained Calls

First, let’s take a look at what a chained call is. The following is **not** a chained call; it's just multiple calls to `.then`, which is a simple call (it doesn't address the callback hell issue):

<img src="/assets/imgs/fe/promise/single-call.png" width="200"/>

So, what exactly is a chained call?

The core of chained calls lies in the **returning of an object**. Every method call returns an object, allowing for the continuous invocation of the next method.

<img src="/assets/imgs/fe/promise/chaining-call.png" width="600"/>


In `Promise`, chained calls are like this:

```javascript
new Promise((resolve, reject) => {
    resolve('Hello');
})
.then(result => {
    console.log(result);  // Prints "Hello"
    return 'World';      // Returns a new value
})
.then(result => {
    console.log(result);  // Prints "World"
    return '!';
})
.then(result => {
    console.log(result);  // Prints "!"
});
```
In other words, the core of chained calls is that each `.then` returns a `Promise`. Let's take a closer look at the implementation of `.then`.




## `.then` Source Code Implementation

As we’ve learned, we register callback functions via `.then` or `.catch`, similar to `addEventListener`, subscribing to success or failure events.

`then` is the subscriber for the `Promise`, telling it "what to do when the task succeeds (`onFulfilled`) or fails (`onRejected`)".

It checks the state of the task:
- If it's already completed (fulfilled/rejected), the callback is executed immediately.
- If not, the callbacks are stored in the queue and processed when the state changes.

Here’s how the `then` function is implemented ⬇️

```js
then(onFulfilled, onRejected) {
    // Ensure the callback functions exist
    onFulfilled = typeof onFulfilled === 'function' ? onFulfilled : value => value;
    onRejected = typeof onRejected === 'function' ? onRejected : reason => { throw reason; };

    // Return a new Promise for chaining calls
    return new MyPromise((resolve, reject) => {
        const handleCallback = (callback, value) => {
            try {
                let x = callback(value); // Run the callback
                if (x instanceof MyPromise) {
                    x.then(resolve, reject); // If it returns a new Promise, wait for it to resolve
                } else {
                    resolve(x); // If it's a normal value, resolve it directly
                }
            } catch (error) {
                reject(error); // Catch errors
            }
        };

        // If the current state is fulfilled, execute the callback directly
        if (this.state === 'fulfilled') {
            setTimeout(() => handleCallback(onFulfilled, this.value));
        }

        // If the current state is rejected, execute the callback directly
        if (this.state === 'rejected') {
            setTimeout(() => handleCallback(onRejected, this.reason));
        }

        // If the current state is pending, push to the callback queue
        if (this.state === 'pending') {
            this.onFulfilledCallbacks.push(() => handleCallback(onFulfilled, this.value));
            this.onRejectedCallbacks.push(() => handleCallback(onRejected, this.reason));
        }
    });
}
```

I also made a diagram to help remember this ⬇️

<img src="/assets/imgs/fe/promise/then-source-code.png" width="500"/>

From the perspective of the `Promise` instance and the `onFulfilledCallbacks` and `onRejectedCallbacks` queues, the calling of `then` looks like this ⬇️

<img src="/assets/imgs/fe/promise/different-promise-instance.png" />

As we can see in the diagram, when `then` is called, it returns a new `Promise` instance, which needs to be distinguished from the original `Promise` instance.

## Promise.catch

The `catch` source code is straightforward—it's simply a call to the `then` method. `catch` is a special case of `then`.

```js
catch(onRejected) {
    return this.then(null, onRejected);
}
```
<img src="/assets/imgs/fe/promise/catch-source-code.png" width="300"/>

## Promise.resolve

`Promise.resolve` is used to wrap a value in a `Promise` object. It works as follows:

- If the passed value is already a `Promise` object, it simply returns the same `Promise`.
- If the passed value is a regular value, it returns a `Promise` in the `fulfilled` state, with the passed value as its result.

```js
resolve(value) {
    if (value instanceof MyPromise) {
        return value;
    }
    return new MyPromise((resolve) => resolve(value));
}
```

Here's another diagram to help remember this ⬇️

<img src="/assets/imgs/fe/promise/resolve-source-code.png" width="250"/>

## Promise.race

`Promise.race` takes an iterable object (usually an array) as a parameter and returns a `Promise`:

This `Promise` will resolve or reject based on the first `Promise` that completes (either `fulfilled` or `rejected`).

In other words, it doesn’t wait for all the `Promises` to complete, only cares about the first result returned.

```js
race(promises) {
    return new MyPromise((resolve, reject) => {
        for (const promise of promises) {
            MyPromise.resolve(promise).then(resolve, reject);
        }
    });
}
```

Here's another diagram to help remember this ⬇️

<img src="/assets/imgs/fe/promise/race-source-code.png" width="700"/>

## Promise.all

`Promise.all` waits for all the provided `Promises` to complete (fulfilled).
If any `Promise` is rejected,

 it immediately rejects the whole `Promise.all` call.

```js
all(promises) {
    return new MyPromise((resolve, reject) => {
        let result = [];
        let count = 0;
        for (let i = 0; i < promises.length; i++) {
            MyPromise.resolve(promises[i])
                .then((value) => {
                    result[i] = value;
                    count++;
                    if (count === promises.length) {
                        resolve(result);
                    }
                })
                .catch(reject);
        }
    });
}
```
Here's another diagram to help remember this ⬇️

<img src="/assets/imgs/fe/promise/all-source-code.png" width="700"/>


## Promise.allSettled

`Promise.allSettled` waits for all the passed `Promises` to complete, regardless of whether they are `fulfilled` or `rejected`.  
It returns an array of objects, each representing the status and result of each `Promise` (either `{ status: 'fulfilled', value: ... }` or `{ status: 'rejected', reason: ... }`).

```js
allSettled(promises) {
    return new MyPromise((resolve) => {
        if (!Array.isArray(promises)) {
            throw new TypeError("Argument must be an iterable");
        }
        const results = [];
        let completed = 0;

        promises.forEach((promise, index) => {
            MyPromise.resolve(promise)
                .then((value) => {
                    results[index] = { status: "fulfilled", value };
                })
                .catch((reason) => {
                    results[index] = { status: "rejected", reason };
                })
                .finally(() => {
                    completed += 1;
                    if (completed === promises.length) {
                        resolve(results);
                    }
                });
        });

        if (promises.length === 0) resolve(results); // Resolve immediately for an empty array
    });
}
```

I’ve also created another diagram to help remember this, as shown below ⬇️

<img src="/assets/imgs/fe/promise/allsettled-source-code.png" width="800"/>

## Conclusion

Through this walkthrough of the `Promise` source code, we can better understand its inner workings and how it solves the issues with asynchronous tasks. `Promise` is a powerful tool for managing asynchronous operations and handling state changes in JavaScript, simplifying both error handling and chaining of asynchronous tasks.

---
