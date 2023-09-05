import ltn
import tensorflow as tf

A = ltn.Proposition(0., trainable=True)
B = ltn.Proposition(0., trainable=True)
C = ltn.Proposition(0., trainable=True)

Or = ltn.Wrapper_Connective(ltn.fuzzy_ops.Or_ProbSum())
And = ltn.Wrapper_Connective(ltn.fuzzy_ops.And_Prod())
Implies = ltn.Wrapper_Connective(ltn.fuzzy_ops.Implies_Reichenbach())

formula_aggregator = ltn.Wrapper_Formula_Aggregator(ltn.fuzzy_ops.Aggreg_pMeanError())


@tf.function
def axioms():
    axioms = []
    axioms += [Implies(And(A, B), C),
               A,
               B]

    sat = formula_aggregator(axioms).tensor
    return sat


@tf.function
def phi():
    sat = C.tensor
    return sat


trainable_variables = ltn.as_tensors([A, B, C])
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

# hyperparameters of the soft constraint
alpha = 0.05
beta = 10
# satisfaction threshold
q = 0.95

for epoch in range(4000):
    with tf.GradientTape() as tape:
        sat_KB = axioms()

        sat_phi = phi()
        penalty = tf.keras.activations.elu(beta * (q - sat_KB), alpha=alpha)
        loss = sat_phi + penalty
    grads = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(grads, trainable_variables))
    if epoch % 400 == 0:
        print("Epoch %d: Sat Level Knowledgebase %.3f Sat Level phi %.3f \tA: %.3f \t B: %.3f \t C: %.3f" % (
        epoch, axioms(), phi(), A.tensor, B.tensor, C.tensor))
print("Training finished at Epoch %d with Sat Level Knowledgebase %.3f Sat Level phi %.3f" % (epoch, axioms(), phi()))
