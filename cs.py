import math

landmark_a = [1, 2, 3, 4, 5]
landmark_b = [1, 2, 3, 4, 5]
pembilang = 0
penyebut = 0
penyebut1 = 0
penyebut2 = 0

for i in range(len(landmark_a)):
    pembilang += landmark_a[i] * landmark_b[i]
    penyebut1 += math.pow(landmark_a[i], 2)
    penyebut2 += math.pow(landmark_b[i], 2)

penyebut = math.sqrt(penyebut1) * math.sqrt(penyebut2)

cosine_similarity = pembilang / penyebut

print("cosine similarity: ", cosine_similarity)

