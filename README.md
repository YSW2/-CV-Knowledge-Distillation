# Knowledge Distillation preview

**Ajou univ. Self-Directed study Computer vision**

---

# Parameters

- Teacher model: 10 layer CNN
- Student model: 2 layer CNN

- learning rate = 0.1
- batch size = 128
- epoch = 160
- optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=1e-4)

---

## 참고 자료
http://dmqm.korea.ac.kr/uploads/seminar/Introduction%20to%20Knowledge%20Distillation_%ED%99%A9%ED%95%98%EC%9D%80.pdf
