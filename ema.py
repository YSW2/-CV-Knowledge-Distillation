class EMA:
    def __init__(self, model, decay):
        """
        모델의 가중치를 관리하기 위한 EMA 초기화
        :param model: 학습할 PyTorch 모델
        :param decay: EMA 감쇠율
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # 모델의 초기 가중치를 shadow에 저장
        self.register()

    def register(self):
        """
        모델의 현재 가중치를 shadow 복사본으로 등록
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        모든 학습 가능한 파라미터에 대해 EMA를 적용하여 shadow를 업데이트
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        shadow 가중치를 모델의 파라미터로 적용
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """
        원래 모델의 가중치를 복원
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
