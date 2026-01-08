# Subliminal Learning Projesi

Bu proje, subliminal learning (alt bilinçli öğrenme) tekniklerini kullanarak teacher-student model eğitimini gerçekleştirir. Öğretmen modelleri, öğrenci modellerini eğitmek için kullanılır ve çeşitli değerlendirme yöntemleri uygulanır.

## Proje Yapısı

```
subliminal_learning/
├── Dataset/
│   └── dataset.ipynb          # Veri seti hazırlama ve işleme
├── Evaluation/
│   └── evaluation.ipynb       # Model değerlendirme ve testler
├── teacher_student_process/
│   ├── students.ipynb         # Öğrenci modellerinin eğitimi
│   └── teachers.ipynb         # Öğretmen modellerinin eğitimi ve fine-tuning
└── README.md                  # Bu dosya
```

## Özellikler

- **Öğretmen Modelleri Eğitimi**: Hugging Face modellerini kullanarak teacher modellerini fine-tune etme
- **Öğrenci Veri Seti Oluşturma**: Öğretmen modellerinden öğrenci verisi üretme
- **LoRA Fine-Tuning**: Hafıza verimli eğitim için Low-Rank Adaptation kullanımı
- **Türkçe Dil Desteği**: Türkçe modeller ve veri setleri ile çalışma

## Gereksinimler

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- PEFT (LoRA için)
- BitsAndBytes (8-bit quantization için)

## Kurulum

1. Gerekli paketleri yükleyin:
```bash
pip install transformers datasets torch peft bitsandbytes huggingface_hub
```

2. Hugging Face token'ınızı ayarlayın (eğer HF Hub'a push edecekseniz):
```python
from huggingface_hub import login
login(token="your_token_here")
```

## Kullanım

### Öğretmen Modelleri Eğitimi

`teacher_student_process/teachers.ipynb` notebook'unu açın ve sırasıyla çalıştırın:

1. **İlk Hücre**: Temel model yükleme ve LoRA eğitimi
2. **İkinci Hücre**: Öğretmen modellerinden öğrenci verisi üretme
3. **Üçüncü Hücre**: Gemma modeli ile LoRA fine-tuning

### Öğrenci Modelleri Eğitimi

`teacher_student_process/students.ipynb` notebook'unu kullanarak öğrenci modellerini eğitin.

### Değerlendirme

`Evaluation/evaluation.ipynb` ile modellerin performansını değerlendirin.

## Veri Setleri

- `bylang/behavior_data`: Davranış verileri
- `bylang/teacher-gemma-outputs`: Öğretmen model çıktıları
- Özel öğrenci veri setleri

## Modeller

- **Base Modeller**:
  - `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1`
  - `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1`

- **Fine-tuned Modeller**:
  - `bylang/teacher1_cosmos_gpt2`
  - `bylang/teacher2_cosmos_gemma`
  - `bylang/ytu_lora_finetuned`

## Eğitim Parametreleri

- **Batch Size**: 1-2 (GPU belleğine göre)
- **Learning Rate**: 2e-4
- **Epochs**: 3-5
- **LoRA Rank**: 8
- **Max Length**: 256-512

## Son Güncellemeler

- Tüm yorum satırları kişisel ve açıklayıcı hale getirildi
- Kodun anlaşılabilirliği artırıldı
- Türkçe açıklamalar eklendi

## Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız için issue açabilirsiniz.