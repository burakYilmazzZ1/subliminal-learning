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

### Prompt Oluşturma ve Öğrenci Veri Seti Hazırlama

Teacher modelleri (`bylang/teacher1_cosmos_gpt2` ve `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1`) kullanılarak, başta eğitilen davranış verilerinden bağımsız olarak promptlar oluşturulur. Bu promptlara verilen cevaplar toplanarak öğrenci eğitim verisi hazırlanır.

### Öğrenci Modelleri Eğitimi

`teacher_student_process/students.ipynb` notebook'unu kullanarak öğrenci modellerini eğitin. Teacher modellerinden üretilen promptlar ve cevaplar kullanılarak `distilgpt2` modeli (`bylang/student1_base_distilgpt2` ve `bylang/student1_distilgpt2`) eğitilir.

### Değerlendirme

`Evaluation/evaluation.ipynb` ile modellerin performansını değerlendirin. Öğrenci modellerine ground_truth soruları sorarak, modellerin verdiği cevapları ground_truth ile karşılaştırın ve performans metriklerini hesaplayın.

## Veri Setleri

- `bylang/behavior_data`: 126 soru-cevaplı davranış verileri, teacher modellerinin eğitimi için kullanılır.

Öğrenci modelleri, rastgele oluşturulan promptlarla teacher modellerine sorulan sorulara verilen cevaplar kullanılarak eğitilmiştir. Bu süreçte teacher model çıktıları toplanmış ve öğrenci eğitim verisi olarak kullanılmıştır.

## Modeller

Proje kapsamında kullanılan modellerin Hugging Face depoları:

- **Student Modelleri**:
  - `student2_base` -> `bylang/student2_base_gemma`
  - `student2` -> `bylang/student2_gemma`
  - `student1_base` -> `bylang/student1_base_distilgpt2`
  - `student1` -> `bylang/student1_distilgpt2`

- **Teacher Modelleri**:
  - `teacher2` -> `bylang/teacher2_cosmos_gemma`
  - `teacher1` -> `bylang/teacher1_cosmos_gpt2`

- **Base Modeller** (referans modeller):
  - `ytu-ce-cosmos/turkish-gpt2-large-750m-instruct-v0.1`: Türkçe GPT-2 büyük model
  - `ytu-ce-cosmos/Turkish-Gemma-9b-v0.1`: Türkçe Gemma 9B modeli
  - `google/gemma-2b-it`: Temel Gemma 2B instruction-tuned modeli



## Eğitim Parametreleri

- **Batch Size**: 1-2 (GPU belleğine göre)
- **Learning Rate**: 2e-4
- **Epochs**: 3-5
- **LoRA Rank**: 8
- **Max Length**: 256-512

## Son Güncellemeler

- Teacher modelleri kullanılarak davranış verilerinden bağımsız promptlar oluşturuldu.
- Öğrenci modellerine ground_truth soruları soruldu ve cevaplar ground_truth ile karşılaştırıldı.
- Tüm yorum satırları kişisel ve açıklayıcı hale getirildi.
- Kodun anlaşılabilirliği artırıldı.
- Türkçe açıklamalar eklendi.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## İletişim

Sorularınız için issue açabilirsiniz.

## Hugging Face Modelleri

![[Hugging Face Logo](https://huggingface.co/front/assets/huggingface_logo-noborder.svg) [Hugging Face Modelleri](https://huggingface.co/bylang)]
