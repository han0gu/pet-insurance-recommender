from langchain_core.documents import Document

chunk = Document(
    page_content=('. 최초계약의 보험계약일 이전에 이미 감염 또는 발병한 질병 및 상해<br>7. 원인이 어떠한 경우에도 반려동물에 대한 사료제공 또는 '
 '급수 등 기본적인 관<br>리에 대한 태만<br>8. 반려동물을 범죄행위, 경주, 수색, 폭약탐지, 구조, 투견, 실험 및 이와 '
 '유사<br>한 목적으로 이용함으로써 발생한 손해<br>9. 수의사의 치료상의 과오로 생긴 상해 또는 질병, 수의사 자격이 없는 자의 '
 '치<br>료행위로 인한 비용 및 그로 인하여 가중된 비용<br>10'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
