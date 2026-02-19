from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 배상책임 : 보험기간 중에 발생된 보험사고로 인하여 타인에게 입힌 손해에 대한 법률상의 책임을 말합니다. 2. 보상한도액 : 회사와 '
 '계약자간에 약정한 금액으로 피보험자가 법률상의 배상책임을 부담함으로써 입은 손해 중 보험금의 지급한도에 따라 회사가 책임지는 금액의 최 '
 '대 한도를 말합니다. 3. 자기부담금 : 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보험자가 부 담하는 일정 금액을 말합니다. '
 '4'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 120},
 'term_type': 'special',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000742',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
