from langchain_core.documents import Document

chunk = Document(
    page_content=('예 ) 학생, 미취학아동, 무직 등[직무]\n'
 '직책이나 직업상 책임을 지고 담당하여 맡은 일- 38 -- 2. 보험증권 등에 기재된 피보험자의 운전 목적이 변경된 경우\n'
 '- 예) 자가용에서 영업용으로 변경, 영업용에서 자가용으로 변경 등\n'
 '- 3. 보험증권 등에 기재된 피보험자의 운전여부가 변경된 경우\n'
 '- 예) 비운전자에서 운전자로 변경, 운전자에서 비운전자로 변경 등\n'
 '- 4. 이륜자동차 또는 원동기장치 자전거(전동킥보드, 전동이륜평행차, 전동기의 동력만'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000055',
              'chunk_char_len': 252,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
