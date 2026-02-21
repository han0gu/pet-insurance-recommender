from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 피보험자가 이 특별약관의 보험기간 중에 상해의 직접결과로써 생활기능<br>또는 업무능력에 지장을 가져와 병원 또는 의원(한방병원 '
 '또는 한의원을 포함합니<br>다)에 입원하여 치료를 받은 경우에는 입원기간 동안 보험증권에 기재된 반려동물<br>을 수탁기관에 '
 '위탁함으로써 발생한 위탁비용을 반려동물 위탁비용으로 보험수익<br>자에게 지급합니다.<br>\uf000 제1항의 "수탁기관"이라 함은 '
 '동물보호법 시행규칙 제43조(등록영업의 세부 범<br>위)에서 정하는 동물위탁관리업자로써, 반려동물 소유자의 위탁을 받아 '
 '반려동물<br>을 영업장'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001250',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
