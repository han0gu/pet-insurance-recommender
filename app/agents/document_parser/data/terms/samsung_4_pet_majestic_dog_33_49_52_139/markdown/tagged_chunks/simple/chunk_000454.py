from langchain_core.documents import Document

chunk = Document(
    page_content=("- 공휴일에 관한 규정' 에 따른 공휴일과 노동절을 제외합니다.\n"
 '# ⑤ 보험료 관련 용어1. 보험료: 손해를 보장하는데 필요한 보험료를 말합니다.⑥ (재가입형) 특별약관 재가입 관련 용어- 1. '
 '최초계약 : 최초로 체결되는 계약을 말합니다.\n'
 '- 2. 재가입계약 : 이 보험의 사업방법서에서 정한 재가입 절차에 따라 재가입된 계약을\n'
 '- 말합니다.\n'
 '# 제3조 (보험금의 지급사유)- ① 회사는 보험증권에 기재된 이 특별약관의 보험기간(이하 「보험기간」 이라 합니다) 중'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000454',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
