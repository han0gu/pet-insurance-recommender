from langchain_core.documents import Document

chunk = Document(
    page_content=('(강아지)【갱신계약】(【갱신계약】은 자동갱신으로 운영합니다)# 제1조(보험금의 지급사유)\uf000 회사는 피보험자가 이 특별약관의 '
 '보험기간 중에 상해의 직접결과로써 생활기능\n'
 '또는 업무능력에 지장을 가져와 병원 또는 의원(한방병원 또는 한의원을 포함합니\n'
 '다)에 입원하여 치료를 받은 경우에는 입원기간 동안 보험증권에 기재된 반려동물\n'
 '을 수탁기관에 위탁함으로써 발생한 위탁비용을 반려동물 위탁비용으로 보험수익\n'
 '자에게 지급합니다.\n'
 '\uf000 제1항의 "수탁기관"이라 함은 동물보호법 시행규칙 제43조(등록영업의 세부 범'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000717',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
