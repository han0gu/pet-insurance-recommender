from langchain_core.documents import Document

chunk = Document(
    page_content=('교정시력이 0.02이하가 된 경우(지급률 35%) ⇒ 장해지급률 35%에서 질병으로 인한 장해지급률 15%를 차감한 지급률 '
 '20%(=35%-15%)에 해당하는 후유장해보험금을 지급 | ① 보험가입 전 한 다리의 관절에 약간의 장해(지급률 5%)가 있었던 '
 '피보험자 가 보험가입 후 상해로 그 다리의 해당관절이 기능을 완전히 잃은 경우(지 급률 30%) ⇒ 보험가입 후 상해로 인한 '
 '장해지급률(30%)에서 보험가입 전 장해지급률 (5%)을 차감한 지급률 25%(=30%-5%)에 해당하는 후유장해보험금을 지급 ② '
 '보험가입 후 질병으로'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000259',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
