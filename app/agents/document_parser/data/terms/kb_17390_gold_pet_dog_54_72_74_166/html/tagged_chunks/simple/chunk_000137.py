from langchain_core.documents import Document

chunk = Document(
    page_content=("성립)</p><br><p id='163' data-category='list' style='font-size:16px'>\uf000 "
 '계약은 계약자의 청약과 회사의 승낙으로 이루어집니다.<br>\uf000 회사는 보험의 목적 및 피보험자가 계약에 적합하지 않은 경우에는 '
 '승낙을 거절하<br>거나 별도의 조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 법<br>등)을 붙여 승낙할 수 '
 '있습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000137',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
