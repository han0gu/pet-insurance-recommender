from langchain_core.documents import Document

chunk = Document(
    page_content=('제한, 일부보장 제외, 보험금 삭감, 보험료 할증<br>약<br>등)을 붙여 승낙할 수 있습니다.<br>관<br>\uf000 회사는 '
 '계약의 청약을 받고, 제1회 보험료를 받은 경우에 건강진단을 받지 않<br>는 계약은 청약일, 진단계약은 진단일(재진단의 경우에는 최종 '
 '진단일)부터 30<br>일 이내에 승낙 또는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다.<br>그러나 30일 이내에 승낙 또는 '
 '거절의 통지가 없으면 승낙된 것으로 봅니다.<br>\uf000 회사가 제1회 보험료를 받고 승낙을 거절한 경우에는 거절통지와 함께 받은 '
 '금'),
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
 'indexing': {'chunk_id': 'chunk_000855',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
