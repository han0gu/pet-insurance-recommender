from langchain_core.documents import Document

chunk = Document(
    page_content=('조건(보험가입금액 제한, 일부보장<br>제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있<br>습니다.</p><br><h1 '
 "id='61' style='font-size:18px'>【 보험가입금액 제한 】</h1><br><p id='62' "
 "data-category='paragraph' style='font-size:16px'>피보험자가 가입을 할 수 있는 최대 보험가입금액을 "
 "제<br>한하는 방법을 말합니다.</p><br><h1 id='63' style='font-size:18px'>【 일부보장 제외(부담보)"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
