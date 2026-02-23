from langchain_core.documents import Document

chunk = Document(
    page_content=("성립)</h1><br><p id='83' data-category='paragraph' "
 "style='font-size:20px'>\uf000 이 특별약관은 계약자의 청약과 회사의 승낙으로 "
 '이루어<br>집니다.<br>\uf000 회사는 피보험자 또는 반려동물이 계약에 적합하지 않은<br>경우에는 승낙을 거절하거나 별도의 '
 '조건(보험가입금액 제<br>한, 일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여<br>승낙할 수 있습니다.</p><br><h1 '
 "id='84' style='font-size:20px'>【보험가입금액 제한】</h1><br><p"),
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
 'indexing': {'chunk_id': 'chunk_000340',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
