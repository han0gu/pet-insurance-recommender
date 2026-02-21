from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>･ 펫퍼민트 반려묘 입원의료비보장 특별약관<br>･ 펫퍼민트 반려묘 통원의료비보장 "
 '특별약관<br>･ 펫퍼민트 반려묘 입원의료비Ⅱ보장 특별약관<br>･ 펫퍼민트 반려묘 통원의료비Ⅱ보장 특별약관<br>･ 펫퍼민트 반려묘 '
 "입원의료비Ⅲ보장 특별약관<br>･ 펫퍼민트 반려묘 통원의료비Ⅲ보장 특별약관</p><br><p id='12' "
 "data-category='paragraph' style='font-size:16px'>\uf000 재가입 적용대상 특별약관이 다음 각 "
 '호의 조건을 충족<br>하고 계약자가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000362',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
