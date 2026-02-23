from langchain_core.documents import Document

chunk = Document(
    page_content=("id='73' style='font-size:14px'>74</footer><h1 id='74' "
 "style='font-size:18px'>특별부활(효력회복) 청약을 승낙합니다.</h1><br><p id='75' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는 제1항의 통지를 지정된 "
 '보험수익자에게 하여야<br>합니다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000207',
              'chunk_char_len': 204,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
