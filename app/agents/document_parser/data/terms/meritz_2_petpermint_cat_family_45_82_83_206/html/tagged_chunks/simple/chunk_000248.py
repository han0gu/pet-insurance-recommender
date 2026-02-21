from langchain_core.documents import Document

chunk = Document(
    page_content=("id='43' data-category='paragraph' style='font-size:20px'>계약의 청약을 권유하기 위해 만든 "
 "자료 등을 말합니다.</p><h1 id='44' style='font-size:20px'>제44조(법령 등의 개정에 따른 계약내용의 "
 "변경)</h1><br><p id='45' data-category='paragraph' "
 "style='font-size:16px'>\uf000 회사는 보험금 지급사유 관련 법률이 개정된 경우에는<br>변경된 내용을 "
 '적용합니다.<br>\uf000 제1항에도 불구하고 다음 각 호 중 어느'),
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
 'indexing': {'chunk_id': 'chunk_000248',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
