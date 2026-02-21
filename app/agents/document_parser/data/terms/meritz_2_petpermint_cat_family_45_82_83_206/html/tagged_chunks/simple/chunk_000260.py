from langchain_core.documents import Document

chunk = Document(
    page_content=("의한 지급보장)</p><br><p id='63' data-category='paragraph' "
 "style='font-size:16px'>회사가 파산 등으로 인하여 보험금 등을 지급하지 못할 경<br>우에는 예금자보호법에서 정하는 "
 "바에 따라 그 지급을 보장<br>합니다.</p><br><h1 id='64' "
 "style='font-size:20px'>【예금자보호제도】</h1><br><p id='65' "
 "data-category='paragraph' style='font-size:16px'>예금자보호제도란 예금보험공사가 평소에"),
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
 'indexing': {'chunk_id': 'chunk_000260',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
