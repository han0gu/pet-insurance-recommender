from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>140</footer><p id='34' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경<br>우, 제2항에 정하는 "
 "조치(마취 비용을 포함합니다.)에 대해<br>서는 보험금을 지급하지 않습니다.</p><h1 id='35' "
 "style='font-size:20px'>제3조(수술의 정의와 장소)</h1><br><p id='36' "
 "data-category='paragraph'"),
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
 'indexing': {'chunk_id': 'chunk_000652',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
