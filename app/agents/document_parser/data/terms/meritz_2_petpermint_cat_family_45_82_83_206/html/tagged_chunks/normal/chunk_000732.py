from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:20px'>\uf000 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경<br>우, 제2항에 정하는 "
 "조치(마취 비용을 포함합니다.)에 대해<br>서는 보험금을 지급하지 않습니다.</p><h1 id='39' "
 "style='font-size:20px'>제3조(수술의 정의와 장소)</h1><br><p id='40' "
 "data-category='paragraph' style='font-size:16px'>\uf000 이 특별약관에 있어서「수술」이라 함은 "
 '수의사가 치료<br>가 필요하다고 인정한 경우로서 수의사의 관리하에'),
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
 'indexing': {'chunk_id': 'chunk_000732',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
