from langchain_core.documents import Document

chunk = Document(
    page_content=('. 스케일링, 발치 등을 포함한 치아의 치과치료비용(단, 치아를 제외한 구강질환만 보<br>장하며 구강질환의 치료 목적임에도 치아에 '
 "행해지는 치료는 보장하지 않습니다)</p><br><p id='50' data-category='paragraph' "
 "style='font-size:14px'>③ 제2항에 정하는 조치에 다른 진료를 병행하여 실시한 경우, 제2항에 정하는 "
 "조치(마취<br>비용을 포함합니다.)에 대해서는 보험금을 지급하지 않습니다.</p><h1 id='51' "
 "style='font-size:14px'>제6조(입원의 정의와"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
