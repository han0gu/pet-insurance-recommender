from langchain_core.documents import Document

chunk = Document(
    page_content=("청구)</h1><br><p id='13' data-category='paragraph' "
 "style='font-size:20px'>\uf000 보험수익자는 다음의 서류를 제출하고 보험금을 청구하<br>여야 "
 "합니다.</p><br><p id='14' data-category='list' style='font-size:20px'>① 청구서(회사 "
 '양식)<br>② 사고증명서(동물병원 진료비 영수증(진료 항목별 영수<br>금액 포함), 동물병원 진료기록부, X-ray 등 '
 "방사선</p><footer id='15'"),
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
 'indexing': {'chunk_id': 'chunk_000288',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
