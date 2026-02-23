from langchain_core.documents import Document

chunk = Document(
    page_content=("경우에는 제17조(알릴 의무 위반의 효과)가<br>적용됩니다.</p><p id='71' data-category='paragraph' "
 "style='font-size:16px'>제31조(강제집행 등으로 인하여 해지된 계약의 특별부활(효<br>력회복))</p><br><p "
 "id='72' data-category='paragraph' style='font-size:16px'>\uf000 회사는 계약자의 "
 '해약환급금 청구권에 대한 강제집행,<br>담보권실행, 국세 및 지방세 체납처분절차에 따라 계약이<br>해지된 경우 해지 당시의 '
 '보험수익자가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
