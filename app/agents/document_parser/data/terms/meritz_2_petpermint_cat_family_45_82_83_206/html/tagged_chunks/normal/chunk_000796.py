from langchain_core.documents import Document

chunk = Document(
    page_content=("부담한 수술당일 치료비 410만원<br>·보험금 지급금액</p><br><p id='38' data-category='list' "
 "style='font-size:20px'>= [(410만원-3만원)×50%, 200만원] 중 적은금액<br>= 200만원(MRI,CT 및 "
 "내시경처치와 수술을 동시에<br>하더라도 수술한도로 지급)</p><br><p id='39' data-category='paragraph' "
 "style='font-size:16px'>\uf000 수술과 MRI,CT 및 내시경처치를 동일한 날에 시행한 경<br>우 수술한 날의 "
 '지급한도'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000796',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
