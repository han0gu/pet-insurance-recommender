from langchain_core.documents import Document

chunk = Document(
    page_content=('계약자가 그 보험금 지급사유가 발생한 사실을 알지 못<br>한 경우에는 청약철회의 효력은 발생하지 않습니다.<br>\uf000 제1항에서 '
 "보험증권을 받은 날에 대한 다툼이 발생한 경<br>우 회사가 이를 증명하여야 합니다.</p><p id='82' "
 "data-category='paragraph' style='font-size:20px'>제21조(약관교부 및 설명의무 "
 "등)</p><br><p id='83' data-category='paragraph' style='font-size:16px'>\uf000 "
 '회사는 계약자가 청약할 때에 계약자에게 약관의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000130',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
