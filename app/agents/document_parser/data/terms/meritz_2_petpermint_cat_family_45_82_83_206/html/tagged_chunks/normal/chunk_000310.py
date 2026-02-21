from langchain_core.documents import Document

chunk = Document(
    page_content=("합계액</p><br><p id='43' data-category='paragraph' "
 "style='font-size:16px'>\uf000 피보험자가 다른 계약에 대하여 보험금 청구를 포기한<br>경우에도 회사의 제1항에 "
 "따른 지급보험금 결정에는 영향을<br>미치지 않습니다.</p><h1 id='44' style='font-size:20px'>제7조(계약 "
 "전 알릴 의무)</h1><br><p id='45' data-category='paragraph' "
 "style='font-size:16px'>계약자 또는 피보험자는 청약할 때(진단계약의"),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000310',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
