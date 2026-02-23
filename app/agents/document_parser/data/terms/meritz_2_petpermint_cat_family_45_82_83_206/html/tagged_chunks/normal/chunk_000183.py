from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 진단계약에서 진단을 받지<br>않은 경우라도 상해로 보험금 지급사유가 발생하는 경<br>우에는 보장을 해드립니다.</p><h1 '
 "id='46' style='font-size:20px'>제27조(제2회 이후 보험료의 납입)</h1><br><p id='47' "
 "data-category='paragraph' style='font-size:16px'>계약자는 제2회 이후의 보험료를 납입기일까지 "
 '납입하여야<br>하며, 회사는 계약자가 보험료를 납입한 경우에는 영수증을<br>발행하여 드립니다'),
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
 'indexing': {'chunk_id': 'chunk_000183',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
