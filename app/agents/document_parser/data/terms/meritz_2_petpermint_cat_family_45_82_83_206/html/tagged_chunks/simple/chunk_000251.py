from langchain_core.documents import Document

chunk = Document(
    page_content=('등), 전화(음성녹취)<br>또는 전자문서 등으로 보장내용 및 가입금액, 보험료 변경<br>내역 및 변경 절차 등을 계약자에게 '
 "알립니다.</p><br><p id='48' data-category='paragraph' "
 "style='font-size:20px'>\uf000 제2항에 따라 계약내용을 변경하는 경우에는 보장내용,</p><footer "
 "id='49' style='font-size:14px'>80</footer><p id='50' "
 "data-category='paragraph' style='font-size:16px'>가입금액 및"),
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
 'indexing': {'chunk_id': 'chunk_000251',
              'chunk_char_len': 295,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
