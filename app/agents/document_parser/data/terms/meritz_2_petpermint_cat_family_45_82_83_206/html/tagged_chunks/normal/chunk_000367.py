from langchain_core.documents import Document

chunk = Document(
    page_content=('수준, 재가입 절차 및 재가입 의사 여부를 확인<br>하는 내용 등을 서면(등기우편 등), 전화(음성녹음), 전자<br>문서, 휴대전화 '
 '문자메시지 또는 이에 준하는 전자적 의사<br>표시 등으로 알려드리고, 회사는 계약자의 재가입의사를 전<br>화(음성녹음), 직접 방문 '
 "또는 전자적 의사표시, 통신판매<br>계약의 경우 통신수단을 통해 확인합니다.</p><br><p id='16' "
 "data-category='paragraph' style='font-size:16px'>\uf000 계약자는 제4항에 따른 재가입안내와 "
 '재가입여부'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000367',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
