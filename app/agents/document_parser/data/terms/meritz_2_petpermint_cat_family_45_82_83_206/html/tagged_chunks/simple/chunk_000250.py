from langchain_core.documents import Document

chunk = Document(
    page_content=('관련 법률의 개정 또는 폐지 등에 따라 약관에서 정한<br>보험금 지급사유의 판정이 불가능한 경우<br>③ 관련 법률의 개정 또는 폐지 '
 "등에 따라 계약유지 필요<br>가 없어지는 경우<br>④ 기타 금융위원회 등의 명령이 있는 경우</p><br><p id='47' "
 "data-category='paragraph' style='font-size:16px'>\uf000 회사는 제2항에 따라 계약이 변경되는 "
 '경우 계약내용 변<br>경일의 15일 이전까지 서면(등기우편 등), 전화(음성녹취)<br>또는 전자문서 등으로 보장내용 및 가입금액, '
 '보험료'),
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
 'indexing': {'chunk_id': 'chunk_000250',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
