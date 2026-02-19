from langchain_core.documents import Document

chunk = Document(
    page_content=('① 관련 법률의 개정 또는 폐지 등에 따라 약관에서 정한 보험금 지급사유 판정기준이 변경되는 경우 ② 관련 법률의 개정 또는 폐지 등에 '
 '따라 약관에서 정한 보험금 지급사유의 판정이 불가능한 경우 ③ 관련 법률의 개정 또는 폐지 등에 따라 계약유지 필요 가 없어지는 경우 ④ '
 '기타 금융위원회 등의 명령이 있는 경우\n'
 '\uf000 회사는 제2항에 따라 계약이 변경되는 경우 계약내용 변 경일의 15일 이전까지 서면(등기우편 등), 전화(음성녹취) 또는 '
 '전자문서 등으로 보장내용 및 가입금액, 보험료 변경 내역 및 변경 절차 등을 계약자에게 알립니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 80},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000170',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
