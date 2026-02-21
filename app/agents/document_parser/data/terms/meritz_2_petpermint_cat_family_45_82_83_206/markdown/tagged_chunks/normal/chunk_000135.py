from langchain_core.documents import Document

chunk = Document(
    page_content=('경일의 15일 이전까지 서면(등기우편 등), 전화(음성녹취)\n'
 '또는 전자문서 등으로 보장내용 및 가입금액, 보험료 변경\n'
 '내역 및 변경 절차 등을 계약자에게 알립니다.\uf000 제2항에 따라 계약내용을 변경하는 경우에는 보장내용,80가입금액 및 납입보험료가 '
 '변경될 수 있으며, 계약내용 변\n'
 '경 시점 이후 잔여보험기간의 보장을 위한 재원인 계약자적\n'
 '립액 및 미경과보험료 정산으로 계약자가 추가로 납입하여\n'
 '야 할(또는 반환받을) 금액이 발생할 수 있습니다.\uf000 제2항에도 불구하고 계약자가 계약내용 변경을 원하지'),
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
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 277,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
