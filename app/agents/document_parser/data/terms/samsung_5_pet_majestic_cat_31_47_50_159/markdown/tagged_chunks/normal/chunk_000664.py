from langchain_core.documents import Document

chunk = Document(
    page_content=('원회의 명령에 따른 약관 개정으로 갱신계약의 보장내용이 변경되는 경우, 회사는 제2조\n'
 '제5항에도 불구하고 다음 각 호에 따라 계약자에게 안내합니다.1. 회사는 갱신전 계약의 보험기간이 만료되는 날 이전까지 중요사항 '
 '변경내역(갱신보\n'
 '험료 변경 제외), 자동갱신 의사를 확인하는 내용 등을 서면(등기우편 등), 전화(음\n'
 '성녹음), 전자문서(SMS 포함) 또는 이에 준하는 전자적 의사표시 등으로 2회 이상- 123 -알려드립니다.- 2. 회사는 계약자의 '
 '자동갱신 의사를 전화(음성녹음), 직접 방문 또는 전자적 의사표시'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000664',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
