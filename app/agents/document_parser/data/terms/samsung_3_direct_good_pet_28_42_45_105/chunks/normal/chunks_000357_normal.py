from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 아래의 의료비 및 비용 또는 손해는 보상하지 않습니다. 1. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약ㆍ예방 '
 '접종비용 및 정기검\n'
 '진, 예방적 검사를 위한 비용\n'
 '2. 임신, 출산(제왕절개를 포함합니다.), 인공유산과 관련된 비용 및 출산 후 증상 치'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 68},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000357',
              'chunk_char_len': 151,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
