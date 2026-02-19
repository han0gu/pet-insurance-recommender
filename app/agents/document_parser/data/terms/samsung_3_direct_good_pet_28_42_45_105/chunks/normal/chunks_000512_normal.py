from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약ㆍ예방 접종비용 및 정기검 진, 예방적 검사를 위한 비용 2. 임신, '
 '출산(제왕절개를 포함합니다.), 인공유산과 관련된 비용 및 출산 후 증상 치 료 비용 3. 중성화, 불임 및 피임을 목적으로 한 수술 및 '
 '처치에 따른 비용 4. 산후 문제행동, 수유에 따르는 칼슘 부족에 의한 경련 및 기타 임신ㆍ출산과 관련 된 질병 치료에 대한 비용 5'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 82},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000512',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
