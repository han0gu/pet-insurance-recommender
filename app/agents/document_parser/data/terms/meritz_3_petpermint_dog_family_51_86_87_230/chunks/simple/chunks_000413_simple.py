from langchain_core.documents import Document

chunk = Document(
    page_content=('없는 상해 또는 질병에 대한 치료 ④ 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 정기검진, 예방적 검사를 '
 '위한 비용 ⑤ 반려동물의 임신·출산, 제왕절개, 인공유산과 관련 된 비용 및 출산 후 증상 치료 비용 ⑥ 중성화, 불임 및 피임을 '
 '목적으로 한 수술 및 처치에'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000413',
              'chunk_char_len': 164,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
