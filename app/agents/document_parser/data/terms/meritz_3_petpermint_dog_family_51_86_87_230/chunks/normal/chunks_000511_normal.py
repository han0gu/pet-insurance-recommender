from langchain_core.documents import Document

chunk = Document(
    page_content=(': 파보 바이러스 감염, 디스템퍼 바이러스 감염, 파라 인플루엔자 감염, 전염성 간염, 아데노 바이러스 2 형 감염, 광견병, 코로나 '
 '바이러스 감염, 렙토스피 라 감염, 필라리아(심장사상충) 감염, 인플루엔자 감염\n'
 '③ 상병명을 알 수 없는 상해 또는 질병에 대한 치료 ④ 백신 접종비용 및 기타 질병예방을 위한 검사 또는 투약·예방 접종비용 및 '
 '정기검진, 예방적 검사를 위한 비용 ⑤ 반려동물의 임신·출산, 제왕절개, 인공유산과 관련'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 157},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000511',
              'chunk_char_len': 242,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
